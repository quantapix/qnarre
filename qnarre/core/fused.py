import torch
import torch.nn as nn
import enum
import torch
import torch.nn.functional as F

from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType

from fused_softmax_lib import scaled_masked_softmax_forward, scaled_masked_softmax_backward
from fused_softmax_lib import scaled_masked_softmax_get_batch_per_block
from fused_softmax_lib import (
    scaled_upper_triang_masked_softmax_forward,
    scaled_upper_triang_masked_softmax_backward,
)

from ..fused_kernels import load_fused_kernels

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add(x, bias, residual, p, training):
    y = F.dropout(x + bias, p=p, training=training)
    if residual is not None:
        y = residual + y
    return y


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, p):
        return bias_dropout_add(x, bias, residual, p, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, p):
    return bias_dropout_add(x, bias, residual, p, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, p):
    return bias_dropout_add(x, bias, residual, p, False)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        s = torch.tensor([scale])
        y = scaled_upper_triang_masked_softmax_forward(x, s[0])
        ctx.save_for_backward(y, s)
        return y

    @staticmethod
    def backward(ctx, x):
        y, s = ctx.saved_tensors
        y = scaled_upper_triang_masked_softmax_backward(x, y, s[0])
        return y, None


def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, "causal mask is only for self attention"
    # Reshaping input to 3D tensor (attn_batches, sq, sk)
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.cuda.amp.autocast(enabled=False):
        probs = ScaledUpperTriangMaskedSoftmax.apply(*args)
    return probs.view(b, np, sq, sk)


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, scale):
        s = torch.tensor([scale])
        y = scaled_masked_softmax_forward(x, mask, s[0])
        ctx.save_for_backward(y, s)
        return y

    @staticmethod
    def backward(ctx, x):
        y, s = ctx.saved_tensors
        y = scaled_masked_softmax_backward(x, y, s[0])
        return y, None, None


def scaled_masked_softmax(inputs, mask, scale):
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return ScaledMaskedSoftmax.apply(*args)


class FusedScaleMaskSoftmax(torch.nn.Module):
    def __init__(
        self,
        x_fp16,
        x_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.x_fp16 = x_fp16
        self.x_bf16 = x_bf16
        if self.x_fp16 and self.x_bf16:
            raise RuntimeError("both fp16 and bf16 flags cannot be active at the same time.")
        self.x_float16 = self.x_fp16 or self.x_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        if not (self.scale is None or softmax_in_fp32):
            raise RuntimeError("softmax should be in fp32 when scaled")
        if self.scaled_masked_softmax_fusion:
            if self.attn_mask_type == AttnMaskType.causal:
                self.fused_softmax_func = scaled_upper_triang_masked_softmax
            elif self.attn_mask_type == AttnMaskType.padding:
                self.fused_softmax_func = scaled_masked_softmax
            else:
                raise ValueError("Invalid attn_mask_type.")

    def forward(self, x, mask):
        # [b, np, sq, sk]
        assert x.dim() == 4
        if self.is_kernel_available(mask, *x.size()):
            return self.forward_fused_softmax(x, mask)
        else:
            return self.forward_torch_softmax(x, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.x_float16  # input must be fp16
            and (
                self.attn_mask_type == AttnMaskType.causal
                or (self.attn_mask_type == AttnMaskType.padding and mask is not None)
            )
            and 16 < sk <= 8192  # sk must be 16 ~ 8192
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 8192:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, x, mask):
        # input.shape = [b, np, sq, sk]
        s = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(x, mask, s)

    def forward_torch_softmax(self, x, mask):
        if self.x_float16 and self.softmax_in_fp32:
            x = x.float()
        if self.scale is not None:
            x = x * self.scale
        y = torch.nn.Softmax(dim=-1)(self.mask_func(x, mask) if mask is not None else x)
        if self.x_float16 and self.softmax_in_fp32:
            if self.x_fp16:
                y = y.half()
            else:
                y = y.bfloat16()
        return y

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return scaled_masked_softmax_get_batch_per_block(sq, sk, b, np)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        import scaled_upper_triang_masked_softmax_cuda

        s = torch.tensor([scale])
        y = scaled_upper_triang_masked_softmax_cuda.forward(x, s[0])
        ctx.save_for_backward(y, s)
        return y

    @staticmethod
    def backward(ctx, x):
        import scaled_upper_triang_masked_softmax_cuda

        y, s = ctx.saved_tensors
        y = scaled_upper_triang_masked_softmax_cuda.backward(x, y, s[0])
        return y, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, scale):
        import scaled_masked_softmax_cuda

        s = torch.tensor([scale])
        y = scaled_masked_softmax_cuda.forward(x, mask, s[0])
        ctx.save_for_backward(y, s)
        return y

    @staticmethod
    def backward(ctx, x):
        import scaled_masked_softmax_cuda

        y, s = ctx.saved_tensors
        y = scaled_masked_softmax_cuda.backward(x, y, s[0])
        return y, None, None


class SoftmaxFusionTypes(enum.Enum):
    upper_triang = 1  # causal mask
    general = 2  # general mask
    none = 3  # no fusion


class FusedScaleMaskSoftmax(nn.Module):
    def __init__(
        self,
        x_fp16,
        x_bf16,
        fusion_type,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.x_fp16 = x_fp16
        self.x_bf16 = x_bf16
        self.x_float16 = self.x_fp16 or self.x_bf16
        assert fusion_type in [
            SoftmaxFusionTypes.upper_triang,
            SoftmaxFusionTypes.general,
            SoftmaxFusionTypes.none,
        ]
        if fusion_type != SoftmaxFusionTypes.none:
            load_fused_kernels()
        self.upper_triang_mask_fusion = fusion_type == SoftmaxFusionTypes.upper_triang
        self.general_mask_fusion = fusion_type == SoftmaxFusionTypes.general
        self.fusion = fusion_type != SoftmaxFusionTypes.none
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, x, mask):
        # [b, np, sq, sk]
        assert x.dim() == 4
        if self.is_kernel_available(mask, *x.size()):
            return self.forward_fused_softmax(x, mask)
        else:
            return self.forward_torch_softmax(x, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np
        if (
            self.fusion  # user wants to fuse
            and self.x_float16  # input must be fp16
            and mask is not None  # mask tensor must not be None
            and 16 < sk <= 2048  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 2048:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)
                if self.upper_triang_mask_fusion:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_fused_softmax(self, x, mask):
        b, np, sq, sk = x.size()
        s = self.scale if self.scale is not None else 1.0
        if self.upper_triang_mask_fusion:
            assert sq == sk, "causal mask is only for self attention"
            y = ScaledUpperTriangMaskedSoftmax.apply(x.view(-1, sq, sk), s)
            return y.view(b, np, sq, sk)
        else:
            return ScaledMaskedSoftmax.apply(x, mask, s)

    def forward_torch_softmax(self, x, mask):
        if self.x_float16 and self.softmax_in_fp32:
            x = x.float()
        if self.scale is not None:
            x = x * self.scale
        y = torch.nn.Softmax(dim=-1)(self.mask_func(x, mask) if mask is not None else x)
        if self.x_float16 and self.softmax_in_fp32:
            if self.x_fp16:
                y = y.half()
            else:
                y = y.bfloat16()
        return y

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
