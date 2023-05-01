import torch

from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType

from fused_softmax_lib import scaled_masked_softmax_forward, scaled_masked_softmax_backward
from fused_softmax_lib import scaled_masked_softmax_get_batch_per_block
from fused_softmax_lib import (
    scaled_upper_triang_masked_softmax_forward,
    scaled_upper_triang_masked_softmax_backward,
)

import torch.nn as nn
import enum
from ..fused_kernels import load_fused_kernels
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_upper_triang_masked_softmax_forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_upper_triang_masked_softmax_backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


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
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = scaled_masked_softmax_backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return ScaledMaskedSoftmax.apply(*args)


class FusedScaleMaskSoftmax(torch.nn.Module):
    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise RuntimeError("both fp16 and bf16 flags cannot be active at the same time.")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
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

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
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

    def forward_fused_softmax(self, input, mask):
        # input.shape = [b, np, sq, sk]
        scale = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        return scaled_masked_softmax_get_batch_per_block(sq, sk, b, np)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class SoftmaxFusionTypes(enum.Enum):
    upper_triang = 1  # causal mask
    general = 2  # general mask
    none = 3  # no fusion


class FusedScaleMaskSoftmax(nn.Module):
    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        fusion_type,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super().__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16

        assert fusion_type in [
            SoftmaxFusionTypes.upper_triang,
            SoftmaxFusionTypes.general,
            SoftmaxFusionTypes.none,
        ], f"Invalid fusion type {fusion_type}"

        if fusion_type != SoftmaxFusionTypes.none:
            load_fused_kernels()  # check fused kernels are installed

        self.upper_triang_mask_fusion = fusion_type == SoftmaxFusionTypes.upper_triang
        self.general_mask_fusion = fusion_type == SoftmaxFusionTypes.general
        self.fusion = fusion_type != SoftmaxFusionTypes.none

        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4
        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.fusion  # user wants to fuse
            and self.input_in_float16  # input must be fp16
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

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0
        if self.upper_triang_mask_fusion:
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            # input is 4D tensor (b, np, sq, sk)
            return ScaledMaskedSoftmax.apply(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    @staticmethod
    def get_batch_per_block(sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
