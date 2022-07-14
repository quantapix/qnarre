# Copyright 2022 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        {"n_dec_lays"},
        dict(
            d_ff=2048,
            d_kv=64,
            d_model=512,
            drop_rate=0.1,
            EOS=1,
            eps=1e-6,
            feed_forward_proj="relu",
            grad_checkpoint=True,
            init_factor=1.0,
            is_enc_dec=True,
            is_parallelizable=True,
            model_type="t5",
            n_heads=8,
            n_lays=6,
            PAD=0,
            relative_attention_num_buckets=32,
            s_vocab=32128,
            y_cache=True,
        ),
    )

    def __init__(self, **kw):
        self.n_dec_lays = n_dec_lays if n_dec_lays is not None else self.n_lays
        super().__init__(PAD=PAD, EOS=EOS, is_enc_dec=is_enc_dec, **kw)

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "dec_m": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        factor = self.cfg.initializer_factor  # Used for testing weights initialization
        if isinstance(module, LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (Model, ForConditionalGeneration, EncoderModel)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, DenseReluDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.cfg.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.cfg.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.cfg.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.cfg.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.cfg.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Attention):
            d_model = self.cfg.d_model
            key_value_proj_dim = self.cfg.d_kv
            n_heads = self.cfg.n_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
            )
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, (Attention, Stack)):
            module.grad_checkpoint = value

    def _shift_right(self, input_ids):
        dec_START = self.cfg.dec_START
        PAD = self.cfg.PAD
        assert (
            dec_START is not None
        ), "self.model.config.dec_START has to be defined. In  it is usually set to the PAD. See  docs for more information"
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), dec_START)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = dec_START
        assert PAD is not None, "self.model.config.PAD has to be defined."
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, PAD)
        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids


MAP = {
    "t5-small": dict(
        archs=["LMHead"],
        dec_START=0,
        n_pos=512,
        eps=1e-06,
        y_prev=True,
        task_params=dict(
            summarization=dict(
                early_stop=True,
                len_penalty=2.0,
                max_len=200,
                min_len=30,
                s_no_repeat_ngram=3,
                n_beams=4,
                prefix="summarize: ",
            ),
            translation_en_to_de=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to German: ",
            ),
            translation_en_to_ro=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to Romanian: ",
            ),
        ),
    ),
    "t5-base": dict(
        archs=["LMHead"],
        d_ff=3072,
        d_model=768,
        dec_START=0,
        n_heads=12,
        n_lays=12,
        n_pos=512,
        eps=1e-06,
        y_prev=True,
        task_params=dict(
            summarization=dict(
                early_stop=True,
                len_penalty=2.0,
                max_len=200,
                min_len=30,
                s_no_repeat_ngram=3,
                n_beams=4,
                prefix="summarize: ",
            ),
            translation_en_to_de=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to German: ",
            ),
            translation_en_to_ro=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to Romanian: ",
            ),
        ),
    ),
    "t5-large": dict(
        archs=["LMHead"],
        d_ff=4096,
        d_model=1024,
        dec_START=0,
        n_heads=16,
        n_lays=24,
        n_pos=512,
        eps=1e-06,
        y_prev=True,
        task_params=dict(
            summarization=dict(
                early_stop=True,
                len_penalty=2.0,
                max_len=200,
                min_len=30,
                s_no_repeat_ngram=3,
                n_beams=4,
                prefix="summarize: ",
            ),
            translation_en_to_de=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to German: ",
            ),
            translation_en_to_ro=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to Romanian: ",
            ),
        ),
    ),
    "t5-3b": dict(
        archs=["LMHead"],
        d_ff=16384,
        d_model=1024,
        d_kv=128,
        dec_START=0,
        n_heads=32,
        n_lays=24,
        n_pos=512,
        eps=1e-06,
        y_prev=True,
        task_params=dict(
            summarization=dict(
                early_stop=True,
                len_penalty=2.0,
                max_len=200,
                min_len=30,
                s_no_repeat_ngram=3,
                n_beams=4,
                prefix="summarize: ",
            ),
            translation_en_to_de=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to German: ",
            ),
            translation_en_to_ro=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to Romanian: ",
            ),
        ),
    ),
    "t5-11b": dict(
        archs=["LMHead"],
        d_ff=65536,
        d_model=1024,
        d_kv=128,
        dec_START=0,
        n_heads=128,
        n_lays=24,
        n_pos=512,
        eps=1e-06,
        y_prev=True,
        task_params=dict(
            summarization=dict(
                early_stop=True,
                len_penalty=2.0,
                max_len=200,
                min_len=30,
                s_no_repeat_ngram=3,
                n_beams=4,
                prefix="summarize: ",
            ),
            translation_en_to_de=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to German: ",
            ),
            translation_en_to_ro=dict(
                early_stop=True,
                max_len=300,
                n_beams=4,
                prefix="translate English to Romanian: ",
            ),
        ),
    ),
}


class Onnx:
    @property
    def inputs(self):
        y = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "mask": {0: "batch", 1: "encoder_sequence"},
        }
        if self.use_past:
            y["mask"][1] = "past_encoder_sequence + sequence"
            y["decoder_input_ids"] = {0: "batch"}
            y["dec_m"] = {
                0: "batch",
                1: "past_decoder_sequence + sequence",
            }
        else:
            y["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            y["dec_m"] = {0: "batch", 1: "decoder_sequence"}
        if self.use_past:
            self.fill_with_past_key_values_(y, direction="inputs")
        return y

    @property
    def default_onnx_opset(self):
        return 13
