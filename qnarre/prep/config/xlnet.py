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

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        ["reuse_len"],
        dict(
            act_ffnet="gelu",
            act_sum="tanh",
            attn_type="bi",
            bi_data=False,
            BOS=1,
            clamp_len=-1,
            d_ff=4096,
            d_head=64,
            drop_sum_last=0.1,
            drop=0.1,
            end_n_top=5,
            EOS=2,
            eps=1e-12,
            init_range=0.02,
            mem_len=512,
            model_type="xlnet",
            n_lays=24,
            PAD=5,
            reuse_len=None,
            s_vocab=32000,
            same_length=False,
            start_n_top=5,
            sum_type="last",
            sum_use_proj=True,
            task_params={"text-generation": {"do_sample": True, "max_len": 250}},
            untie_r=True,
            use_mems_eval=True,
            use_mems_train=False,
        ),
    )

    def __init__(self, d_model=1024, n_heads=16, **kw):
        if d_model % n_heads != 0:
            raise ValueError(f"'d_model % n_heads' ({d_model % n_heads}) should be equal to 0")
        if "d_head" in kw:
            if kw["d_head"] != d_model // n_heads:
                raise ValueError(
                    f"`d_head` ({kw['d_head']}) should be equal to `d_model // n_heads` ({d_model // n_heads})"
                )
        self.d_head = d_model // n_heads
        if "y_cache" in kw:
            use_mems_eval = kw["y_cache"]
        super().__init__(PAD=PAD, BOS=BOS, EOS=EOS, **kw)

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RelativeAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.o,
                module.r,
                module.r_r_bias,
                module.r_s_bias,
                module.r_w_bias,
                module.seg_embed,
            ]:
                param.data.normal_(mean=0.0, std=self.cfg.init_range)
        elif isinstance(module, Model):
            module.mask_emb.data.normal_(mean=0.0, std=self.cfg.init_range)


MAP = {
    "xlnet-base-cased": dict(
        archs=["LMHead"],
        d_ff=3072,
        d_model=768,
        mem_len=None,
        n_heads=12,
        n_lays=12,
    ),
    "xlnet-large-cased": dict(
        archs=["LMHead"],
        d_model=1024,
        mem_len=None,
        n_heads=16,
    ),
}
