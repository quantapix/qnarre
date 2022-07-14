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
        [],
        dict(
            act="gelu_new",
            attention_types=[[["global", "local"], 12]],
            BOS=50256,
            d_ff=None,
            d_hidden=2048,
            drop_attn=0.0,
            drop_embed=0.0,
            drop_resid=0.0,
            drop_sum_first=0.1,
            EOS=50256,
            eps=1e-5,
            init_range=0.02,
            model_type="gpt_neo",
            n_heads=16,
            n_lays=24,
            n_pos=2048,
            s_vocab=50257,
            s_win=256,
            sum_act=None,
            sum_proj=True,
            sum_type="cls_index",
            sum_use_proj=True,
            y_cache=True,
        ),
    )

    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions

    def _init_weights(self, module):
        if isinstance(module, (qc.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embed):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, qc.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTNeoModel):
            module.gradient_checkpointing = value


def custom_unfold(input, dimension, size, step):
    import torch

    shape = input.size()
    rank = len(shape)
    sizedim = shape[dimension]
    low_indices = torch.arange(0, sizedim, step)
    min_length = torch.div(sizedim - size, step, rounding_mode="floor") + 1
    indices = torch.arange(size) + low_indices[:min_length][:, None]
    s = [slice(None)] * rank
    s[dimension] = indices
    sliced = input[s]
    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))
    return sliced.permute(perm)


def custom_get_block_length_and_num_blocks(seq_length, s_win):
    import torch

    candidates = torch.arange(1, s_win)
    remainders = torch.remainder(seq_length, candidates)
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    largest_divisor = torch.max(divisors)
    return largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode="floor")


GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neo-1.3B": "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/config.json",
}
