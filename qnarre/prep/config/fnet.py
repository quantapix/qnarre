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

from collections import OrderedDict

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            act="gelu_new",
            BOS=1,
            d_ff=3072,
            d_hidden=768,
            drop=0.1,
            EOS=2,
            eps=1e-12,
            init_range=0.02,
            model_type="fnet",
            n_lays=12,
            n_pos=512,
            n_typ=4,
            PAD=3,
            s_vocab=32000,
            tpu_short_seq_length=512,
            use_tpu_fourier_optimizations=False,
            grad_checkpoint=True,
        ),
    )

    def _init_weights(self, module):
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
            # NOTE: Original code uses same initialization as weights for biases as well.
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
        if isinstance(module, FNetEncoder):
            module.gradient_checkpointing = value


MAP = {
    "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/config.json",
}
