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

from argparse import ArgumentParser
from torch import nn

from ..config.prophetnet import PreTrained
from ...models.prophetnet import ForCondGen

from transformers import (
    ForCondGen,
    XLMForCondGen,
    logging,
)

from transformers_old.modeling_prophetnet import (
    ForCondGen as ForCondGenOld,
)
from transformers_old.modeling_xlm_prophetnet import (
    XLMForCondGen as XLMForCondGenOld,
)


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def to_pytorch(src_path, save_path):
    if "xprophetnet" in src_path:
        prophet_old = XLMForCondGenOld.from_pretrained(src_path)
        prophet, loading_info = XLMForCondGen.from_pretrained(src_path, output_loading_info=True)
    else:
        prophet_old = ForCondGenOld.from_pretrained(src_path)
        prophet, loading_info = ForCondGen.from_pretrained(src_path, output_loading_info=True)
    special_keys = ["key_proj", "value_proj", "query_proj"]
    mapping = {
        "self_attn": "ngram_self_attn",
        "cross_attn": "encoder_attn",
        "cross_attn_layer_norm": "encoder_attn_layer_norm",
        "feed_forward_layer_norm": "final_layer_norm",
        "feed_forward": "",
        "intermediate": "fc1",
        "output": "fc2",
        "key_proj": "k_proj",
        "query_proj": "q_proj",
        "value_proj": "v_proj",
        "word_embeddings": "embed_tokens",
        "embeddings_layer_norm": "emb_layer_norm",
        "relative_pos_embeddings": "relative_linear",
        "ngram_embeddings": "ngram_input_embed",
        "position_embeddings": "embed_positions",
    }
    for key in loading_info["missing_keys"]:
        attributes = key.split(".")
        if attributes[0] == "lm_head":
            model = prophet
            old_model = prophet_old
        else:
            model = prophet.prophetnet
            old_model = prophet_old.model
        is_key_init = False
        for attribute in attributes:
            if attribute in mapping:
                old_attribute = mapping[attribute]
                if not hasattr(old_model, old_attribute) and len(old_attribute) > 0:
                    old_attribute = attribute
            elif hasattr(old_model, attribute):
                old_attribute = attribute

            if attribute == "weight":
                assert old_model.weight.shape == model.weight.shape, "Shapes have to match!"
                model.weight = old_model.weight
                logger.info(f"{attribute} is initialized.")
                is_key_init = True
                break
            elif attribute == "bias":
                assert old_model.bias.shape == model.bias.shape, "Shapes have to match!"
                model.bias = old_model.bias
                logger.info(f"{attribute} is initialized")
                is_key_init = True
                break
            elif attribute in special_keys and hasattr(old_model, "in_proj_weight"):
                embed_dim = old_model.in_proj_weight.shape[0] // 3
                param = getattr(model, attribute)
                param.weight.shape == old_model.in_proj_weight[
                    :embed_dim, :
                ].shape, "Shapes have to match"
                param.bias.shape == old_model.in_proj_bias[:embed_dim].shape, "Shapes have to match"
                if attribute == "query_proj":
                    model.query_proj.weight = nn.Parameter(old_model.in_proj_weight[:embed_dim, :])
                    model.query_proj.bias = nn.Parameter(old_model.in_proj_bias[:embed_dim])

                elif attribute == "key_proj":
                    model.key_proj.weight = nn.Parameter(
                        old_model.in_proj_weight[embed_dim : 2 * embed_dim, :]
                    )
                    model.key_proj.bias = nn.Parameter(
                        old_model.in_proj_bias[embed_dim : 2 * embed_dim]
                    )
                elif attribute == "value_proj":
                    model.value_proj.weight = nn.Parameter(
                        old_model.in_proj_weight[2 * embed_dim :, :]
                    )
                    model.value_proj.bias = nn.Parameter(old_model.in_proj_bias[2 * embed_dim :])
                is_key_init = True
                break
            elif attribute == "position_embeddings":
                assert (
                    model.position_embeddings.weight.shape[-1]
                    == old_model.embed_positions.weight.shape[-1]
                ), "Hidden size has to match"
                assert (
                    model.position_embeddings.weight.shape[0] == 512
                ), "We want 512 position_embeddings."
                model.position_embeddings.weight = nn.Parameter(
                    old_model.embed_positions.weight[:512, :]
                )
                is_key_init = True
                break
            if attribute.isdigit():
                model = model[int(attribute)]
                old_model = old_model[int(old_attribute)]
            else:
                model = getattr(model, attribute)

                if old_attribute == "":
                    old_model = old_model
                else:
                    if not hasattr(old_model, old_attribute):
                        raise ValueError(f"{old_model} does not have {old_attribute}")
                    old_model = getattr(old_model, old_attribute)
        if not is_key_init:
            raise ValueError(f"{key} was not correctly initialized!")
    print(f"Saving to: {save_path}")
    prophet.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    y = x.parse_args()
    to_pytorch(y.src_path, y.save_path)
