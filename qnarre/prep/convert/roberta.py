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

import pathlib
import torch

from argparse import ArgumentParser
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncLayer

from transformers.utils import logging

from ..config.bert import PreTrained
from ...models.bert import ForMasked, ForSeqClassifier


logging.set_verbosity_info()

log = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def to_pytorch(src_path, save_path, classification_head):
    roberta = FairseqRobertaModel.from_pretrained(src_path)
    roberta.eval()  # disable drop
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    cfg = PreTrained(
        s_vocab=roberta_sent_encoder.embed_tokens.num_embeddings,
        d_hidden=roberta.args.encoder_embed_dim,
        n_lays=roberta.args.n_enc_lays,
        n_heads=roberta.args.n_enc_heads,
        d_ffnet=roberta.args.encoder_ffn_embed_dim,
        n_pos=514,
        n_typ=1,
        norm_eps=1e-5,
    )
    if classification_head:
        cfg.n_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", cfg)
    m = ForSeqClassifier(cfg) if classification_head else ForMasked(cfg)
    m.eval()
    m.roberta.embeddings.tok_embed.weight = roberta_sent_encoder.embed_tokens.weight
    m.roberta.embeddings.pos_embed.weight = roberta_sent_encoder.embed_positions.weight
    m.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        m.roberta.embeddings.token_type_embeddings.weight
    )
    m.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    m.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias
    for i in range(cfg.n_lays):
        layer = m.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncLayer = roberta_sent_encoder.layers[i]
        self_attn = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((cfg.d_hidden, cfg.d_hidden))
        )
        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias
        self_output = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias
        intermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias
        bert_output = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
    if classification_head:
        m.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        m.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        m.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        m.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        m.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        m.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        m.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        m.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        m.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        m.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias
    input_ids = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1
    our_output = m(input_ids)[0]
    if classification_head:
        their_output = roberta.model.classification_heads["mnli"](
            roberta.extract_features(input_ids)
        )
    else:
        their_output = roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {save_path}")
    m.save_pretrained(save_path)


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--roberta_checkpoint_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    x.add_argument("--classification_head", action="store_true")
    y = x.parse_args()
    to_pytorch(y.roberta_checkpoint_path, y.save_path, y.classification_head)
