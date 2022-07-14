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

import json
import os
import re
import fairseq
import torch

from collections import OrderedDict
from argparse import ArgumentParser
from os.path import basename, dirname
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers.file_utils import WEIGHTS_NAME
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FS
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import logging

from ..config.fsmt import PreTrained
from ...run.fsmt import ForConditionalGen


logging.set_verbosity_warning()

json_indent = 2

# based on the results of a search on a range of `n_beams`, `len_penalty` and `early_stop`
# values against wmt19 test data to obtain the best BLEU scores, we will use the following defaults:
#
# * `n_beams`: 5 (higher scores better, but requires more memory/is slower, can be adjusted by users)
# * `early_stop`: `False` consistently scored better
# * `len_penalty` varied, so will assign the best one depending on the model
best_score_hparams = {
    # fairseq:
    "wmt19-ru-en": {"len_penalty": 1.1},
    "wmt19-en-ru": {"len_penalty": 1.15},
    "wmt19-en-de": {"len_penalty": 1.0},
    "wmt19-de-en": {"len_penalty": 1.1},
    # allenai:
    "wmt16-en-de-dist-12-1": {"len_penalty": 0.6},
    "wmt16-en-de-dist-6-1": {"len_penalty": 0.6},
    "wmt16-en-de-12-1": {"len_penalty": 0.8},
    "wmt19-de-en-6-6-base": {"len_penalty": 0.6},
    "wmt19-de-en-6-6-big": {"len_penalty": 0.6},
}

org_names = {}
for m in ["wmt19-ru-en", "wmt19-en-ru", "wmt19-en-de", "wmt19-de-en"]:
    org_names[m] = "facebook"
for m in [
    "wmt16-en-de-dist-12-1",
    "wmt16-en-de-dist-6-1",
    "wmt16-en-de-12-1",
    "wmt19-de-en-6-6-base",
    "wmt19-de-en-6-6-big",
]:
    org_names[m] = "allenai"


def rewrite_dict_keys(d):
    d2 = dict(
        (re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v)
        for k, v in d.items()
    )
    keep_keys = "<s> <pad> </s> <unk>".split()
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k]  # restore
    return d2


def to_pytorch(fsmt_checkpoint_path, save_path):
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(save_path, exist_ok=True)
    print(f"Writing results to {save_path}")
    checkpoint_file = basename(fsmt_checkpoint_path)
    fsmt_folder_path = dirname(fsmt_checkpoint_path)
    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    models = cls.hub_models()
    kw = {"bpe": "fastbpe", "tokenizer": "moses"}
    data_name_or_path = "."
    print(f"using checkpoint {checkpoint_file}")
    chkpt = hub_utils.from_pretrained(
        fsmt_folder_path, checkpoint_file, data_name_or_path, archive_map=models, **kw
    )
    args = vars(chkpt["args"]["model"])
    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]
    data_root = dirname(save_path)
    model_dir = basename(save_path)
    src_dict_file = os.path.join(fsmt_folder_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_folder_path, f"dict.{tgt_lang}.txt")
    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    s_src_vocab = len(src_vocab)
    src_vocab_file = os.path.join(save_path, "vocab-src.json")
    print(f"Generating {src_vocab_file} of {s_src_vocab} of {src_lang} records")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))
    do_lower_case = True
    for k in src_vocab.keys():
        if not k.islower():
            do_lower_case = False
            break
    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    s_tgt_vocab = len(tgt_vocab)
    tgt_vocab_file = os.path.join(save_path, "vocab-tgt.json")
    print(f"Generating {tgt_vocab_file} of {s_tgt_vocab} of {tgt_lang} records")
    with open(tgt_vocab_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))
    merges_file = os.path.join(save_path, VOCAB_FS["merges_file"])
    for fn in ["bpecodes", "code"]:  # older fairseq called the merges file "code"
        fsmt_merges_file = os.path.join(fsmt_folder_path, fn)
        if os.path.exists(fsmt_merges_file):
            break
    with open(fsmt_merges_file, encoding="utf-8") as fin:
        merges = fin.read()
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    print(f"Generating {merges_file}")
    with open(merges_file, "w", encoding="utf-8") as fout:
        fout.write(merges)
    fsmt_model_config_file = os.path.join(save_path, "config.json")
    assert args["bpe"] == "fastbpe", f"need to extend tokenizer to support bpe={args['bpe']}"
    assert (
        args["tokenizer"] == "moses"
    ), f"need to extend tokenizer to support bpe={args['tokenizer']}"

    model_conf = {
        "archs": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "drop_act": args["drop_act"],
        "act_fun": "relu",
        "drop_attn": args["drop_attn"],
        "d_hidden": args["decoder_embed_dim"],
        "drop": args["drop"],
        "init_std": 0.02,
        "n_pos": args["max_source_positions"],
        "n_lays": args["n_enc_lays"],
        "s_src_vocab": s_src_vocab,
        "s_tgt_vocab": s_tgt_vocab,
        "langs": [src_lang, tgt_lang],
        "n_enc_heads": args["n_enc_heads"],
        "d_enc_ffn": args["encoder_ffn_embed_dim"],
        "drop_enc": args["drop_enc"],
        "n_enc_lays": args["n_enc_lays"],
        "n_dec_heads": args["n_dec_heads"],
        "d_dec_ffn": args["decoder_ffn_embed_dim"],
        "drop_dec": args["drop_dec"],
        "n_dec_lays": args["n_dec_lays"],
        "BOS": 0,
        "PAD": 1,
        "EOS": 2,
        "is_enc_dec": True,
        "scale": not args["no_scale_embedding"],
        "tie_word_embeds": args["share_all_embeddings"],
    }
    model_conf["n_beams"] = 5
    model_conf["early_stop"] = False
    if model_dir in best_score_hparams and "len_penalty" in best_score_hparams[model_dir]:
        model_conf["len_penalty"] = best_score_hparams[model_dir]["len_penalty"]
    else:
        model_conf["len_penalty"] = 1.0
    print(f"Generating {fsmt_model_config_file}")
    with open(fsmt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))
    fsmt_tokenizer_config_file = os.path.join(save_path, TOKENIZER_CONFIG_FILE)
    tokenizer_conf = {
        "langs": [src_lang, tgt_lang],
        "model_max_length": 1024,
        "do_lower_case": do_lower_case,
    }
    print(f"Generating {fsmt_tokenizer_config_file}")
    with open(fsmt_tokenizer_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_conf, ensure_ascii=False, indent=json_indent))
    model = chkpt["models"][0]
    model_state_dict = model.state_dict()
    model_state_dict = OrderedDict(("model." + k, v) for k, v in model_state_dict.items())
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        "model.encoder_embed_tokens.weight",
        "model.decoder_embed_tokens.weight",
        "model.encoder.embed_positions._float_tensor",
        "model.decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)
    config = PreTrained.from_pretrained(save_path)
    model_new = ForConditionalGen(config)
    model_new.load_state_dict(model_state_dict, strict=False)
    pytorch_weights_dump_path = os.path.join(save_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)
    print("Conversion is done!")
    print("\nLast step is to upload the files to s3")
    print(f"cd {data_root}")
    print(f"transformers-cli upload {model_dir}")


if __name__ == "__main__":
    x = ArgumentParser()
    x.add_argument("--src_path", default=None, type=str, required=True)
    x.add_argument("--save_path", default=None, type=str, required=True)
    args = x.parse_args()
    to_pytorch(args.src_path, args.save_path)
