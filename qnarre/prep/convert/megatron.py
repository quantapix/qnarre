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

import numpy as np
import re
import tensorflow as tf
import torch

from argparse import ArgumentParser
from os.path import abspath
from transformers.utils import logging

from ..config.megatron import PreTrained
from ...models.megatron import ForPreTraining

import os
import re
import zipfile

logging.set_verbosity_info()

log = logging.get_logger(__name__)


def load_src_weights(model, config, tf_checkpoint_path):
    tf_path = abspath(tf_checkpoint_path)
    log.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        log.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            log.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    log.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        if pointer.shape != array.shape:
            raise ValueError(
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            )
        log.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def recursive_print(name, val, spaces=0):
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, n_heads, d_hidden):
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [n_heads * d_hidden * num_splits, :]
        saved_shape = (n_heads, d_hidden, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [n_heads * num_splits * d_hidden, :]
        saved_shape = (n_heads, num_splits, d_hidden) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def convert_megatron_checkpoint(args, input_state_dict, config):
    output_state_dict = {}
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        config.tokenizer_type = ds_args.tokenizer_type
        config.s_vocab = ds_args.padded_vocab_size
        config.n_pos = ds_args.n_pos
        config.d_hidden = ds_args.d_hidden
        config.n_lays = ds_args.n_lays
        config.n_heads = ds_args.n_heads
        config.d_ff = (
            ds_args.ffn_hidden_size if "ffn_hidden_size" in ds_args else 4 * ds_args.d_hidden
        )
    heads = config.n_heads
    hidden_size_per_head = config.d_hidden // heads
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to s_vocab rows.
    word_embeddings = word_embeddings[: config.s_vocab, :]
    # Store the word embeddings.
    output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

    # The position embeddings.
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    assert pos_embeddings.size(0) == config.n_pos and pos_embeddings.size(1) == config.d_hidden
    # Store the position embeddings.
    output_state_dict["bert.embeddings.position_embeddings.weight"] = pos_embeddings

    # The token-type embeddings.
    tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
    # Store the position embeddings.
    output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attention.output.dense.",
        "mlp.dense_h_to_4h": ".intermediate.dense.",
        "mlp.dense_4h_to_h": ".output.dense.",
    }

    # Keep track of the attention/query/value tensor.
    attention_qkv_weight = None

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"bert.encoder.layer.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):

            ln_name = "attention.ln" if op_name.startswith("input") else "ln"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Transpose the QKV matrix.
        elif op_name == "attention.query_key_value" and weight_or_bias == "weight":

            # Make sure the QKV pointer is nil.
            assert attention_qkv_weight is None, ""

            out_val = fix_query_key_value_ordering(
                val, checkpoint_version, 3, heads, hidden_size_per_head
            )
            # Store the tensor as we need the bias as well to interleave QKV and biases.
            attention_qkv_weight = out_val

        # Transpose the bias.
        elif op_name == "attention.query_key_value" and weight_or_bias == "bias":

            # Make sure we read the weight tensor.
            assert attention_qkv_weight is not None, ""

            # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
            q = attention_qkv_weight[0 * config.d_hidden : 1 * config.d_hidden, :]
            k = attention_qkv_weight[1 * config.d_hidden : 2 * config.d_hidden, :]
            v = attention_qkv_weight[2 * config.d_hidden : 3 * config.d_hidden, :]

            out_val = fix_query_key_value_ordering(
                val, checkpoint_version, 3, heads, hidden_size_per_head
            )
            # Split the bias.
            q_bias = out_val[0 * config.d_hidden : 1 * config.d_hidden]
            k_bias = out_val[1 * config.d_hidden : 2 * config.d_hidden]
            v_bias = out_val[2 * config.d_hidden : 3 * config.d_hidden]

            # Store.
            output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
            output_state_dict[f"{layer_name}.attention.self.query.bias"] = q_bias
            output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
            output_state_dict[f"{layer_name}.attention.self.key.bias"] = k_bias
            output_state_dict[f"{layer_name}.attention.self.value.weight"] = v
            output_state_dict[f"{layer_name}.attention.self.value.bias"] = v_bias

            # Clear the stored tensor.
            attention_qkv_weight = None

        # Copy weights and biases as is.
        elif weight_or_bias in ["weight", "bias"]:

            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + weight_or_bias] = val

    # The final layernorm.
    output_state_dict["bert.encoder.ln.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["bert.encoder.ln.bias"] = transformer["final_layernorm.bias"]

    # The pooler.
    pooler = lm["pooler"]

    # Store the matrix and the bias.
    output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
    output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

    # The LM head from Megatron (for RACE).
    lm_head = model["lm_head"]

    # The transform matrix.
    output_state_dict["cls.predictions.transform.dense.weight"] = lm_head["dense.weight"]
    output_state_dict["cls.predictions.transform.dense.bias"] = lm_head["dense.bias"]

    # The transform LN.
    output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head["layernorm.weight"]
    output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head["layernorm.bias"]

    # For the decoder, we replicate the weights.
    output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
    output_state_dict["cls.predictions.bias"] = lm_head["bias"]

    # The classifier from Megatron (for MLNI).
    binary_head = model["binary_head"]

    # Store the classifier.
    output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
    output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

    # It should be done!
    return output_state_dict


def main():
    parser = ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "path_to_checkpoint", type=str, help="Path to the ZIP file containing the checkpoint"
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    args = parser.parse_args()
    basename = os.path.dirname(args.path_to_checkpoint)
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")
    if args.config_file == "":
        config = PreTrained()
        config.s_vocab = input_state_dict["model"]["lm_head"]["bias"].numel()
    else:
        config = PreTrained.from_json_file(args.config_file)
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    print("Saving config")
    config.save_pretrained(basename)
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


if __name__ == "__main__":
    main()
