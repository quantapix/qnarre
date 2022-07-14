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

from collections import OrderedDict

from ... import core as qc


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            act_fun="gelu",
            BOS=0,
            d_dec_ffn=4096,
            d_enc_ffn=4096,
            d_model=1024,
            drop_act=0.0,
            drop_attn=0.0,
            drop_dec=0.0,
            drop_enc=0.0,
            drop_proj=0.0,
            drop=0.1,
            EOS=2,
            forced_EOS=2,
            grad_checkpoint=True,
            init_std=0.02,
            is_enc_dec=True,
            model_type="mbart",
            n_dec_heads=16,
            n_dec_lays=12,
            n_enc_heads=16,
            n_enc_lays=12,
            n_pos=1024,
            PAD=1,
            s_vocab=50265,
            scale=False,
            y_cache=True,
        ),
    )

    def _init_weights(self, module):
        std = self.cfg.init_std
        if isinstance(module, qc.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, qc.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_grad_checkpoint(self, module, value=False):
        if isinstance(module, (MBartDecoder, MBartDecoder)):
            module.grad_checkpoint = value

    @property
    def dummy_inputs(self):
        pad = self.cfg.PAD
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad]], device=self.device)
        dummy_inputs = {
            "mask": input_ids.ne(pad),
            "input_ids": input_ids,
        }
        return dummy_inputs


MAP = {
    "facebook/mbart-large-cc25": dict(
        add_bias_logits=False,
        add_final_norm=True,
        archs=["MBartForConditionalGeneration"],
        id2label={"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},
        label2id={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        max_len=1024,
        n_beams=5,
        n_lays=12,
        normalize_embedding=True,
        num_labels=3,
        pre_norm=True,
        s_vocab=250027,
        scale=True,
        static_position_embeddings=False,
        task_params={"translation_en_to_ro": {"dec_START": 250020}},
        y_prev=True,
    ),
}


class Onnx:
    @property
    def inputs(self):
        if self.task in ["default", "seq2seq-lm"]:
            y = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            if self.use_past:
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
        elif self.task == "causal-lm":
            y = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            if self.use_past:
                n_enc_lays, _ = self.n_lays
                for i in range(n_enc_lays):
                    y[f"prev_kv.{i}.key"] = {
                        0: "batch",
                        2: "past_sequence + sequence",
                    }
                    y[f"prev_kv.{i}.value"] = {
                        0: "batch",
                        2: "past_sequence + sequence",
                    }
        else:
            y = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("mask", {0: "batch", 1: "encoder_sequence"}),
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),
                    ("dec_m", {0: "batch", 1: "decoder_sequence"}),
                ]
            )

        return y

    @property
    def outputs(self):
        if self.task in ["default", "seq2seq-lm"]:
            y = super().outputs
        else:
            y = super().outputs
            if self.use_past:
                n_enc_lays, _ = self.n_lays
                for i in range(n_enc_lays):
                    y[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    y[f"present.{i}.value"] = {
                        0: "batch",
                        2: "past_sequence + sequence",
                    }
        return y

    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer,
        batch_size=-1,
        seq_length=-1,
        is_pair=False,
        framework=None,
    ):
        encoder_inputs = (
            self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size, seq_length, is_pair, framework
            )
        )
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = (
            self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size, decoder_seq_length, is_pair, framework
            )
        )
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        y = dict(**encoder_inputs, **decoder_inputs)
        if self.use_past:
            batch, encoder_seq_length = y["input_ids"].shape
            decoder_seq_length = y["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.n_heads
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.d_model // num_encoder_attention_heads,
            )
            decoder_past_length = decoder_seq_length + 3
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.d_model // num_decoder_attention_heads,
            )
            y["dec_m"] = torch.cat(
                [y["dec_m"], torch.ones(batch, decoder_past_length)],
                dim=1,
            )
            y["prev_kv"] = []
            n_enc_lays, n_dec_lays = self.n_lays
            min_num_layers = min(n_enc_lays, n_dec_lays)
            max_num_layers = max(n_enc_lays, n_dec_lays) - min_num_layers
            remaining_side_name = "encoder" if n_enc_lays > n_dec_lays else "decoder"
            for _ in range(min_num_layers):
                y["prev_kv"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                y["prev_kv"].append((torch.zeros(shape), torch.zeros(shape)))
        return y

    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer,
        batch_size=-1,
        seq_length=-1,
        is_pair=False,
        framework=None,
    ):
        y = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )
        if self.use_past:
            batch, seqlen = y["input_ids"].shape
            past_key_values_length = seqlen + 2
            n_enc_lays, _ = self.n_lays
            num_encoder_attention_heads, _ = self.n_heads
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.d_model // num_encoder_attention_heads,
            )

            y["mask"] = torch.cat([y["mask"], torch.ones(batch, past_key_values_length)], dim=1)
            y["prev_kv"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(n_enc_lays)
            ]
        return y

    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer,
        batch_size=-1,
        seq_length=-1,
        is_pair=False,
        framework=None,
    ):
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.DEFAULT_FIXED_BATCH, num_token_to_add=0
        )
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length,
            fixed_dimension=OnnxConfig.DEFAULT_FIXED_SEQUENCE,
            num_token_to_add=token_to_add,
        )
        dummy_input = [" ".join([tokenizer.unk]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer,
        batch_size=-1,
        seq_length=-1,
        is_pair=False,
        framework=None,
    ):
        if self.task in ["default", "seq2seq-lm"]:
            y = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer,
                batch_size=batch_size,
                seq_length=seq_length,
                is_pair=is_pair,
                framework=framework,
            )

        elif self.task == "causal-lm":
            y = self._generate_dummy_inputs_for_causal_lm(
                tokenizer,
                batch_size=batch_size,
                seq_length=seq_length,
                is_pair=is_pair,
                framework=framework,
            )
        else:
            y = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer,
                batch_size=batch_size,
                seq_length=seq_length,
                is_pair=is_pair,
                framework=framework,
            )
        return y

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
