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
import copy


class PreTrained(qc.PreTrained):
    hs = qc.Hypers(
        [],
        dict(
            BOS=None,
            dataset_split="train",
            dataset="wiki_dpr",
            decoder_start_token_id=None,
            do_deduplication=True,
            do_marginalize=False,
            doc_sep=" // ",
            EOS=None,
            exclude_bos_score=False,
            forced_eos_token_id=None,
            index_name="compressed",
            index_path=None,
            is_composition=True,
            is_enc_dec=True,
            label_smoothing=0.0,
            max_combined_length=300,
            model_type="rag",
            n_docs=5,
            output_retrieved=False,
            PAD=None,
            passages_path=None,
            prefix=None,
            reduce_loss=False,
            retrieval_batch_size=8,
            retrieval_vector_size=768,
            s_vocab=None,
            title_sep=" / ",
            use_dummy_dataset=False,
            y_cache=True,
        ),
    )

    @classmethod
    def from_pretrained(cls, *args, **kw):
        kw["_fast_init"] = False
        return super().from_pretrained(*args, **kw)

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path=None,
        generator_pretrained_model_name_or_path=None,
        retriever=None,
        **kw,
    ):
        kwargs_question_encoder = {
            argument[len("question_encoder_") :]: value
            for argument, value in kw.items()
            if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_") :]: value
            for argument, value in kw.items()
            if argument.startswith("generator_")
        }

        # remove question_encoder, generator kw from kw
        for key in kwargs_question_encoder.keys():
            del kw["question_encoder_" + key]
        for key in kwargs_generator.keys():
            del kw["generator_" + key]
        question_encoder = kwargs_question_encoder.pop("model", None)
        if question_encoder is None:
            assert question_encoder_pretrained_model_name_or_path is not None
            from ..auto.modeling_auto import AutoModel

            if "config" not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig

                question_encoder_config, kwargs_question_encoder = AutoConfig.from_pretrained(
                    question_encoder_pretrained_model_name_or_path,
                    **kwargs_question_encoder,
                    return_unused_kwargs=True,
                )
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder
            )

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert generator_pretrained_model_name_or_path is not None
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config, kwargs_generator = AutoConfig.from_pretrained(
                    generator_pretrained_model_name_or_path,
                    **kwargs_generator,
                    return_unused_kwargs=True,
                )

                kwargs_generator["config"] = generator_config

            generator = AutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path, **kwargs_generator
            )

        # instantiate config with corresponding kw
        config = kw.get("config", None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kw
            )

        return cls(
            question_encoder=question_encoder,
            generator=generator,
            config=config,
            retriever=retriever,
        )

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config, generator_config, **kw
    ):
        return cls(
            question_encoder=question_encoder_config.to_dict(),
            generator=generator_config.to_dict(),
            **kw,
        )

    def to_dict(self):
        y = copy.deepcopy(self.__dict__)
        y["question_encoder"] = self.question_encoder.to_dict()
        y["generator"] = self.generator.to_dict()
        y["model_type"] = self.__class__.model_type
        return y
