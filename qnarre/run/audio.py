# Copyright 2021 Quantapix Authors. All Rights Reserved.
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
# fine-tune Wav2Vec2 for audio classification

import warnings
from dataclasses import dataclass, field
from random import randint

import datasets
import numpy as np
from datasets import DatasetDict, load_dataset

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
)


def random_subsample(wav: np.ndarray, max_len: float, sample_rate=16000):
    sample_length = int(round(sample_rate * max_len))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    train_split_name = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to "
            "'validation'"
        },
    )
    max_length_seconds = field(
        default=20,
        metadata={
            "help": "Audio clips will be randomly cut to this length during training if the value is set."
        },
    )


add_argument("--model_name", type=str, default="facebook/wav2vec2-base", required=True)


@dataclass
class ModelArguments:
    freeze_feature_encoder = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    mask = field(
        default=True,
        metadata={"help": "Whether to generate an attention mask in the feature extractor."},
    )
    freeze_feature_extractor = field(
        default=None,
        metadata={"help": "Whether to freeze the feature extractor layers of the model."},
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder`"
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`."
                "Only make use of `--freeze_feature_encoder`."
            )


def main():
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name, data_args.dataset_config, split=data_args.train_split_name
    )
    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name, data_args.dataset_config, split=data_args.eval_split_name
    )

    if data_args.audio_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column {data_args.audio_column} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.label_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column {data_args.label_column} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor or model_args.model_name,
        return_attention_mask=model_args.mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_version,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column]:
            wav = random_subsample(
                audio["array"],
                max_len=data_args.max_length_seconds,
                sample_rate=feature_extractor.sampling_rate,
            )
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [label for label in batch[data_args.label_column]]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column]:
            wav = audio["array"]
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [label for label in batch[data_args.label_column]]

        return output_batch

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = raw_datasets["train"].features[data_args.label_column].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name,
        n_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetune="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_version,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name,
        from_tf=bool(".ckpt" in model_args.model_name),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_version,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kw = {
        "finetuned_from": model_args.model_name,
        "tasks": "audio-classification",
        "dataset": data_args.dataset_name,
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kw)
    else:
        trainer.create_model_card(**kw)


if __name__ == "__main__":
    main()
