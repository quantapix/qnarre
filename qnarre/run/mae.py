from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTMAEConfig,
    ViTMAEForPreTraining,
)

add_argument("--dataset_name", type=str, default="cifar10")


@dataclass
class DataTrainingArguments:
    image_column_name = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_dir = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir = field(
        default=None, metadata={"help": "A folder containing the validation data."}
    )
    train_val_split = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    mask_ratio = field(
        default=0.75,
        metadata={"help": "The ratio of the number of masked tokens in the input sequence."},
    )
    norm_pix_loss = field(
        default=True,
        metadata={"help": "Whether or not to train with normalized pixel values as target."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_lr = field(
        default=1e-3,
        metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."},
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    return {"pixel_values": pixel_values}


def main():
    ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Load pretrained model and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kw = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_version,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kw)
    elif model_args.model_name:
        config = ViTMAEConfig.from_pretrained(model_args.model_name, **config_kw)
    else:
        config = ViTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
        }
    )

    # create feature extractor
    if model_args.feature_extractor:
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            model_args.feature_extractor, **config_kw
        )
    elif model_args.model_name:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.model_name, **config_kw)
    else:
        feature_extractor = ViTFeatureExtractor()

    # create model
    if model_args.model_name:
        model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name,
            from_tf=bool(".ckpt" in model_args.model_name),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_version,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model")
        model = ViTMAEForPreTraining(config)

    if training_args.do_train:
        column_names = ds["train"].column_names
    else:
        column_names = ds["validation"].column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
    transforms = Compose(
        [
            Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            RandomResizedCrop(
                feature_extractor.size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ]
    )

    def preprocess_images(examples):
        """Preprocess a batch of images by applying transforms."""

        examples["pixel_values"] = [transforms(image) for image in examples[image_column_name]]
        return examples

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = (
                ds["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        ds["train"].set_transform(preprocess_images)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_images)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.grad_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_lr is not None:
        training_args.lr = training_args.base_lr * total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
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
        "tasks": "masked-auto-encoding",
        "dataset": data_args.dataset_name,
        "tags": ["masked-auto-encoding"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kw)
    else:
        trainer.create_model_card(**kw)


if __name__ == "__main__":
    main()
