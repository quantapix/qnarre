import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import psutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from requests import HTTPError

from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from .dynamic_module_utils import custom_object_save
from .file_utils import (
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    EntryNotFoundError,
    ModelOutput,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    copy_func,
    has_file,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
)
from .generation_utils import GenerationMixin
from .utils import logging
from .utils.versions import require_version_core


logger = logging.get_logger(__name__)


_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    global _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = True


def get_parameter_device(x):
    try:
        return next(x.parameters()).device
    except StopIteration:

        def find_tensor_attributes(x):
            return [(k, v) for k, v in x.__dict__.items() if torch.is_tensor(v)]

        gen = x._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


class ModuleUtilsMixin:
    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kw):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kw):
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (
            module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0
        )
        return None

    def add_memory_hooks(self):
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable=False, exclude_embeddings=False):
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter
                for name, parameter in self.named_parameters()
                if name not in embedding_param_names
            ]
            return sum(
                p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable
            )
        else:
            return sum(
                p.numel() for p in self.parameters() if p.requires_grad or not only_trainable
            )

    def estimate_tokens(self, input_dict):
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        else:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            return 0

    def floating_point_ops(self, input_dict, exclude_embeddings=True):
        return (
            6
            * self.estimate_tokens(input_dict)
            * self.num_parameters(exclude_embeddings=exclude_embeddings)
        )


class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None

    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = None
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = None
    # a list of of tensor names to ignore when saving the model (useful for keys that aren't
    # trained, but which are deterministic, or tied variables)
    _keys_to_ignore_on_save = None

    is_parallelizable = False
    grad_checkpoint = False

    @property
    def dummy_inputs(self):
        """
        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self):
        """
        :str: Identifies that this is a PyTorch model.
        """
        return "pt"

    def __init__(self, config, *inputs, **kw):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_grad_checkpoint()

    def _backward_compatibility_grad_checkpoint(self):
        if self.grad_checkpoint and getattr(self.config, "grad_checkpoint", False):
            self.grad_checkpoint_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "grad_checkpoint")

    @classmethod
    def _from_config(cls, config, **kw):
        torch_dtype = kw.pop("torch_dtype", None)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kw)
        else:
            model = cls(config, **kw)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype):
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None

    def _init_weights(self, module):
        raise NotImplementedError(f"Make sure `_init_weights` is implemented for {self.__class__}")

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_enc_dec", False) and getattr(
            self.config, "tie_encoder_decoder", False
        ):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(encoder, decoder, base_model_prefix):
        uninitialized_encoder_weights = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer,
            encoder_pointer,
            module_name,
            uninitialized_encoder_weights,
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = set(
                    [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
                )
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(
                            decoder_modules[decoder_name], type(encoder_modules[encoder_name])
                        ) and len(encoder_modules) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, uninitialized_encoder_weights
        )
        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.s_vocab = new_num_tokens
        self.s_vocab = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens=None):
        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. "
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(self.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens=None,
        transposed=False,
    ):
        if new_num_tokens is None:
            return old_lm_head

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. "
                f"You should either use a different resize function or make sure that `old_lm_head` are an instance of {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (
            (old_lm_head_dim, new_num_tokens)
            if not transposed
            else (new_num_tokens, old_lm_head_dim)
        )
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(self.device, dtype=old_lm_head.weight.dtype)

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # XXX: put the long block of code in a wrapper
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    # Copy old lm head weights to new lm head
                    if not transposed:
                        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
                            :num_tokens_to_copy, :
                        ]
                    else:
                        new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
                            :, :num_tokens_to_copy
                        ]

                    # Copy bias weights to new lm head
                    if has_new_lm_head_bias:
                        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
                            :num_tokens_to_copy
                        ]
        else:
            # Copy old lm head weights to new lm head
            if not transposed:
                new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
                    :num_tokens_to_copy, :
                ]
            else:
                new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
                    :, :num_tokens_to_copy
                ]

            # Copy bias weights to new lm head
            if has_new_lm_head_bias:
                new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
                    :num_tokens_to_copy
                ]

        return new_lm_head

    def resize_position_embeddings(self, new_num_position_embeddings):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self):
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(
                union_heads
            )  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def grad_checkpoint_enable(self):
        if not self.grad_checkpoint:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_grad_checkpoint, value=True))

    def grad_checkpoint_disable(self):
        if self.grad_checkpoint:
            self.apply(partial(self._set_grad_checkpoint, value=False))

    @property
    def is_grad_checkpoint(self):
        return any(hasattr(m, "grad_checkpoint") and m.grad_checkpoint for m in self.modules())

    def save_pretrained(
        self,
        save_directory,
        save_config=True,
        state_dict=None,
        save_function=torch.save,
        push_to_hub=False,
        **kw,
    ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kw.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kw)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.archs = [model_to_save.__class__.__name__]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kw):
        config = kw.pop("config", None)
        state_dict = kw.pop("state_dict", None)
        cache_dir = kw.pop("cache_dir", None)
        from_tf = kw.pop("from_tf", False)
        from_flax = kw.pop("from_flax", False)
        ignore_mismatched_sizes = kw.pop("ignore_mismatched_sizes", False)
        force_download = kw.pop("force_download", False)
        resume_download = kw.pop("resume_download", False)
        proxies = kw.pop("proxies", None)
        output_loading_info = kw.pop("output_loading_info", False)
        local_files_only = kw.pop("local_files_only", False)
        use_auth_token = kw.pop("use_auth_token", None)
        revision = kw.pop("revision", None)
        mirror = kw.pop("mirror", None)
        from_pipeline = kw.pop("_from_pipeline", None)
        from_auto_class = kw.pop("_from_auto", False)
        _fast_init = kw.pop("_fast_init", True)
        torch_dtype = kw.pop("torch_dtype", None)
        low_cpu_mem_usage = kw.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kw,
            )
        else:
            model_kwargs = kw

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index"
                    )
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                        "weights."
                    )
                elif os.path.join(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME):
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} but "
                        "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or "
                        f"{FLAX_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
                pretrained_model_name_or_path
            ):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                else:
                    filename = WEIGHTS_NAME

                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=filename,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )

            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                    "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                    "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                    "login` and pass `use_auth_token=True`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                    "this model name. Check the model page at "
                    f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                if filename == WEIGHTS_NAME:
                    has_file_kwargs = {
                        "revision": revision,
                        "mirror": mirror,
                        "proxies": proxies,
                        "use_auth_token": use_auth_token,
                    }
                    if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                            "there is a file for TensorFlow weights. Use `from_tf=True` to load this model from those "
                            "weights."
                        )
                    elif has_file(
                        pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME} but "
                            "there is a file for Flax weights. Use `from_flax=True` to load this model from those "
                            "weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}, "
                            f"{TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                        )
                else:
                    raise EnvironmentError(
                        f"{pretrained_model_name_or_path} does not appear to have a file named {filename}."
                    )
            except HTTPError:
                raise EnvironmentError(
                    "We couldn't connect to 'https://huggingface.co/' to load this model and it looks like "
                    f"{pretrained_model_name_or_path} is not the path to a directory conaining a a file named "
                    f"{WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}.\n"
                    "Checkout your internet connection or see how to run the library in offline mode at "
                    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or "
                    f"{FLAX_WEIGHTS_NAME}."
                )

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(
                    f"loading weights file {archive_file} from cache at {resolved_archive_file}"
                )
        else:
            resolved_archive_file = None

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if state_dict is None:
                try:
                    state_dict = torch.load(resolved_archive_file, map_location="cpu")
                except Exception as e:
                    try:
                        with open(resolved_archive_file) as f:
                            if f.read().startswith("version"):
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please install "
                                    "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                                    "you cloned."
                                )
                            else:
                                raise ValueError from e
                    except (UnicodeDecodeError, ValueError):
                        raise OSError(
                            f"Unable to load weights from pytorch checkpoint file for '{pretrained_model_name_or_path}' "
                            f"at '{resolved_archive_file}'. "
                            "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
                        )

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        torch_dtype = next(iter(state_dict.values())).dtype
                    else:
                        raise ValueError(
                            f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            if low_cpu_mem_usage:
                # save the keys
                loaded_state_dict_keys = [k for k in state_dict.keys()]
                del state_dict  # free CPU memory - will reload again later

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                with no_init_weights(_enable=_fast_init):
                    model = cls(config, *model_args, **model_kwargs)
        else:
            with no_init_weights(_enable=_fast_init):
                model = cls(config, *model_args, **model_kwargs)

        if from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6]
                )  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see "
                    "https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for installation instructions."
                )
                raise
        elif from_pt:

            if low_cpu_mem_usage:
                cls._load_state_dict_into_model_low_mem(
                    model, loaded_state_dict_keys, resolved_archive_file
                )
            else:
                (
                    model,
                    missing_keys,
                    unexpected_keys,
                    mismatched_keys,
                    error_msgs,
                ) = cls._load_state_dict_into_model(
                    model,
                    state_dict,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    _fast_init=_fast_init,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model

    @classmethod
    def _load_state_dict_into_model(
        cls,
        model,
        state_dict,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        _fast_init=True,
    ):

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        loaded_keys = list(state_dict.keys())
        prefix = model.base_model_prefix

        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(prefix)]
            expected_keys = [
                ".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys
            ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
        # matching the weights in the model.
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                    model_key = f"{prefix}.{checkpoint_key}"
                elif add_prefix_to_model:
                    # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                    model_key = ".".join(checkpoint_key.split(".")[1:])

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    mismatched_keys.append(
                        (
                            checkpoint_key,
                            state_dict[checkpoint_key].shape,
                            model_state_dict[model_key].shape,
                        )
                    )
                    del state_dict[checkpoint_key]

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if _fast_init:
            # retrieve unintialized modules and initialize
            uninitialized_modules = model.retrieve_modules_from_names(
                missing_keys, add_prefix=add_prefix_to_model, remove_prefix=remove_prefix_from_model
            )
            for module in uninitialized_modules:
                model._init_weights(module)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(
                    list(module.parameters(recurse=False)), modifier_rank=0
                ):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        beg_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            beg_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are training to load is corrupted. Are you sure it was "
                    "properly saved?"
                )

        load(model_to_load, prefix=beg_prefix)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized because the shapes did not match:\n{mismatched_warning}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = set([".".join(key.split(".")[:-1]) for key in names])

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            set([".".join(key.split(".")[:-2]) for key in names if key[-1].isdigit()])
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                name = (
                    ".".join(name.split(".")[1:])
                    if name.startswith(self.base_model_prefix)
                    else name
                )
            elif add_prefix:
                name = (
                    ".".join([self.base_model_prefix, name])
                    if len(name) > 0
                    else self.base_model_prefix
                )

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def _load_state_dict_into_model_low_mem(
        cls, model, loaded_state_dict_keys, resolved_archive_file
    ):
        require_version_core("torch>=1.9")
        if is_deepspeed_zero3_enabled():
            raise ValueError("low_cpu_mem_usage arg cannot be used with DeepSpeed ZeRO-3")

        # a helper util to find the last sub-module and the param/buffer name
        def find_submodule_and_param_name(model, long_key):
            split_key = long_key.split(".")
            submodule = model
            while len(split_key) > 1:
                if hasattr(submodule, split_key[0]):
                    submodule = getattr(submodule, split_key[0])
                    del split_key[0]
                else:
                    submodule = None
                    break
            return submodule, split_key[0]

        # dematerialize param storage for keys that are going to be replaced by state_dict, by
        # putting those on the meta device
        for k in loaded_state_dict_keys:
            submodule, param_name = find_submodule_and_param_name(model, k)
            if submodule is not None:
                # selectively switch to the meta device only those params/buffers that will
                # be next replaced from state_dict. This a complex way to do p.to_("meta")
                # since we have no in-place to_ for tensors.
                new_val = getattr(submodule, param_name)
                if isinstance(new_val, torch.nn.Parameter):
                    # isinstance returns False for Params on meta device, so switch after the check
                    new_val = torch.nn.Parameter(new_val.to("meta"))
                else:
                    new_val = new_val.to("meta")
                setattr(submodule, param_name, new_val)

        # only now can load state_dict
        state_dict = torch.load(resolved_archive_file, map_location="cpu")

        # materialize state_dict entries one by one on CPU
        for k in loaded_state_dict_keys:
            submodule, param_name = find_submodule_and_param_name(model, k)
            if submodule is not None:
                new_val = state_dict[k]
                if isinstance(getattr(submodule, param_name), torch.nn.Parameter):
                    new_val = torch.nn.Parameter(new_val)
                setattr(submodule, param_name, new_val)

        del state_dict

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
    object="model", object_class="AutoModel", object_files="model checkpoint"
)


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


@dataclass
class SquadHeadOutput(ModelOutput):
    loss = None
    top_beg_log_probs = None
    top_beg_index = None
    top_end_log_probs = None
    top_end_index = None
    cls_logits = None


class SQuADHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beg_n_top = config.beg_n_top
        self.end_n_top = config.end_n_top

        self.logits_beg = PoolerStartLogits(config)
        self.logits_end = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(
        self,
        hiddens,
        beg_positions=None,
        end_positions=None,
        cls_index=None,
        is_impossible=None,
        p_mask=None,
        return_dict=False,
    ):
        logits_beg = self.logits_beg(hiddens, p_mask=p_mask)

        if beg_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (beg_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            logits_end = self.logits_end(hiddens, beg_positions=beg_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            beg_loss = loss_fct(logits_beg, beg_positions)
            end_loss = loss_fct(logits_end, end_positions)
            total_loss = (beg_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(
                    hiddens, beg_positions=beg_positions, cls_index=cls_index
                )
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to beg_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hiddens.size()
            beg_log_probs = nn.functional.softmax(logits_beg, dim=-1)  # shape (bsz, slen)

            top_beg_log_probs, top_beg_index = torch.topk(
                beg_log_probs, self.beg_n_top, dim=-1
            )  # shape (bsz, beg_n_top)
            top_beg_index_exp = top_beg_index.unsqueeze(-1).expand(
                -1, -1, hsz
            )  # shape (bsz, beg_n_top, hsz)
            x_beg = torch.gather(hiddens, -2, top_beg_index_exp)  # shape (bsz, beg_n_top, hsz)
            x_beg = x_beg.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, beg_n_top, hsz)

            hidden_states_expanded = hiddens.unsqueeze(2).expand_as(
                x_beg
            )  # shape (bsz, slen, beg_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            logits_end = self.logits_end(hidden_states_expanded, x_beg=x_beg, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(logits_end, dim=1)  # shape (bsz, slen, beg_n_top)

            top_end_log_probs, top_end_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, beg_n_top)
            top_end_log_probs = top_end_log_probs.view(-1, self.beg_n_top * self.end_n_top)
            top_end_index = top_end_index.view(-1, self.beg_n_top * self.end_n_top)

            x_beg = torch.einsum("blh,bl->bh", hiddens, beg_log_probs)
            cls_logits = self.answer_class(hiddens, x_beg=x_beg, cls_index=cls_index)

            if not return_dict:
                return (
                    top_beg_log_probs,
                    top_beg_index,
                    top_end_log_probs,
                    top_end_index,
                    cls_logits,
                )
            else:
                return SquadHeadOutput(
                    top_beg_log_probs=top_beg_log_probs,
                    top_beg_index=top_beg_index,
                    top_end_log_probs=top_end_log_probs,
                    top_end_index=top_end_index,
                    cls_logits=cls_logits,
                )


class SequenceSummary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.summy_type = getattr(config, "summy_type", "last")
        if self.summy_type == "attn":
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, "sum_use_proj") and config.sum_use_proj:
            if hasattr(config, "sum_proj") and config.sum_proj and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.d_model
            self.summary = nn.Linear(config.d_model, num_classes)
        activation_string = getattr(config, "sum_act", None)
        self.activation = get_activation(activation_string) if activation_string else Identity()
        self.first_dropout = Identity()
        if hasattr(config, "drop_sum_first") and config.drop_sum_first > 0:
            self.first_dropout = nn.Dropout(config.drop_sum_first)
        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(self, hiddens, cls_index=None):
        if self.summy_type == "last":
            output = hiddens[:, -1]
        elif self.summy_type == "first":
            output = hiddens[:, 0]
        elif self.summy_type == "mean":
            output = hiddens.mean(dim=1)
        elif self.summy_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hiddens[..., :1, :],
                    hiddens.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hiddens.size(-1),))
            # shape of cls_index: (bsz, XX, 1, d_model) where XX are optional leading dim of hiddens
            output = hiddens.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, d_model)
        elif self.summy_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


def unwrap_model(model):
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
