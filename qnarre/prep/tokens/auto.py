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

import importlib
import json
import os
from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module
from ...file_utils import get_file_from_repo
from ...core.tokens.utils import PreTrainedTokenizer
from ...core.tokens.base import TOKENIZER_CONFIG_FILE
from ...core.tokens.fast import PreTrainedTokenizerFast
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
)

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("albert", ("AlbertTokenizer", "AlbertTokenizerFast")),
    ]
)

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def tokenizer_class_from_name(class_name):
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            return getattr(module, class_name)

    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    return None


def get_tokenizer_config(
    pretrained_model_name_or_path,
    cache_dir=None,
    force_download=False,
    resume_download=False,
    proxies=None,
    use_auth_token=None,
    revision=None,
    local_files_only=False,
    **kw,
):
    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        use_auth_token=use_auth_token,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved_config_file is None:
        logger.info(
            "Could not locate the tokenizer configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoTokenizer:
    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kw):
        config = kw.pop("config", None)
        kw["_from_auto"] = True

        use_fast = kw.pop("use_fast", True)
        tokenizer_type = kw.pop("tokenizer_type", None)
        trust_remote_code = kw.pop("trust_remote_code", False)

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_tuple is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES.keys())}."
                )

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast and tokenizer_fast_class_name is not None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)

            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_name} is not currently imported."
                )

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kw)

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kw)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kw
                )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        # If we have the tokenizer class from the tokenizer config or the model config we're good!
        if config_tokenizer_class is not None:
            tokenizer_class = None
            if tokenizer_auto_map is not None:
                if not trust_remote_code:
                    raise ValueError(
                        f"Loading {pretrained_model_name_or_path} requires you to execute the tokenizer file in that repo "
                        "on your local machine. Make sure you have read the code there to avoid malicious use, then set "
                        "the option `trust_remote_code=True` to remove this error."
                    )
                if kw.get("revision", None) is None:
                    logger.warning(
                        "Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure "
                        "no malicious code has been contributed in a newer revision."
                    )

                if use_fast and tokenizer_auto_map[1] is not None:
                    class_ref = tokenizer_auto_map[1]
                else:
                    class_ref = tokenizer_auto_map[0]

                module_file, class_name = class_ref.split(".")
                tokenizer_class = get_class_from_dynamic_module(
                    pretrained_model_name_or_path, module_file + ".py", class_name, **kw
                )

            elif use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)

            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kw)

        # Otherwise we have to be creative.
        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warning(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class__}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]
            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(
                    pretrained_model_name_or_path, *inputs, **kw
                )
            else:
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(
                        pretrained_model_name_or_path, *inputs, **kw
                    )
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING.keys())}."
        )

    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None):
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError(
                "You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class"
            )
        if slow_tokenizer_class is not None and issubclass(
            slow_tokenizer_class, PreTrainedTokenizerFast
        ):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        if fast_tokenizer_class is not None and issubclass(
            fast_tokenizer_class, PreTrainedTokenizer
        ):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        if (
            slow_tokenizer_class is not None
            and fast_tokenizer_class is not None
            and issubclass(fast_tokenizer_class, PreTrainedTokenizerFast)
            and fast_tokenizer_class.slow_tokenizer_class != slow_tokenizer_class
        ):
            raise ValueError(
                "The fast tokenizer class you are passing has a `slow_tokenizer_class` attribute that is not "
                "consistent with the slow tokenizer class you passed (fast tokenizer has "
                f"{fast_tokenizer_class.slow_tokenizer_class} and you passed {slow_tokenizer_class}. Fix one of those "
                "so they match!"
            )

        # Avoid resetting a set slow/fast tokenizer if we are passing just the other ones.
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class))
