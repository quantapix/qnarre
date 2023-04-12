import copy
import json
import os
import re
import torch

from requests import HTTPError

from transformers import custom_object_save
from transformers.utils import logging
from transformers.file_utils import (
    CONFIG_NAME,
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
)
from .base import Hypers, Module

log = logging.get_logger(__name__)


class PreTrained(Module):
    hs = Hypers(
        [
            "activation",
            "archs",
            "bad_words_ids",
            "BOS",
            "cross_attention_hidden_size",
            "dec_START",
            "EOS",
            "finetune",
            "forced_BOS",
            "forced_EOS",
            "id2label",
            "label2id",
            "n_labels",
            "PAD",
            "prefix",
            "problem",
            "remove_invalid_values",
            "SEP",
            "task_params",
            "tokenizer_class",
            "torch_dtype",
        ],
        {
            "add_cross": False,
            "chunk_ff": 0,
            "d_model": 0,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "drop": 0.0,
            "early_stop": False,
            "encoder_no_repeat_ngram_size": 0,
            "is_dec": False,
            "is_enc_dec": False,
            "len_penalty": 1.0,
            "max_len": 20,
            "min_len": 0,
            "min_len": 10,
            "n_beam_groups": 1,
            "n_beams": 1,
            "n_heads": 0,
            "n_lays": 0,
            "name_or_path": "",
            "num_return_sequences": 1,
            "out_dict_gen": False,
            "out_scores": False,
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "s_no_repeat_ngram": 0,
            "s_vocab": 0,
            "temperature": 1.0,
            "tie_encoder_decoder": False,
            "tie_word_embeds": True,
            "top_k": 50,
            "top_p": 1.0,
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": False,
            "y_attn": False,
            "y_kw": True,
            "y_hidden": False,
        },
    )

    def __init__(self, ps={}, hs=[], **kw):
        super().__init__(ps, [self.hs] + hs, **kw)
        cfg = self.get_cfg(kw)
        if cfg.id2label is not None:
            cfg.id2label = dict((int(k), v) for k, v in cfg.id2label.items())
        if cfg.torch_dtype is not None and isinstance(cfg.torch_dtype, str):
            cfg.torch_dtype = getattr(torch, cfg.torch_dtype)
        if cfg.problem is not None:
            assert cfg.problem in ("multi_label", "regression", "single_label")

    grad_checkpoint = True
    is_composition = False
    _auto_class = None

    @property
    def y_kw(self):
        return self.y_kw and not self.torchscript

    @property
    def n_labels(self):
        return len(self.id2label)

    @n_labels.setter
    def n_labels(self, x):
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != x:
            self.id2label = {i: f"LABEL_{i}" for i in range(x)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, path, **kw):
        assert not os.path.isfile(path)
        os.makedirs(path, exist_ok=True)
        if self._auto_class is not None:
            custom_object_save(self, path, config=self)
        y = os.path.join(path, CONFIG_NAME)
        self.to_json_file(y, use_diff=True)
        log.info(f"Config saved in {y}")

    @classmethod
    def from_pretrained(cls, path, **kw):
        y, kw = cls.get_config_dict(path, **kw)
        if "model_type" in y and hasattr(cls, "model_type") and y["model_type"] != cls.model_type:
            log.warning(f"Using {y['model_type']} to instantiate {cls.model_type}")
        return cls.from_dict(y, **kw)

    @classmethod
    def get_config_dict(cls, path, **kw):
        x = copy.deepcopy(kw)
        y, kw = cls._get_config_dict(path, **kw)
        if "configuration_files" in y:
            f = get_configuration_file(y["configuration_files"])
            y, kw = cls._get_config_dict(path, _configuration_file=f, **x)
        return y, kw

    @classmethod
    def _get_config_dict(cls, path, **kw):
        local_files_only = kw.pop("local_files_only", False)
        from_pipeline = kw.pop("_from_pipeline", None)
        user_agent = {"file_type": "config", "from_auto_class": kw.pop("_from_auto", False)}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        if is_offline_mode() and not local_files_only:
            log.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
        path = str(path)
        if os.path.isfile(path) or is_remote_url(path):
            x = path
        else:
            f = kw.pop("_configuration_file", CONFIG_NAME)
            if os.path.isdir(path):
                x = os.path.join(path, f)
            else:
                x = hf_bucket_url(path, filename=f, revision=kw.pop("revision", None), mirror=None)
        try:
            x2 = cached_path(
                x,
                cache_dir=kw.pop("cache_dir", None),
                force_download=kw.pop("force_download", False),
                proxies=kw.pop("proxies", None),
                resume_download=kw.pop("resume_download", False),
                local_files_only=local_files_only,
                use_auth_token=kw.pop("use_auth_token", None),
                user_agent=user_agent,
            )
        except RepositoryNotFoundError as e:
            raise OSError() from e
        except RevisionNotFoundError as e:
            raise OSError() from e
        except EntryNotFoundError as e:
            raise OSError() from e
        except HTTPError as e:
            raise OSError() from e
        except OSError as e:
            raise e
        try:
            y = cls._dict_from_json_file(x2)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise OSError() from e
        if x2 == x:
            log.info(f"loading {x}")
        else:
            log.info(f"loading {x} from cache at {x2}")
        return y, kw

    @classmethod
    def from_dict(cls, x, **kw):
        return_unused_kw = kw.pop("return_unused_kw", False)
        y = cls(**x)
        ks = []
        for k, v in kw.items():
            if hasattr(y, k):
                setattr(y, k, v)
                if k != "torch_dtype":
                    ks.append(k)
        for k in ks:
            kw.pop(k, None)
        log.info(f"Model config {y}")
        if return_unused_kw:
            return y, kw
        else:
            return y

    @classmethod
    def from_json_file(cls, x):
        return cls(**cls._dict_from_json_file(x))

    @classmethod
    def _dict_from_json_file(cls, x):
        with open(x, "r", encoding="utf-8") as r:
            y = r.read()
        return json.loads(y)

    def __eq__(self, x):
        return self.__dict__ == x.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self):
        y = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            y["model_type"] = self.__class__.model_type
        if "_auto_class" in y:
            del y["_auto_class"]
        self.dict_torch_dtype_to_str(y)
        return y

    def to_diff_dict(self):
        d = PreTrained().to_dict()
        c = self.__class__().to_dict() if not self.is_composition else {}
        y = {}
        for k, v in self.to_dict().items():
            if k not in d or v != d[k] or (k in c and v != c[k]):
                y[k] = v
        self.dict_torch_dtype_to_str(y)
        return y

    def to_json_string(self, use_diff=True):
        if use_diff is True:
            y = self.to_diff_dict()
        else:
            y = self.to_dict()
        return json.dumps(y, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, path, use_diff=True):
        with open(path, "w", encoding="utf-8") as w:
            w.write(self.to_json_string(use_diff=use_diff))

    def update(self, x):
        for k, v in x.items():
            setattr(self, k, v)

    def get_mask(self, m, shape, device=None):
        if m.dim() == 3:
            m = m[:, None, :, :]
        else:
            assert m.dim() == 2
            if self.cfg.is_dec:

                def for_dec(x):
                    b, n = shape
                    xs = torch.arange(n, device=device)
                    y = xs[None, None, :].repeat(b, n, 1) <= xs[None, :, None]
                    y = y.to(x.dtype)
                    if y.shape[1] < x.shape[1]:
                        d = x.shape[1] - y.shape[1]
                        y = torch.cat(
                            [torch.ones((b, n, d), device=device, dtype=y.dtype), y], axis=-1
                        )
                    y = y[:, None, :, :] * x[:, None, None, :]
                    return y

                m = for_dec(m)
            else:
                m = m[:, None, None, :]
        m = m.to(dtype=self.cfg.dtype)
        m = (1.0 - m) * -10000.0
        return m

    def get_head_m(self, x, n_lays, is_chunked=False):
        if x is None:
            y = [None] * n_lays
        else:
            if x.dim() == 1:
                y = x.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                y = y.expand(n_lays, -1, -1, -1, -1)
            elif x.dim() == 2:
                y = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert y.dim() == 5
            y = y.to(dtype=self.cfg.dtype)
            if is_chunked is True:
                y = y.unsqueeze(-1)
        return y

    def get_head_m2(self, x, n_lays):
        if x is None:
            y = [None] * n_lays
        else:
            if x.dim() == 1:
                y = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                y = y.expand(n_lays, -1, -1, -1, -1)
            elif x.dim() == 2:
                y = x.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            assert y.dim() == 5
            y = y.to(dtype=self.cfg.dtype)
        return y


_CFG_FILE = re.compile(r"config\.(.*)\.json")


def get_configuration_file(xs):
    map = {}
    for x in xs:
        s = _CFG_FILE.search(x)
        if s is not None:
            map[s.groups()[0]] = x
    ks = sorted(map.keys())
    y = CONFIG_NAME
    for k in ks:
        y = map[k]
    return y
