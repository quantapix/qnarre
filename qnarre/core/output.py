from collections import OrderedDict
from dataclasses import fields, dataclass
from transformers.file_utils import is_tensor


class Output(OrderedDict):
    def __post_init__(self):
        fs = fields(self)
        assert len(fs)
        assert all(f.default is None for f in fs[1:])
        first = getattr(self, fs[0].name)
        others_none = all(getattr(self, f.name) is None for f in fs[1:])
        if others_none and not is_tensor(first):
            if isinstance(first, dict):
                xs = first.items()
                first_iter = True
            else:
                try:
                    xs = iter(first)
                    first_iter = True
                except TypeError:
                    first_iter = False
            if first_iter:
                for x in xs:
                    if (
                        not isinstance(x, (list, tuple))
                        or not len(x) == 2
                        or not isinstance(x[0], str)
                    ):
                        break
                    setattr(self, x[0], x[1])
                    if x[1] is not None:
                        self[x[0]] = x[1]
            elif first is not None:
                self[fs[0].name] = first
        else:
            for f in fs:
                v = getattr(self, f.name)
                if v is not None:
                    self[f.name] = v

    def __delitem__(self, *args, **kw):
        raise Exception()

    def setdefault(self, *args, **kw):
        raise Exception()

    def pop(self, *args, **kw):
        raise Exception()

    def update(self, *args, **kw):
        raise Exception()

    def __getitem__(self, k):
        if isinstance(k, str):
            return {k: v for (k, v) in self.items()}[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, k, v):
        if k in self.keys() and v is not None:
            super().__setitem__(k, v)
        super().__setattr__(k, v)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        super().__setattr__(k, v)

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


@dataclass
class Base(Output):
    y: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    globals: tuple = None


@dataclass
class WithCaches(Output):
    y: tuple = None
    attns: tuple = None
    caches: tuple = None
    hiddens: tuple = None


@dataclass
class WithCrosses(Output):
    y: tuple = None
    attns: tuple = None
    crosses: tuple = None
    hiddens: tuple = None


@dataclass
class WithLoss(Output):
    logits = None
    attns: tuple = None
    hiddens: tuple = None
    globals: tuple = None
    loss = None


@dataclass
class WithMems(Output):
    y: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None


@dataclass
class WithPools(Output):
    y: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    globals: tuple = None
    pools: tuple = None


@dataclass
class CachesCrosses(Output):
    y: tuple = None
    attns: tuple = None
    caches: tuple = None
    crosses: tuple = None
    hiddens: tuple = None


@dataclass
class PoolsCrosses(Output):
    y: tuple = None
    attns: tuple = None
    caches: tuple = None
    crosses: tuple = None
    hiddens: tuple = None
    pools: tuple = None


@dataclass
class Seq2Seq(Output):
    y: tuple = None
    attns: tuple = None
    caches: tuple = None
    crosses: tuple = None
    hiddens: tuple = None
    enc_y: tuple = None
    enc_attns: tuple = None
    enc_hiddens: tuple = None
    enc_globals: tuple = None


@dataclass
class LossCaches(Output):
    logits = None
    attns: tuple = None
    caches: tuple = None
    hiddens: tuple = None
    loss = None


@dataclass
class LossCrosses(Output):
    logits = None
    attns: tuple = None
    caches: tuple = None
    crosses: tuple = None
    hiddens: tuple = None
    loss = None


@dataclass
class LossMems(Output):
    logits: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    mems: tuple = None
    loss: tuple = None


@dataclass
class LossQA(Output):
    logits_beg = None
    logits_end = None
    attns: tuple = None
    hiddens: tuple = None
    globals: tuple = None
    loss = None


@dataclass
class LossQAPools(Output):
    logits_beg: tuple = None
    logits_end: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    pools: tuple = None
    loss: tuple = None


@dataclass
class LossSeq(Output):
    logits: tuple = None
    orders: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    loss: tuple = None


@dataclass
class LossSeq2Seq(Output):
    logits = None
    attns: tuple = None
    caches: tuple = None
    crosses: tuple = None
    hiddens: tuple = None
    enc_y: tuple = None
    enc_attns: tuple = None
    enc_hiddens: tuple = None
    enc_globals: tuple = None
    loss = None


@dataclass
class LossSeq2SeqQA(Output):
    logits_beg = None
    logits_end = None
    caches: tuple = None
    crosses: tuple = None
    attns: tuple = None
    hiddens: tuple = None
    enc_y: tuple = None
    enc_attns: tuple = None
    enc_hiddens: tuple = None
    enc_globals: tuple = None
    loss = None
