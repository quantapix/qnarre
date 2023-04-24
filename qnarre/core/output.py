from dataclasses import fields, dataclass


class Output(dict):
    def __init__(self, *xs, **kw):
        n = len(xs)
        if n > 0:
            x0 = xs[0]
            if isinstance(x0, dict):
                x0.update(kw)
                kw = x0
                n = 0
            else:
                try:
                    for x in iter(x0):
                        if (
                            not isinstance(x, (list, tuple))
                            or not len(x) == 2
                            or not isinstance(x[0], str)
                        ):
                            break
                        if x[1] is not None:
                            kw.update(tuple(x))
                        n = 0
                except TypeError:
                    pass
        for i, f in enumerate(fields(self)):
            v = xs[i] if i < n else None
            if v is None:
                v = kw.get(f.name, f.default)
            setattr(self, f.name, v)
            self[f.name] = v


@dataclass(init=False)
class Base(Output):
    y: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    globals: tuple | None = None


@dataclass
class WithCaches(Output):
    y: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    hiddens: tuple | None = None


@dataclass
class WithCrosses(Output):
    y: tuple | None = None
    attns: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None


@dataclass
class WithLoss(Output):
    logits: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    globals: tuple | None = None
    loss: tuple | None = None


@dataclass
class WithMems(Output):
    y: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    mems: tuple | None = None


@dataclass
class WithPools(Output):
    y: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    globals: tuple | None = None
    pools: tuple | None = None


@dataclass
class CachesCrosses(Output):
    y: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None


@dataclass
class PoolsCrosses(Output):
    y: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None
    pools: tuple | None = None


@dataclass
class Seq2Seq(Output):
    y: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None
    enc_y: tuple | None = None
    enc_attns: tuple | None = None
    enc_hiddens: tuple | None = None
    enc_globals: tuple | None = None


@dataclass
class LossCaches(Output):
    logits: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    hiddens: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossCrosses(Output):
    logits: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossMems(Output):
    logits: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    mems: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossQA(Output):
    logits_beg: tuple | None = None
    logits_end: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    globals: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossQAPools(Output):
    logits_beg: tuple | None = None
    logits_end: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    pools: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossSeq(Output):
    logits: tuple | None = None
    orders: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossSeq2Seq(Output):
    logits: tuple | None = None
    attns: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    hiddens: tuple | None = None
    enc_y: tuple | None = None
    enc_attns: tuple | None = None
    enc_hiddens: tuple | None = None
    enc_globals: tuple | None = None
    loss: tuple | None = None


@dataclass
class LossSeq2SeqQA(Output):
    logits_beg: tuple | None = None
    logits_end: tuple | None = None
    caches: tuple | None = None
    crosses: tuple | None = None
    attns: tuple | None = None
    hiddens: tuple | None = None
    enc_y: tuple | None = None
    enc_attns: tuple | None = None
    enc_hiddens: tuple | None = None
    enc_globals: tuple | None = None
    loss: tuple | None = None
