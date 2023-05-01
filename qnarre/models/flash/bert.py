import re
import logging
from functools import partial

from collections.abc import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

from einops import rearrange

from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp, FusedMLP
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.bert_padding import index_first_axis, index_first_axis_residual
from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm, layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None


logger = logging.getLogger(__name__)


class BertPooler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fused_bias_fc = getattr(cfg, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(cfg.hidden_size, cfg.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fused_bias_fc = getattr(cfg, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(cfg, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(cfg.hidden_size, cfg.hidden_size)
        approximate = "tanh" if cfg.hidden_act in ["gelu_new", "gelu_fast"] else "none"
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm(
                hidden_states, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
            )
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fused_bias_fc = getattr(cfg, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = BertPredictionHeadTransform(cfg)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(cfg.hidden_size, cfg.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.predictions = BertLMPredictionHead(cfg)
        self.seq_relationship = nn.Linear(cfg.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrained(nn.Module):
    def __init__(self, cfg, *inputs, **kwargs):
        super().__init__()
        if not isinstance(cfg, BertConfig):
            raise ValueError(
                "Parameter cfg in `{}(cfg)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, model_name, cfg, *inputs, **kwargs):
        model = cls(cfg, *inputs, **kwargs)
        load_return = model.load_state_dict(
            remap_state_dict(state_dict_from_pretrained(model_name), cfg), strict=False
        )
        logger.info(load_return)
        return model


class ForPreTraining(PreTrained):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dense_seq_output = getattr(cfg, "dense_seq_output", False)
        self.last_layer_subset = getattr(cfg, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(cfg, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = (
            nn.CrossEntropyLoss
            if not use_xentropy
            else partial(CrossEntropyLoss, inplace_backward=True)
        )
        self.model = Model(cfg)
        self.cls = BertPreTrainingHeads(cfg)
        self.mlm_loss = loss_cls(ignore_index=0)
        self.nsp_loss = loss_cls(ignore_index=-1)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=cfg.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.model.emb.word_embeddings.weight

    def forward(self, x, mask=None, labels=None, next_sentence_label=None, **kw):
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        ys = self.model(
            x,
            mask=mask.bool() if mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
            **kw,
        )
        sequence_output, pooled_output = ys.last_hidden_state, ys.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(
                    rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx
                )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        loss = None
        if labels is not None and next_sentence_label is not None:
            if self.dense_seq_output and labels is not None:
                masked_lm_loss = self.mlm_loss(
                    prediction_scores, labels.flatten()[masked_token_idx]
                )
            else:
                masked_lm_loss = self.mlm_loss(
                    rearrange(prediction_scores, "... v -> (...) v"),
                    rearrange(labels, "... -> (...)"),
                )
            next_sentence_loss = self.nsp_loss(
                rearrange(seq_relationship_score, "... t -> (...) t"),
                rearrange(next_sentence_label, "... -> (...)"),
            )
            loss = masked_lm_loss.float() + next_sentence_loss.float()
        return BertForPreTrainingOutput(
            loss=loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )


class Model(PreTrained):
    def __init__(self, cfg, add_pool=True):
        super().__init__(cfg)
        self.pad_vocab_size_multiple = getattr(cfg, "pad_vocab_size_multiple", 1)
        if cfg.vocab_size % self.pad_vocab_size_multiple != 0:
            cfg.vocab_size += self.pad_vocab_size_multiple - (
                cfg.vocab_size % self.pad_vocab_size_multiple
            )
        self.fused_dropout_add_ln = getattr(cfg, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        assert cfg.position_embedding_type == "absolute"
        assert cfg.hidden_act in ["gelu", "gelu_new", "gelu_fast"]

        self.emb = BertEmbeddings(
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.max_position_embeddings,
            cfg.type_vocab_size,
            padding_idx=cfg.pad_token_id,
        )
        self.emb_drop = nn.Dropout(cfg.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.enc = Encoder(cfg)
        self.pool = BertPooler(cfg) if add_pool else None

        self.apply(partial(_init_weights, initializer_range=cfg.initializer_range))

    def forward(
        self,
        x,
        position_ids=None,
        token_type_ids=None,
        mask=None,
        masked_tokens_mask=None,
        **kw,
    ):
        ys = self.emb(x, **kw)
        if not self.fused_dropout_add_ln:
            ys = self.emb_ln(ys)
        else:
            ys = layer_norm(ys, self.emb_ln.weight, self.emb_ln.bias, self.emb_ln.eps)
        ys = self.emb_drop(ys)
        if masked_tokens_mask is not None:
            batch_size, seqlen = x.shape[:2]
            first_col_mask = torch.zeros(batch_size, seqlen, dtype=torch.bool, device=x.device)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None
        ys = self.enc(ys, key_padding_mask=mask, subset_mask=subset_mask)
        if masked_tokens_mask is None:
            pooled_output = self.pool(ys) if self.pool is not None else None
        else:
            if mask is not None:
                subset_idx = subset_mask[mask]
                pool_input = ys[first_col_mask[mask][subset_idx]]
                ys = ys[masked_tokens_mask[mask][subset_idx]]
            else:
                pool_input = ys[first_col_mask[subset_mask]]
                ys = ys[masked_tokens_mask[subset_mask]]
            pooled_output = self.pool(pool_input, pool=False) if self.pool is not None else None
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=ys,
            pooler_output=pooled_output,
        )


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_flash_attn = getattr(cfg, "use_flash_attn", False)
        self.lays = nn.ModuleList([create_block(cfg, layer_idx=i) for i in range(cfg.n_lays)])

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask} if key_padding_mask is not None else None
            )
            for layer in self.lays:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            b, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.lays:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, b, seqlen)
            else:
                for layer in self.lays[:-1]:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                hidden_states = self.lays[-1](hidden_states_subset, mixer_kwargs=mixer_kwargs)
        return hidden_states


def create_mixer_cls(cfg, cross_attn=False, return_residual=False):
    use_flash_attn = getattr(cfg, "use_flash_attn", False)
    fused_bias_fc = getattr(cfg, "fused_bias_fc", False)
    mixer_cls = partial(
        MHA,
        num_heads=cfg.num_attention_heads,
        cross_attn=cross_attn,
        dropout=cfg.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
    )
    return mixer_cls


def create_mlp_cls(cfg, layer_idx=None, return_residual=False):
    inner_dim = cfg.intermediate_size
    fused_mlp = getattr(cfg, "fused_mlp", False)
    if fused_mlp:
        assert cfg.hidden_act in ["gelu_new", "gelu_fast"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if not fused_mlp:
        approximate = "tanh" if cfg.hidden_act in ["gelu_new", "gelu_fast"] else "none"
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    else:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(cfg, "mlp_checkpoint_lvl", 0)
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    return mlp_cls


def create_block(cfg, layer_idx=None):
    last_layer_subset = getattr(cfg, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == cfg.n_lays - 1
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(cfg, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(cfg, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=cfg.layer_norm_eps)
    block = Block(
        cfg.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=cfg.hidden_dropout_prob,
        resid_dropout2=cfg.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(cfg, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


def remap_state_dict(state_dict, cfg):
    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(cfg, "last_layer_subset", False)
    for d in range(cfg.n_lays):
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == cfg.n_lays - 1):
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    # Word embedding
    pad_vocab_size_multiple = getattr(cfg, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, cfg.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, cfg.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, cfg.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    return state_dict
