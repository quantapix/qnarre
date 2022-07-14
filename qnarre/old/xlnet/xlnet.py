from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
import modeling


def _get_initializer(FLAGS):
    """Get variable intializer."""
    if FLAGS.init == "uniform":
        initializer = tf.initializers.random_uniform(
            minval=-FLAGS.init_range, maxval=FLAGS.init_range, seed=None
        )
    elif FLAGS.init == "normal":
        initializer = tf.initializers.random_normal(stddev=FLAGS.init_std, seed=None)
    else:
        raise ValueError("Initializer {} not supported".format(FLAGS.init))
    return initializer


class XLNetConfig(object):
    """XLNetConfig contains hyperparameters that are specific to a model checkpoint;
    i.e., these hyperparameters should be the same between
    pretraining and finetuning.

    The following hyperparameters are defined:
      n_lays, the number of layers.
      d_model, the hidden size.
      n_heads, the number of attention heads.
      d_head, the dimension size of each attention head.
      d_inner, the hidden size in feed-forward layers.
      ff_activation, "relu" or "gelu".
      untie_r, whether to untie the biases in attention.
      n_token, the vocab size.
    """

    def __init__(self, FLAGS=None, json_path=None):
        """Constructing an XLNetConfig.
        One of FLAGS or json_path should be provided."""

        assert FLAGS is not None or json_path is not None

        self.keys = [
            "n_lays",
            "d_model",
            "n_heads",
            "d_head",
            "d_inner",
            "ff_activation",
            "untie_r",
            "n_token",
        ]

        if FLAGS is not None:
            self.init_from_flags(FLAGS)

        if json_path is not None:
            self.init_from_json(json_path)

    def init_from_flags(self, FLAGS):
        for key in self.keys:
            setattr(self, key, getattr(FLAGS, key))

    def init_from_json(self, json_path):
        with tf.gfile.Open(json_path) as f:
            json_data = json.load(f)
            for key in self.keys:
                setattr(self, key, json_data[key])

    def to_json(self, json_path):
        """Save XLNetConfig to a json file."""
        json_data = {}
        for key in self.keys:
            json_data[key] = getattr(self, key)

        json_dir = os.path.dirname(json_path)
        if not tf.gfile.Exists(json_dir):
            tf.gfile.MakeDirs(json_dir)
        with tf.gfile.Open(json_path, "w") as f:
            json.dump(json_data, f, indent=4, sort_keys=True)


def create_run_config(is_training, is_finetune, FLAGS):
    kw = dict(
        is_training=is_training,
        use_tpu=FLAGS.use_tpu,
        use_bfloat16=FLAGS.use_bfloat16,
        drop=FLAGS.drop,
        dropatt=FLAGS.dropatt,
        init=FLAGS.init,
        init_range=FLAGS.init_range,
        init_std=FLAGS.init_std,
        clamp_len=FLAGS.clamp_len,
    )

    if not is_finetune:
        kw.update(
            dict(
                mem_len=FLAGS.mem_len,
                reuse_len=FLAGS.reuse_len,
                bi_data=FLAGS.bi_data,
                clamp_len=FLAGS.clamp_len,
                same_length=FLAGS.same_length,
            )
        )

    return RunConfig(**kw)


class RunConfig(object):
    """RunConfig contains hyperparameters that could be different
    between pretraining and finetuning.
    These hyperparameters can also be changed from run to run.
    We store them separately from XLNetConfig for flexibility.
    """

    def __init__(
        self,
        is_training,
        use_tpu,
        use_bfloat16,
        drop,
        dropatt,
        init="normal",
        init_range=0.1,
        init_std=0.02,
        mem_len=None,
        reuse_len=None,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
    ):
        """
        Args:
          is_training, whether in training mode.
          use_tpu, whether TPUs are used.
          use_bfloat16, use bfloat16 instead of float32.
          drop: float, drop rate.
          dropatt: float, drop rate on attention probabilities.
          init, the initialization scheme, either "normal" or "uniform".
          init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
          init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
          mem_len, the number of tokens to cache.
          reuse_len, the number of tokens in the currect batch to be cached
            and reused in the future.
          bi_data, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
          clamp_len, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
          same_length, whether to use the same attention length for each token.
        """

        self.init = init
        self.init_range = init_range
        self.init_std = init_std
        self.is_training = is_training
        self.drop = drop
        self.dropatt = dropatt
        self.use_tpu = use_tpu
        self.use_bfloat16 = use_bfloat16
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length


class XLNetModel(object):
    """A wrapper of the XLNet model used during both pretraining and finetuning."""

    def __init__(
        self,
        xlnet_config,
        run_config,
        input_ids,
        seg_ids,
        input_mask,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        inp_q=None,
        **kw,
    ):

        initializer = _get_initializer(run_config)

        tfm_args = dict(
            n_token=xlnet_config.n_token,
            initializer=initializer,
            attn_type="bi",
            n_lays=xlnet_config.n_lays,
            d_model=xlnet_config.d_model,
            n_heads=xlnet_config.n_heads,
            d_head=xlnet_config.d_head,
            d_inner=xlnet_config.d_inner,
            ff_activation=xlnet_config.ff_activation,
            untie_r=xlnet_config.untie_r,
            is_training=run_config.is_training,
            use_bfloat16=run_config.use_bfloat16,
            use_tpu=run_config.use_tpu,
            drop=run_config.drop,
            dropatt=run_config.dropatt,
            mem_len=run_config.mem_len,
            reuse_len=run_config.reuse_len,
            bi_data=run_config.bi_data,
            clamp_len=run_config.clamp_len,
            same_length=run_config.same_length,
        )

        input_args = dict(
            inp_k=input_ids,
            seg_id=seg_ids,
            input_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            inp_q=inp_q,
        )
        tfm_args.update(input_args)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            (self.output, self.new_mems, self.lookup_table) = modeling.transformer_xl(**tfm_args)

        self.input_mask = input_mask
        self.initializer = initializer
        self.xlnet_config = xlnet_config
        self.run_config = run_config

    def get_pooled_out(self, sum_type, use_summ_proj=True):
        """
        Args:
          sum_type, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation.
          use_summ_proj, whether to use a linear projection during pooling.

        Returns:
          float32 Tensor in shape [bsz, d_model], the pooled representation.
        """

        xlnet_config = self.xlnet_config
        run_config = self.run_config

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            summary = modeling.summarize_sequence(
                sum_type=sum_type,
                hidden=self.output,
                d_model=xlnet_config.d_model,
                n_heads=xlnet_config.n_heads,
                d_head=xlnet_config.d_head,
                drop=run_config.drop,
                dropatt=run_config.dropatt,
                is_training=run_config.is_training,
                input_mask=self.input_mask,
                initializer=self.initializer,
                use_proj=use_summ_proj,
            )

        return summary

    def get_sequence_output(self):
        """
        Returns:
          float32 Tensor in shape [len, bsz, d_model]. The last layer hidden
          representation of XLNet.
        """

        return self.output

    def get_new_memory(self):
        """
        Returns:
          list of float32 Tensors in shape [mem_len, bsz, d_model], the new
          memory that concatenates the previous memory with the current input
          representations.
          The length of the list equals n_lays.
        """
        return self.new_mems

    def get_embedding_table(self):
        """
        Returns:
          float32 Tensor in shape [n_token, d_model]. The embedding lookup table.
          Used for tying embeddings between input and output layers.
        """
        return self.lookup_table

    def get_initializer(self):
        """
        Returns:
          A tf initializer. Used to initialize variables in layers on top of XLNet.
        """
        return self.initializer
