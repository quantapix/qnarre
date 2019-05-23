# Copyright 2019 Quantapix Authors. All Rights Reserved.
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
# https://arxiv.org/pdf/1901.02860.pdf

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.feeds.dset.trafo-xl import dset as dset

import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.gfile import Exists as exists
import model
import data_utils
import tpu_estimator

import numpy as np
from time import sleep

# TPU parameters
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url."
)
flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, "
    "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool("use_tpu",
                  default=True,
                  help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1, help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host",
                     default=8,
                     help="number of cores per host")

# Experiment (data/checkpoint/directory) parameters
flags.DEFINE_string("dir_data",
                    default="",
                    help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir",
                    default="",
                    help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path",
                    default="",
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("dir_model", default=None, help="Estimator dir_model.")
flags.DEFINE_bool("do_eval",
                  default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_bool("track_mean",
                  default=True,
                  help="Trace mean loss during training.")
flags.DEFINE_string("eval_ckpt_path",
                    None,
                    help="Checkpoint path for evaluation."
                    "If set, dir_model will be ignored."
                    "If unset, will use the latest ckpt in dir_model.")
flags.DEFINE_string("warm_start_path",
                    None,
                    help="Checkpoint path for warm start."
                    "If set, will clear Adam states."
                    "Note that the new dir_model should be different"
                    " from warm_start_path.")

# Optimization paramenters
flags.DEFINE_float("learning_rate",
                   default=2.5e-4,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25, help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio",
                   default=0.01,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps",
                     default=0,
                     help="Number of steps for linear lr warmup.")

# Training parameters
flags.DEFINE_integer("train_batch_size",
                     default=60,
                     help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size",
                     default=60,
                     help="Size of valid batch.")
flags.DEFINE_integer("train_steps",
                     default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations",
                     default=500,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps",
                     default=10000,
                     help="number of steps for model checkpointing.")

# Evaluation parameters
flags.DEFINE_integer("max_eval_batch",
                     default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False, help="Run evaluation only.")
flags.DEFINE_integer(
    "start_eval_steps",
    default=10000,
    help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split",
                    "valid",
                    help="Which data split to evaluate.")

# Model paramenters
flags.DEFINE_integer("tgt_len", default=70, help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70, help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False, help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")

flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
flags.DEFINE_integer("d_model", default=500, help="Dimension of the model.")
flags.DEFINE_integer("d_embed",
                     default=500,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10, help="Number of attention heads.")
flags.DEFINE_integer("d_head",
                     default=50,
                     help="Dimension of each attention head.")
flags.DEFINE_integer(
    "d_inner",
    default=1000,
    help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False, help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight",
                  default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val",
                     default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool(
    "proj_share_all_but_first",
    default=False,
    help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim",
                  default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init",
                  default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std",
                   default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std",
                   default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range",
                   default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def metric_fn(loss):
    """Evaluation metric Fn which runs on CPU."""
    perplexity = tf.exp(tf.reduce_mean(loss))
    bpc = tf.reduce_mean(loss) / tf.constant(math.log(2))
    return {
        "perplexity": tf.metrics.mean(perplexity),
        "bpc": tf.metrics.mean(bpc),
    }


def get_model_fn(n_token, cutoffs, train_bin_sizes, eval_bin_sizes):
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        batch_size = params["batch_size"]

        mems = params["cache"]
        inp = tf.transpose(features["inputs"], [1, 0])
        tgt = tf.transpose(features["labels"], [1, 0])

        bin_sizes = train_bin_sizes if is_training else eval_bin_sizes
        if bin_sizes:
            inp_perms = [tf.transpose(features["inp_mask"], [1, 0])]
            tgt_perms = [tf.transpose(features["tgt_mask"], [1, 0])]

            head_tgt = tf.transpose(features["head_labels"], [1, 0])

            for b in range(len(bin_sizes)):
                inp_perm = tf.transpose(features["inp_perm_{}".format(b)],
                                        [1, 0, 2])
                tgt_perm = tf.transpose(features["tgt_perm_{}".format(b)],
                                        [1, 0, 2])

                inp_perms.append(inp_perm)
                tgt_perms.append(tgt_perm)
        else:
            inp_perms, tgt_perms, head_tgt = None, None, None

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range, maxval=FLAGS.init_range, seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(stddev=FLAGS.init_std,
                                                        seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std, seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        tf.logging.info("Vocab size : {}".format(n_token))
        tf.logging.info("Batch size : {}".format(batch_size))

        loss, new_mems = model.transformer(dec_inp=inp,
                                           target=tgt,
                                           mems=mems,
                                           n_token=n_token,
                                           n_layer=FLAGS.n_layer,
                                           d_model=FLAGS.d_model,
                                           d_embed=FLAGS.d_embed,
                                           n_head=FLAGS.n_head,
                                           d_head=FLAGS.d_head,
                                           d_inner=FLAGS.d_inner,
                                           dropout=FLAGS.dropout,
                                           dropatt=FLAGS.dropatt,
                                           initializer=initializer,
                                           is_training=is_training,
                                           mem_len=FLAGS.mem_len,
                                           cutoffs=cutoffs,
                                           div_val=FLAGS.div_val,
                                           tie_projs=tie_projs,
                                           input_perms=inp_perms,
                                           target_perms=tgt_perms,
                                           head_target=head_tgt,
                                           same_length=FLAGS.same_length,
                                           clamp_len=FLAGS.clamp_len,
                                           use_tpu=FLAGS.use_tpu,
                                           untie_r=FLAGS.untie_r,
                                           proj_same_dim=FLAGS.proj_same_dim)

        total_loss = tf.reduce_mean(loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            if FLAGS.use_tpu:
                with tf.colocate_with(total_loss):
                    total_loss = tf.contrib.tpu.cross_replica_sum(total_loss) \
                               / FLAGS.num_hosts / FLAGS.num_core_per_host
            metric_loss = tf.tile(tf.reshape(total_loss, [1, 1]),
                                  [batch_size, 1])
            eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=(metric_fn, [metric_loss]))

            eval_spec.cache = new_mems

            return eval_spec

        # Configuring the optimization step.
        global_step = tf.train.get_global_step()

        # increase the learning rate linearly
        if FLAGS.warmup_steps > 0:
            warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                        * FLAGS.learning_rate
        else:
            warmup_lr = 0.0

        # number of parameters
        num_params = np.sum(
            [np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info("#params: {}".format(num_params))

        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.trainable_variables()]))
        # for v in tf.trainable_variables():
        #   tf.logging.info(format_str.format(v.name, v.get_shape()))

        # decay the learning rate using the cosine schedule
        decay_lr = tf.train.cosine_decay(
            FLAGS.learning_rate,
            global_step=global_step - FLAGS.warmup_steps,
            decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
            alpha=FLAGS.min_lr_ratio)

        learning_rate = tf.where(global_step < FLAGS.warmup_steps, warmup_lr,
                                 decay_lr)

        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(
                tf.train.AdamOptimizer(learning_rate=learning_rate))
            #GradientDescentOptimizer
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads_and_vars = optimizer.compute_gradients(total_loss)
        gradients, variables = zip(*grads_and_vars)
        clipped, _ = tf.clip_by_global_norm(gradients, FLAGS.clip)
        train_op = optimizer.apply_gradients(
            zip(clipped, variables), global_step=tf.train.get_global_step())

        # Constucting TPUEstimatorSpec with cache.
        train_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op)

        if FLAGS.mem_len < FLAGS.tgt_len:
            new_mems = [new_mems[:FLAGS.mem_len] for mem_t in new_mems]
        train_spec.cache = new_mems

        return train_spec

    return model_fn


def get_cache_fn(mem_len):
    def cache_fn(batch_size):
        mems = []
        for l in xrange(FLAGS.n_layer):
            if mem_len > 0:
                mems.append(
                    tf.zeros([mem_len, batch_size, FLAGS.d_model],
                             dtype=tf.float32))
            else:
                mems.append(tf.zeros([mem_len], dtype=tf.float32))

        return mems

    return cache_fn


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]

    if FLAGS.save_steps == 0:
        FLAGS.save_steps = None

    if not FLAGS.do_eval_only:
        # Get train input function
        train_input_fn, train_record_info = data_utils.get_input_fn(
            record_info_dir=FLAGS.record_info_dir,
            split="train",
            per_host_bsz=FLAGS.train_batch_size // FLAGS.num_hosts,
            tgt_len=FLAGS.tgt_len,
            num_core_per_host=FLAGS.num_core_per_host,
            num_hosts=FLAGS.num_hosts,
            use_tpu=FLAGS.use_tpu)
        train_bin_sizes = train_record_info["bin_sizes"]
        num_train_batch = train_record_info["num_batch"]

        # Get train cache function
        train_cache_fn = get_cache_fn(FLAGS.mem_len)
    else:
        train_bin_sizes = []
        num_train_batch = None
        train_cache_fn = None

    if FLAGS.do_eval or FLAGS.do_eval_only:
        assert FLAGS.num_hosts == 1
        # Get eval input function
        eval_input_fn, eval_record_info = data_utils.get_input_fn(
            record_info_dir=FLAGS.record_info_dir,
            split=FLAGS.eval_split,
            per_host_bsz=FLAGS.eval_batch_size // FLAGS.num_hosts,
            tgt_len=FLAGS.tgt_len,
            num_core_per_host=FLAGS.num_core_per_host,
            num_hosts=FLAGS.num_hosts,
            use_tpu=FLAGS.use_tpu)
        eval_bin_sizes = eval_record_info["bin_sizes"]
        num_eval_batch = eval_record_info["num_batch"]

        if FLAGS.max_eval_batch > 0:
            num_eval_batch = min(FLAGS.max_eval_batch, num_eval_batch)

        # Get eval cache function
        eval_cache_fn = get_cache_fn(FLAGS.mem_len)
        model_fn = get_model_fn(n_token, cutoffs, train_bin_sizes,
                                eval_bin_sizes)
    else:
        eval_cache_fn = None
        model_fn = get_model_fn(n_token, cutoffs, train_bin_sizes, [])

    ##### Create estimator
    # TPU Configuration
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        dir_model=FLAGS.dir_model,
        session_config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations,
            num_shards=FLAGS.num_core_per_host * FLAGS.num_hosts,
            per_host_input_for_training=per_host_input),
        keep_checkpoint_max=100000,  # effectively save all checkpoints
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.save_steps)

    # warm start
    warm_start_from = None
    if FLAGS.warm_start_path is not None:
        warm_start_from = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.warm_start_path)

    # TPU Estimator
    estimator = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        train_cache_fn=train_cache_fn,
        eval_cache_fn=eval_cache_fn,
        use_tpu=FLAGS.use_tpu,
        config=run_config,
        params={
            "dir_data": FLAGS.dir_data,
            "track_mean": FLAGS.track_mean
        },
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        warm_start_from=warm_start_from)

    if FLAGS.do_eval_only:
        if FLAGS.eval_ckpt_path is not None:
            ret = estimator.evaluate(input_fn=eval_input_fn,
                                     steps=num_eval_batch,
                                     checkpoint_path=FLAGS.eval_ckpt_path)
            tf.logging.info("=" * 200)
            log_str = "Eval results | "
            for key, val in ret.items():
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)
            tf.logging.info("=" * 200)
        else:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.dir_model)
            eval_results = []
            for eval_checkpoint in ckpt_state.all_model_checkpoint_paths:
                if not exists(eval_checkpoint + ".index"): continue
                global_step = int(eval_checkpoint.split("-")[-1])
                if global_step < FLAGS.start_eval_steps or global_step > FLAGS.train_steps:
                    continue
                ret = estimator.evaluate(input_fn=eval_input_fn,
                                         steps=num_eval_batch,
                                         checkpoint_path=eval_checkpoint)
                eval_results.append(ret)

            eval_results.sort(key=lambda x: x["perplexity"])

            tf.logging.info("=" * 200)
            log_str = "Best results | "
            for key, val in eval_results[0].items():
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)
            tf.logging.info("=" * 200)
    else:
        if not FLAGS.do_eval:
            estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)
        else:
            for step in range(0, FLAGS.train_steps, num_train_batch):
                train_steps = min(FLAGS.train_steps - step, num_train_batch)
                estimator.train(input_fn=train_input_fn, steps=train_steps)
                estimator.evaluate(input_fn=eval_input_fn,
                                   steps=num_eval_batch)


if __name__ == "__main__":
    tf.app.run()


    import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import model
import data_utils

from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np

# GPU config
flags.DEFINE_integer("num_hosts", default=1, help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host",
                     default=8,
                     help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("dir_data",
                    default="",
                    help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir",
                    default="",
                    help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path",
                    default="",
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("dir_model", default=None, help="Estimator dir_model.")
flags.DEFINE_bool("do_train", default=True, help="Whether to run training.")
flags.DEFINE_bool("do_eval",
                  default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path",
                    None,
                    help="Checkpoint path for do_test evaluation."
                    "If set, dir_model will be ignored."
                    "If unset, will use the latest ckpt in dir_model.")
flags.DEFINE_string("warm_start_path",
                    None,
                    help="Checkpoint path for warm start."
                    "If set, will clear Adam states."
                    "Note that the new dir_model should be different"
                    " from warm_start_path.")

# Optimization config
flags.DEFINE_float("learning_rate",
                   default=2.5e-4,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25, help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio",
                   default=0.004,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps",
                     default=0,
                     help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size",
                     default=60,
                     help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size",
                     default=60,
                     help="Size of valid batch.")
flags.DEFINE_integer("train_steps",
                     default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations",
                     default=500,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps",
                     default=10000,
                     help="number of steps for model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False, help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch",
                     default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False, help="Run evaluation only.")
flags.DEFINE_integer(
    "start_eval_steps",
    default=10000,
    help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split",
                    "valid",
                    help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("tgt_len", default=70, help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70, help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False, help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")

flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
flags.DEFINE_integer("d_model", default=500, help="Dimension of the model.")
flags.DEFINE_integer("d_embed",
                     default=500,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10, help="Number of attention heads.")
flags.DEFINE_integer("d_head",
                     default=50,
                     help="Dimension of each attention head.")
flags.DEFINE_integer(
    "d_inner",
    default=1000,
    help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False, help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight",
                  default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val",
                     default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool(
    "proj_share_all_but_first",
    default=False,
    help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim",
                  default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init",
                  default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std",
                   default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std",
                   default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range",
                   default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def get_model_fn(n_token, cutoffs):
    def model_fn(inp, tgt, mems, is_training):
        inp = tf.transpose(inp, [1, 0])
        tgt = tf.transpose(tgt, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range, maxval=FLAGS.init_range, seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(stddev=FLAGS.init_std,
                                                        seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std, seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        loss, new_mems = model.transformer(dec_inp=inp,
                                           target=tgt,
                                           mems=mems,
                                           n_token=n_token,
                                           n_layer=FLAGS.n_layer,
                                           d_model=FLAGS.d_model,
                                           d_embed=FLAGS.d_embed,
                                           n_head=FLAGS.n_head,
                                           d_head=FLAGS.d_head,
                                           d_inner=FLAGS.d_inner,
                                           dropout=FLAGS.dropout,
                                           dropatt=FLAGS.dropatt,
                                           initializer=initializer,
                                           proj_initializer=proj_initializer,
                                           is_training=is_training,
                                           mem_len=FLAGS.mem_len,
                                           cutoffs=cutoffs,
                                           div_val=FLAGS.div_val,
                                           tie_projs=tie_projs,
                                           input_perms=None,
                                           target_perms=None,
                                           head_target=None,
                                           same_length=FLAGS.same_length,
                                           clamp_len=FLAGS.clamp_len,
                                           use_tpu=False,
                                           untie_r=FLAGS.untie_r,
                                           proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.trainable_variables()]))
        # for v in tf.trainable_variables():
        #   tf.logging.info(format_str.format(v.name, v.get_shape()))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars
        else:
            return loss, new_mems

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
    model_fn = get_model_fn(n_token=n_token, cutoffs=cutoffs)

    model_ret = model_fn(inp=inp, tgt=tgt, mems=mems, is_training=is_training)

    return model_ret


def train(n_token, cutoffs, ps_device):
    ##### Get input function and model function
    train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1,
        use_tpu=False)

    tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

    ##### Create computational graph
    train_set = train_input_fn({
        "batch_size": FLAGS.train_batch_size,
        "dir_data": FLAGS.dir_data
    })

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
            tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            mems_i = [
                tf.placeholder(tf.float32,
                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)
            ]

            loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=True,
                inp=inputs[i],
                tgt=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)
            tower_grads_and_vars.append(grads_and_vars_i)

    ## average losses and gradients across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
    grads, all_vars = zip(*grads_and_vars)

    ## clip gradient
    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))

    ## configure the optimizer
    global_step = tf.train.get_or_create_global_step()

    # warmup stage: increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                    * FLAGS.learning_rate
    else:
        warmup_lr = 0.0

    # decay stage: decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    # choose warmup or decay
    learning_rate = tf.where(global_step < FLAGS.warmup_steps, warmup_lr,
                             decay_lr)

    # get the train op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    ##### Training loop
    tower_mems_np = [[
        np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model],
                 dtype=np.float32) for layer in range(FLAGS.n_layer)
    ] for core in range(FLAGS.num_core_per_host)]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.warm_start_path is not None:
            tf.logging.info("warm start from {}".format(FLAGS.warm_start_path))
            saver.restore(sess, FLAGS.warm_start_path)

        fetches = [
            loss, tower_new_mems, global_step, gnorm, learning_rate, train_op
        ]

        total_loss, prev_step = 0., -1
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, curr_step = fetched[:3]
            total_loss += loss_np

            if curr_step > 0 and curr_step % FLAGS.iterations == 0:
                curr_loss = total_loss / (curr_step - prev_step)
                tf.logging.info(
                    "[{}] | gnorm {:.2f} lr {:8.6f} "
                    "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
                        curr_step, fetched[-3], fetched[-2], curr_loss,
                        math.exp(curr_loss), curr_loss / math.log(2)))
                total_loss, prev_step = 0., curr_step

            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.dir_model, "model.ckpt")
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

            if curr_step == FLAGS.train_steps:
                break


def evaluate(n_token, cutoffs, ps_device):
    ##### Get input function and model function
    eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split=FLAGS.eval_split,
        per_host_bsz=FLAGS.eval_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1,
        use_tpu=False)

    num_batch = eval_record_info["num_batch"]
    if FLAGS.max_eval_batch > 0:
        num_batch = FLAGS.max_eval_batch
    tf.logging.info("num of batches {}".format(num_batch))

    ##### Create computational graph
    eval_set = eval_input_fn({
        "batch_size": FLAGS.eval_batch_size,
        "dir_data": FLAGS.dir_data
    })

    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
            tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            mems_i = [
                tf.placeholder(tf.float32,
                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)
            ]

            loss_i, new_mems_i = single_core_graph(n_token=n_token,
                                                   cutoffs=cutoffs,
                                                   is_training=False,
                                                   inp=inputs[i],
                                                   tgt=labels[i],
                                                   mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)

    ## sum losses across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
        loss = tower_losses[0]

    ##### Evaluation loop
    tower_mems_np = [[
        np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model],
                 dtype=np.float32) for layer in range(FLAGS.n_layer)
    ] for core in range(FLAGS.num_core_per_host)]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.dir_model)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path
        tf.logging.info("Evaluate {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, tf.size(label_feed)]

        format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
            len(str(num_batch)))

        total_loss, total_cnt = 0, 0
        for step in range(num_batch):
            if step % (num_batch // 10) == 0:
                tf.logging.info(format_str.format(step, num_batch))

            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, cnt_np = fetched[:3]
            total_loss += loss_np * cnt_np
            total_cnt += cnt_np

        avg_loss = total_loss / total_cnt
        tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token {}".format(n_token))

    if FLAGS.do_train:
        train(n_token, cutoffs, "/gpu:0")
    if FLAGS.do_eval:
        evaluate(n_token, cutoffs, "/gpu:0")


if __name__ == "__main__":
    tf.app.run()


    import os
import tensorflow as tf


def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu

    return _assign


def average_grads_and_vars(tower_grads_and_vars):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0) / len(grad_and_vars)
        return tf.IndexedSlices(values, indices,
                                grad_and_vars[0][0].dense_shape)

    average_grads_and_vars = []
    for grad_and_vars in zip(*tower_grads_and_vars):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads_and_vars.append(grad_and_var)
    return average_grads_and_vars


def load_from_checkpoint(saver, logdir):
    sess = tf.get_default_session()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(logdir,
                                             ckpt.model_checkpoint_path))
        return True
    return False



# Data
DATA_ROOT=../data/enwik8/

# Model
N_LAYER=12
D_MODEL=512
D_EMBED=512
N_HEAD=8
D_HEAD=64
D_INNER=2048

# Training
TGT_LEN=512
MEM_LEN=512

BSZ=24
NUM_CORE=4

# Testing
TEST_TGT_LEN=80
TEST_MEM_LEN=2100
TEST_CLAMP_LEN=820

TEST_BSZ=10
TEST_NUM_CORE=1

if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=enwik8 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=enwik8 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-enwik8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=200 \
        --save_steps=4000 \
        --do_train=True \
        --do_eval=False \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-enwik8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi


# Data
DATA_ROOT=../data/one-billion-words/

# Model
DIV_VAL=4
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=8
D_HEAD=128
D_INNER=4096

# Training
TGT_LEN=256
MEM_LEN=256

BSZ=256
NUM_CORE=4

# Testing
TEST_TGT_LEN=32
TEST_MEM_LEN=128
TEST_CLAMP_LEN=-1

TEST_BSZ=16
TEST_NUM_CORE=1


if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
      --dir_data=${DATA_ROOT}/ \
      --dataset=lm1b \
      --tgt_len=${TGT_LEN} \
      --per_host_train_bsz=${BSZ} \
      --per_host_valid_bsz=${BSZ} \
      --num_passes=1 \
      --use_tpu=False \
      ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
      --dir_data=${DATA_ROOT}/ \
      --dataset=lm1b \
      --tgt_len=${TEST_TGT_LEN} \
      --per_host_test_bsz=${TEST_BSZ} \
      --num_passes=1 \
      --use_tpu=False \
      ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-lm1b \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=False \
        --proj_same_dim=False \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=200 \
        --save_steps=4000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-lm1b \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=False \
        --proj_same_dim=False \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi



# Data
DATA_ROOT=../data/text8/

# Model
N_LAYER=12
D_MODEL=512
D_EMBED=512
N_HEAD=8
D_HEAD=64
D_INNER=2048

# Training
TGT_LEN=512
MEM_LEN=512

BSZ=24
NUM_CORE=4

# Testing
TEST_TGT_LEN=80
TEST_MEM_LEN=2100
TEST_CLAMP_LEN=820

TEST_BSZ=10
TEST_NUM_CORE=1

if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=text8 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=text8 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-text8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=200 \
        --save_steps=4000 \
        --do_train=True \
        --do_eval=False \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-text8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi



# Data
DATA_ROOT=../data/wikitext-103/

# Model
DIV_VAL=1
N_LAYER=16
D_MODEL=410
D_EMBED=410
N_HEAD=10
D_HEAD=41
D_INNER=2100

# Training
TGT_LEN=150
MEM_LEN=150

BSZ=60
NUM_CORE=4

# Testing
TEST_TGT_LEN=64
TEST_MEM_LEN=640
TEST_CLAMP_LEN=400

TEST_BSZ=10
TEST_NUM_CORE=1


if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=wt103 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --dir_data=${DATA_ROOT}/ \
        --dataset=enwik8 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-wt103 \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=200 \
        --save_steps=4000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --dir_data=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --dir_model=EXP-wt103 \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
