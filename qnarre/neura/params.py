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

import tensorflow as T

from absl import flags as F

from qnarre.neura import utils as U


def load_flags():
    # from official.utils.flags import core as fu
    # fu.define_base()
    # fu.define_performance(
    # all_reduce_alg=True,
    # dtype=False,
    # inter_op=False,
    # intra_op=False,
    # max_train_steps=False,
    # num_parallel_calls=False,
    # synthetic_data=True,
    # )
    # fu.define_image()
    # fu.define_benchmark()
    # F.adopt_module_key_flags(fu)
    F.DEFINE_bool('do_eval', None, '')
    F.DEFINE_bool('do_train', None, '')
    F.DEFINE_float('stop_threshold', None, '')
    F.DEFINE_float('train_epochs', None, '')
    F.DEFINE_integer('batch_size', None, '')
    F.DEFINE_integer('checkpoint_steps', None, '')
    F.DEFINE_integer('epochs_between_evals', None, '')
    F.DEFINE_integer('eval_batch_size', None, '')
    F.DEFINE_integer('eval_steps', None, '')
    F.DEFINE_integer('iters_per_loop', None, '')
    F.DEFINE_integer('train_steps', None, '')
    F.DEFINE_integer('warmup_steps', None, '')
    F.DEFINE_string('dir_data', None, '')
    # F.DEFINE_string('log_dir', None, '')
    F.DEFINE_string('dir_model', None, '')
    F.DEFINE_string('model', None, '')
    F.DEFINE_string('dir_save', None, '')
    df = ['channels_first', 'channels_last']
    F.DEFINE_enum('data_format', None, df, '')


def load_params():
    f = 'channels_first' if T.test.is_built_with_cuda() else 'channels_last'
    return U.Params(_params, data_format=F.FLAGS.data_format or f)


_params = dict(
    layout=None,
    features=None,
)

_params2 = dict(
    epochs_between_evals=None,
    # len_bucket_step=1.1,
    # vocab_divisor=1,
    adam_beta1=0.9,
    adam_beta2=0.997,  # 0.999
    adam_epsilon=1e-9,  # 1e-6
    adamw_decay=0.0,
    add_relative=False,
    all_reduce_alg=None,
    alpha=0.6,
    attn_bdims='',
    attn_type='dot_attn',
    beam_size=4,
    causal_self_attn=True,
    clip_grad_norm=2.0,  # 0.0 no gradient clipping,
    compress_steps=0,
    conv_first_kernel=3,
    daisy_chain_vars=True,
    data_format=None,
    dataset=None,
    dist_strategy=None,
    drop_long_seqs=False,
    ds_src_len=0,
    ds_tgt_len=0,
    dtype=None,
    eval_frequency=100,
    eval_steps=1,
    extra_decode_len=50,
    factored_logits=False,
    ffn_bdims='',
    ffn_layer='dense_dense',
    fixed_batch_size=False,
    full_predict=False,
    gpu_thread_mode=None,
    grad_noise_scale=0.0,
    group_epsilon=1e-5,
    heads_share_embed=False,
    init_gain=1.5,  # 1.0
    initializer='uniform_unit_scaling',  # 'orthogonal',
    input_frames=1,
    inter_op=None,
    intra_op=None,
    kernel_height=3,
    kernel_width=1,
    label_smoothing=0.1,
    learn_rate=2e-4,  # 2.0, squad 5e-6,
    loss_scale=None,
    lr_constant=0.1,
    lr_schedule='constant*linear_warmup*rsqrt_decay',
    lr_warmup_steps=200,
    max_position=0,
    max_train_steps=None,
    min_len_bucket=8,
    min_length=0,
    mixed_precision_loss=32768,
    mixed_precision_loss_scaler='exponential',
    mlm_preds=20,
    mlm_prob=0.15,
    model=None,
    multiply_mode='sqrt_depth',
    no_data_parallel=False,
    norm_epsilon=1e-6,
    norm_type='layer',  # 'batch', layer', 'noam', 'none'.
    num_gpu=None,
    num_groups=8,
    num_parallel_calls=None,
    num_sampled_classes=0,
    opt_multistep_accumulate_steps=None,
    opt_zero_grads=False,
    overload_metric='',
    pack_dataset=False,
    pad_batch=False,
    pad_remover=False,  # True,
    parallel_batches=None,
    params_profile=None,
    penalty=0.1,
    post_cmd='dan',
    pre_cmd='n',
    prepend_mode='none',
    prepost_bdims='',
    private_threads=None,
    prox_bias=False,
    run_autoregressive=False,
    sampl_gold_mixin_prob=0.5,
    sampl_method='argmax',  # 'argmax' or 'random'
    sampl_prob=0.0,
    sampl_temp=1.0,
    sampl_warmup_steps=50000,
    self_attn_type='dot_attn',
    shared_embed=False,
    shared_weights=True,
    short_seq_prob=0.1,
    shuffle_size=512,
    split_tgts_chunk_len=0,
    split_tgts_max_chunks=100,
    split_to_len=0,
    src_len=0,
    steps_between_evals=None,
    stop_threshold=None,
    summarize_grads=False,
    summarize_vars=False,
    symbol_modality_shards=16,
    synthetic_data=None,
    target_frames=1,
    tgt_len=0,
    train_epochs=None,
    train_steps=1000,
    unidirectional_encoder=False,
    use_custom_ops=True,
    use_target_embed=True,
    warm_start_from='',
    warmup_steps=16000,
    weight_decay=1e-6,
    weight_noise=0.0,
    weights_fn={},
)
