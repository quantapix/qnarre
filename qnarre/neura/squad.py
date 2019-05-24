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
# https://arxiv.org/pdf/1810.04805.pdf
# https://arxiv.org/pdf/1806.03822.pdf
# https://arxiv.org/pdf/1606.05250.pdf

from qnarre.neura import bert
from qnarre.neura.session import session_for

from qnarre.feeds.dset.squad import dset as squad_dset
from qnarre.neura.layers.squad import model as squad_model
from qnarre.neura.layers.squad import adapter as squad_adapter


def dset_for(ps, kind):
    ds, feats = squad_dset(ps, kind)
    if kind == 'train':
        ds = ds.shuffle(10000)
    ds = ds.batch(1 if ps.eager_mode else ps.batch_size)
    ds = ds.map(lambda d: squad_adapter(ps, feats, d))
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(ps, compiled=False):
    m = squad_model(ps)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
            # target_tensors=[ins[4]],
        )
    print(m.summary())
    return m


params = dict(
    batch_size=8,
    learn_rate=5e-6,
    max_ans_len=30,
    max_qry_len=64,
    max_seq_len=384,
    n_best_size=20,
    null_score_diff_threshold=0.0,
    seq_stride=128,
    train_epochs=2.0,
    use_fp16=False,
    use_xla=False,
    warmup_split=0.1,
    #
    dset='squad',
    dset_subset='reply_spans',  # 'query_valid', 'possibles'
    model='squad',
    optimizer='sgd',
    eager_mode=True,
)


def main(_):
    ps = bert.load_params().override(params)
    # tf.autograph.set_verbosity(1)
    # print(tf.autograph.to_code(Trafo.embed.python_function))
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    from absl import app, flags, logging
    logging.set_verbosity(logging.INFO)  # DEBUG
    bert.load_flags()
    flags.DEFINE_integer('xxx', None, '')
    app.run(main)

###
"""
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /results/predictions.json
"""
