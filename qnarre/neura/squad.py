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

from datetime import datetime

import tensorflow as T

from qnarre.neura import bert
from qnarre.neura.layers import Squad
from qnarre.feeds.dset.squad_ds import dataset as squad_ds

from qnarre.neura import utils as U

KS = T.keras

# KL = KS.layers
# KC = KS.callbacks


def model_for(params):
    PS = params
    FS = PS.features
    seq = KS.Input(**FS.input_kw(FS.SEQ))
    typ = KS.Input(**FS.input_kw(FS.TYP))
    opt = KS.Input(**FS.input_kw(FS.OPT))
    beg = KS.Input(**FS.input_kw(FS.BEG))
    end = KS.Input(**FS.input_kw(FS.END))
    uid = KS.Input(**FS.input_kw(FS.UID))
    ins = [seq, typ, opt, beg, end, uid]
    y = Squad(PS)(ins)
    m = KS.Model(inputs=ins, outputs=[y])
    m.compile(
        optimizer=U.adam_opt(PS),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return m


def dset_for(kind, params):
    PS = params
    ds, data = squad_ds(kind, PS)
    if kind == 'train':
        ds = ds.shuffle(buffer_size=50000)
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=T.data.experimental.AUTOTUNE)
    return ds, data


def main(_):
    sid = datetime.now().strftime('%Y%m%d-%H%M%S')
    PS = bert.load_params().override(_params)
    U.train_sess(sid, PS, model_for, dset_for)


_params = dict(
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
)

_fspecs = {
    'SEQ': {
        'name': 'sequence',
        'dtype': 'int32',
        'shape': (None, ),
    },
    'TYP': {
        'name': 'types',
        'dtype': 'int32',
        'shape': (None, ),
    },
    'OPT': {
        'name': 'optimal',
        'dtype': 'int32',
        'shape': (None, ),
    },
    'BEG': {
        'name': 'begin',
        'dtype': 'int32',
        'shape': (),
    },
    'END': {
        'name': 'end',
        'dtype': 'int32',
        'shape': (),
    },
    'UID': {
        'name': 'uid',
        'dtype': 'int32',
        'shape': (),
    },
}


class Features(U.Features):
    def __init__(self, params, **kw):
        super().__init__(**kw)
        PS = params
        sh = (PS.max_seq_len, )
        self.shapes[self.SEQ] = sh
        self.shapes[self.TYP] = sh
        self.shapes[self.OPT] = sh


_params.update(
    features=Features(_params, specs=_fspecs),
    data_dir='.data/squad',
    log_dir='.model/squad/logs',
    model_dir='.model/squad',
    save_dir='.model/squad/save',
)

if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    bert.load_flags()
    from absl import flags as F
    F.DEFINE_float('null_score_diff_threshold', None, '')
    F.DEFINE_float('warmup_split', None, '')
    F.DEFINE_integer('max_ans_len', None, '')
    F.DEFINE_integer('max_qry_len', None, '')
    F.DEFINE_integer('n_best_size', None, '')
    F.DEFINE_integer('seq_stride', None, '')
    from absl import app
    app.run(main)

###
"""
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /results/predictions.json
"""
