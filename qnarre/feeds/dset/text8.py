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

import json
import lzma
import unicodedata
import zipfile

import pathlib as pth

from qnarre.neura import tf
from qnarre.feeds.prep import utils as U
from qnarre.feeds.prep import layout as L


def dataset(ps, kind):
    ps.update(layout=L.Topics(ps.tokenizer(_reader(ps, kind))))
    return tf.Dataset.from_generator(
        lambda: _converter(ps, kind),
        ps.features.tf_dtypes,
        ps.features.tf_shapes,
    )


def prep(ps, kind):
    data = zipfile.ZipFile('text8.zip').extractall()
    data = open('text8', 'r', encoding='utf-8').read()
    num_test_chars = 5000000
    train_data = data[:-2 * num_test_chars]
    valid_data = data[-2 * num_test_chars:-num_test_chars]
    test_data = data[-num_test_chars:]
    for fn, part in [('train.txt', train_data), ('valid.txt', valid_data),
                     ('test.txt', test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing...')
        # Change space ' ' to underscore '_'
        part_str = ' '.join(['_' if c == ' ' else c for c in part.strip()])
        print('- Writing...')
        f = open(fn, 'w').write(part_str)
        f = open(fn + '.raw', 'w', encoding='utf-8').write(part)


def _reader(ps, kind):
    p = pth.Path(ps.data_dir)
    for n in _names[kind]:
        with lzma.open(p / (n + '.json.xz'), mode='rt') as f:
            for t in json.load(f)['data']:
                cs = []
                for p in t['paragraphs']:
                    ctx = _normalize(p['context'])
                    qs = []
                    for q in p['qas']:
                        ans = []
                        for a in q.get('answers', ()):
                            tx = _normalize(a['text'])
                            s = a['answer_start']
                            if ctx.find(tx, s) == s:
                                ans.append(
                                    L.Answer(
                                        text=tx,
                                        tokens=L.Tokens(),
                                        span=L.Span(s, s + len(tx)),
                                        uid=U.next_uid('answer'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        vs = []
                        for v in q.get('plausible_answers', ()):
                            tx = _normalize(v['text'])
                            s = v['answer_start']
                            if ctx.find(tx, s) == s:
                                vs.append(
                                    L.Answer(
                                        text=tx,
                                        tokens=L.Tokens(),
                                        span=L.Span(s, s + len(tx)),
                                        uid=U.next_uid('answer'),
                                    ))
                            else:
                                print('Mismatched', ctx[:20], tx[:20])
                        qs.append(
                            L.Question(
                                qid=q['id'],
                                text=_normalize(q['question']),
                                unfit=q.get('is_impossible', False),
                                tokens=L.Tokens(),
                                answers=ans,
                                viables=vs,
                            ))
                    cs.append(
                        L.Context(
                            text=ctx,
                            tokens=L.Tokens(),
                            questions=qs,
                        ))
                yield L.Topic(
                    title=_normalize(t['title']),
                    contexts=cs,
                )


def _normalize(txt):
    return ' '.join(unicodedata.normalize('NFD', txt).split())


def _converter(PS, kind):
    FS = ps.features
    for _, c, q, ans in ps.layout.answers():
        cs, qs = c.tokens, q.tokens
        if ps.max_qry_len:
            qs = qs[:ps.max_qry_len]
        end, ql = len(cs), len(qs)
        sl = ps.max_seq_len - ql - 3
        ss, b = [], 0
        while b < end:
            e = end
            e = (b + sl) if e - b > sl else e
            ss.append(L.Span(begin=b, end=e))
            if e == end:
                break
            b = min(e, b + ps.doc_stride)
        ql += 2
        for si, s in enumerate(ss):
            seq = [ps.CLS] + qs + [ps.SEP] + cs[s.begin:s.end] + [ps.SEP]
            typ = [0] * ql + [1] * (len(s) + 1)

            def _optim(i):
                o, oi = None, -1
                for s2i, s2 in enumerate(ss):
                    if i >= s2.begin and i < s2.end:
                        left = i - s2.begin
                        right = s2.end - i - 1
                        o2 = min(left, right) + 0.01 * len(s2)
                        if o is None or o2 > o:
                            o, oi = o2, s2i
                return 1 if si == oi else 0

            opt = [0] * ql
            opt += [_optim(idx) for idx in range(s.begin, s.end)] + [0]
            assert len(seq) == len(typ) == len(opt)
            pad = [0] * (ps.max_seq_len - len(seq))
            if pad:
                seq += pad
                typ += pad
                opt += pad
            beg, end = 0, 0
            if kind == 'train':
                if not q.unfit:
                    beg, end = ans.span.begin, ans.span.end
                    if b >= s.begin and e <= s.end:
                        beg += ql - s.begin
                        end += ql - s.end
            yield seq, typ, opt, beg, end, ans.uid


_names = {
    'train': ('train-v2.0', 'train-v1.1'),
    'test': ('dev-v2.0', 'dev-v1.1'),
}
