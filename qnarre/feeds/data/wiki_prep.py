
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

import glob

import pathlib as pth

from absl import flags

from qfeeds.data.shell import Shell


def prep(sh):
    sh.run(
        'mkdir -p extracted_articles',
        'mkdir -p intermediate_files',
        'mkdir -p final_text_file_single',
        'mkdir -p final_text_files_sharded',
        'mkdir -p final_tfrecords_sharded',
    )


def extract(sh, path):
    c = 'python -m wikiextractor.WikiExtractor {}/wikidump.xml'
    c += ' -b 1000M --processes 4 -o {}/extracted_articles'
    sh.run(c.format(path, path), )


def remove_tags(path):
    with open(path / '/intermediate_files/wikipedia.txt', 'w') as out:
        for d in glob.glob('extracted_articles/*/', recursive=False):
            for n in glob.glob(d + 'wiki_*', recursive=True):
                print(n)
                ls = []
                opened = False
                with open(n, 'r') as f:
                    for ln in f:
                        if '<doc id=' in ln:
                            opened = True
                        elif '</doc>' in ln:
                            opened = False
                            for o in ls[1:]:
                                if o != '\n':
                                    out.write(o.rstrip() + ' ')
                            out.write('\n\n')
                            ls = []
                        else:
                            if opened:
                                ls.append(ln)


def nltk_sentences(path):
    import nltk
    nltk.download('punkt')
    pin = path / 'intermediate_files/wikipedia.txt'
    pout = path / 'final_text_file_single/wikipedia.segmented.nltk.txt'
    with open(pin) as f:
        with open(pout, 'w') as out:
            for ln in f:
                if ln != '\n':
                    for s in nltk.tokenize.sent_tokenize(ln):
                        out.write(s + '\n')
                    out.write('\n')


def spacy_sentences(path):
    import spacy
    # spacy.prefer_gpu()
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    pin = path / 'intermediate_files/wikipedia.txt'
    pout = path / 'final_text_file_single/wikipedia.segmented.txt'
    with open(pin) as f:
        with open(pout, 'w') as out:
            for ln in f:
                if ln != '\n':
                    for s in nlp(ln).sents:
                        out.write(s.text + '\n')


def shard_text(path):
    pin = path / 'final_text_file_single/wikipedia.segmented.nltk.txt'
    total = 0
    with open(pin) as f:
        for ln in f:
            total += 1
    print('Input has', total, 'lines')
    shard, idx = 396000, 0
    c, ic, ls = 0, 1, []
    with open(pin) as f:
        for ln in f:
            if ((c < shard and ic < total)
                    or (c >= shard and ln != '\n' and ic < total)):
                ls.append(ln)
                c += 1
                ic += 1
            else:
                p = 'final_text_files_sharded/wikipedia.segmented.part.{}.txt'
                with open(path / p.format(idx), 'w') as out:
                    for o in ls:
                        out.write(o)
                c, ls = 0, []
                idx += 1


def preproc(sh):
    sh.run(
        'rm -rf xarg_list.txt', 'touch xarg_list.txt', 'SHARD_COUNT=0',
        'for file in final_text_files_sharded/*; do',
        '  echo ${SHARD_COUNT} >> xarg_list.txt',
        '  SHARD_COUNT=$((SHARD_COUNT+1))', 'done',
        'xargs -n 1 --max-procs=$(nproc) --arg-file=xarg_list.txt preprocessing.sh'
    )


def define_download_flags():
    flags.DEFINE_string(name='dir_data', default='.data/wiki', help='Data dir')


def main(_):
    dd = flags.FLAGS.dir_data
    path = pth.Path.cwd() / dd
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    sh = Shell(path)
    prep(sh)
    extract(Shell(pth.Path.cwd()), dd)
    remove_tags(dd)
    nltk_sentences(dd)
    # spacy_sentences(dd)
    shard_text(dd)
    preproc(sh)


if __name__ == '__main__':
    define_download_flags()
    from absl import app
    app.run(main)
