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

import csv

import pathlib as pth

from qnarre import load_from


class Report:
    fields = (
        'Activism',
        'Agency',
        'Author',
        'Coherence',
        'Credibility',
        'Date',
        'Fragment',
        'Genre',
        'Judgment',
        'Kind',
        'Loss',
        'Name',
        'Narrative',
        'Page',
        'Para',
        'Reality',
        'Source',
        'Text',
        'Title',
        'Topic',
        'Turmoil',
        'Type',
    )
    exclude = ()

    def __init__(self, dst):
        self.csv = csv.DictWriter(dst, self.fields)
        self.csv.writeheader()

    def write(self, node):
        if node.__class__ not in self.exclude:
            ls = node.fields
            if isinstance(ls, list):
                for fs in ls:
                    self.csv.writerow(fs)
            else:
                self.csv.writerow(ls)


def report(root, **kw):
    kw.update(root=root)
    print('Loading from {}...'.format(str(root)))
    ns = set(n for n in load_from(pth.Path('merged.org'), **kw).net.nodes())
    for n in ns:
        print(n)
    print('...done')
    print('Reporting...')
    with open(root / 'merged.csv', 'w', newline='') as f:
        r = Report(f)
        for n in ns:
            r.write(n)
    print('...done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('-r', '--root', help='Path to root', default=None)
    args = args.parse_args()
    report(pth.Path.cwd() / (args.root or 'sample'))
