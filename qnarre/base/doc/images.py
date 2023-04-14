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

import os
import pathlib as pth

from wand.image import Image

from .counter import counters
from .base import config, num_to_name, lister


class Images:

    kind = 'png'
    export_args = ((('scanned', '.'), ('split', '+'), ('missing', '-'),
                    ('resized', '>'), ('failed', 'F')), 'Splitting:')

    @staticmethod
    def rename_all(path):
        for i, p in enumerate(sorted(lister(path)), start=1):
            s = p.suffix
            d = p.with_name(num_to_name(i)).with_suffix(s)
            p.rename(d)

    @classmethod
    def filepath(cls, path, i):
        return path / '{}.{}'.format(num_to_name(i), cls.kind)

    def __init__(self, src, dst, width=500, blur=1):
        self.src = src
        self.dst = dst
        self.width = width
        # self.blur = 5 if dst.endswith(config.OPEN) else blur
        self.blur = blur

    def resize(self, dst, img, i=1):
        s = img.sequence
        if s and len(s) > 1:
            for p in s:
                i = p.index + 1
                with Image(image=p) as p:
                    p.alpha_channel = False
                    w = self.width
                    h = int(w / (p.width / p.height))
                    p.resize(width=w, height=h, blur=self.blur)
                    f = self.filepath(dst, i)
                    p.save(filename=str(f))
                    yield f
        else:
            img.alpha_channel = False
            w = self.width
            h = int(w / (img.width / img.height))
            img.resize(width=self.width, height=h, blur=self.blur)
            f = self.filepath(dst, i)
            img.save(filename=str(f))
            yield f

    def split_pdf(self, src, dst):
        with Image(filename=str(src), resolution=300) as pdf:
            with pdf.convert(self.kind) as img:
                yield from self.resize(dst, img)

    def resize_all(self, src, dst):
        for i, f in enumerate(sorted(str(p) for p in lister(src)), start=1):
            with Image(filename=f) as img:
                with img.convert(self.kind) as img:
                    yield from self.resize(dst, img, i)


class Pngs(Images):
    def export_all(self, ctxt, **kw):
        kw.update(ctxt=ctxt)
        with counters(self.export_args, kw) as cs:
            for p in ctxt.sources.keys():
                s = (self.src / p).with_suffix('.pdf')
                if s.exists():
                    d = (self.dst / p).with_suffix('.slides')
                    if d.exists():
                        cs.incr('.')
                    else:
                        d.mkdir(parents=True, exist_ok=True)
                        for _ in self.split_pdf(s, d):
                            pass
                        cs.incr('+')
                else:
                    cs.incr('-')
            return cs


class Jpgs(Images):

    kind = 'jpeg'

    def __init__(self, src, dst, width=1000, blur=1):
        super().__init__(src, dst, width, blur)

    def export_all(self, ctxt, **kw):
        kw.update(ctxt=ctxt)
        src = self.src / 'pictures'
        with counters(self.export_args, kw) as cs:
            for p in ctxt.sources.keys():
                s = (src / p).with_suffix('.pdf')
                if s.exists():
                    d = (self.dst / p).with_suffix('.slides')
                    if d.exists():
                        cs.incr('.')
                    else:
                        d.mkdir(parents=True, exist_ok=True)
                        for _ in self.split_pdf(s, d):
                            pass
                        cs.incr('+')
                    continue
                else:
                    s = s.with_suffix('')
                    if s.exists() and s.is_dir():
                        d = (self.dst / p).with_suffix('.slides')
                        if d.exists():
                            cs.incr('.')
                        else:
                            d.mkdir(parents=True, exist_ok=True)
                            for _ in self.resize_all(s, d):
                                pass
                            cs.incr('>')
                        continue
                    cs.incr('-')
            return cs


class Orgs(Pngs):

    org_frame = ('', '')  # None

    @classmethod
    def frame(cls):
        if not cls.org_frame:
            t = pth.Path(config.web_templates + 'frame.org').read_text()
            cls.org_frame = t.split(r'{% block frame_content %}')
        return cls.org_frame

    def __init__(self, src, dst, width=750, blur=1):
        super().__init__(src, dst, width, blur)

    def export_all(self, ctxt, **kw):
        kw.update(ctxt=ctxt)
        with counters(self.export_args, kw) as cs:
            for c in ('affidavits', 'hearings'):
                # 'exhibits', 'messages',
                # 'pictures', 'reports', 'submissions', 'trials',
                # 'discoveries', 'financial', 'letters', 'orders',
                # 'services', 'transcripts'):
                for s in lister(self.src / c, suffs=('.pdf', )):
                    # print(s)
                    d = (self.dst / s.relative_to(self.src)).with_suffix('')
                    if d.exists():
                        cs.incr('.')
                    else:
                        b = d.parent
                        d.mkdir(parents=True, exist_ok=True)
                        f, e = self.frame()
                        for p in self.split_pdf(s, d):
                            p = p.relative_to(b)
                            f += '#+NAME: {}\n[[./{}]]\n'.format(
                                p.stem, str(p))
                        d.with_suffix('.org').write_text(f + e)
                        cs.incr('+')
                for s in lister(self.src / c, suffs=('.org', )):
                    d = (self.dst / s.relative_to(self.src))
                    try:
                        d.unlink()
                    except FileNotFoundError:
                        pass
                    d.symlink_to(os.path.relpath(s, d.parent))
            return cs


if __name__ == '__main__':
    cwd = pth.Path.cwd()
    Images.rename_all(cwd)
    with os.scandir(cwd) as es:
        for e in es:
            p = pth.Path(e.path)
            if p.is_dir():
                Images.rename_all(p)
