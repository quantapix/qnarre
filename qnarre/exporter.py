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

import markdown

import pathlib as pth

from .base import config

markdown_settings = {
    'extension_configs': {
        'markdown.extensions.codehilite': {
            'css_class': 'highlight'
        },
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'extensions': [
        'markdown.extensions.codehilite', 'markdown.extensions.extra',
        'markdown.extensions.meta', 'markdown.extensions.toc',
        'markdown.extensions.fenced_code'
    ],
    'output_format':
    'html5',
}


class Exporter:

    _topic = None
    _subject = None

    html_frame = None

    markdown = markdown.Markdown(**markdown_settings)

    @classmethod
    def frame(cls):
        if not cls.html_frame:
            t = pth.Path(config.web_templates + 'frame.html').read_text()
            t = t.replace(r'{% endblock %}', '')
            fb, fe = t.split(r'{% block frame_content %}')
            cls.html_frame = (fb, r'<div class="container">', link_begin,
                              link_title, link_end, r'</div>', fe)
        return cls.html_frame

    def __init__(self, **kw):
        super().__init__(**kw)

    def mboxer(self, ctype=config.HTML, **kw):
        yield from self.hdr.mboxer(**kw)
        yield 'subject', self.subject(**kw)
        if ctype == config.PLAIN:
            yield 'text/' + ctype, '\n'.join(self.plainer(**kw))
        else:
            yield 'text/' + ctype, '\n'.join(self.htmer(self.frame(), **kw))

    def plainer(self, **kw):
        yield self.text(**kw)

    def htmer(self, frame=None, **kw):
        if frame:
            yield frame[0]
            yield frame[1]
            yield from self.hdr.htmer(None, frame, **kw)
        yield self.markdown.reset().convert(self.text(**kw))
        if frame:
            yield frame[-3]
            yield frame[-2]
            yield frame[-1]

    def blogger(self, **kw):
        yield from self.hdr.blogger(**kw)
        yield self.text(**kw)
        yield from self.hdr.footer(**kw)


link_begin = """
<div class="row {}">
<div class="col-10">
<div class="card with-margin" style="background-color: #{};">
<div class="card-block">
"""

link_title = """
<h6 class="text-muted">{} <strong>{}:</strong></h6>
"""

link_end = """
</div>
</div>
</div>
</div>
"""
