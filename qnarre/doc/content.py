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

import mimetypes

import pprint as pp

from .log import Logger
from .base import config, digest
from .nominals import para_join, nominal
from .resource import Resource

log = Logger(__name__)


class Registry(Resource):
    @classmethod
    def globals(cls):
        return globals()

    def __init__(self, elems=None, **kw):
        super().__init__(elems, **kw)
        for k, v in tuple(self.items()):
            v = self.add_once(v)
            if v:
                self[k] = v
            else:
                del self[k]

    def __repr__(self):
        es = {k: v for k, v in self.items() if v and k is not v}
        es = pp.pformat(es, indent=4)
        return '{}({})'.format(type(self).__name__, es)

    @property
    def elems(self):
        return [v for k, v in self.items() if v and k is not v]

    rename_msg = Resource.rename

    def add_once(self, v):
        if isinstance(v, tuple):
            return tuple(self.add_once(i) for i in v)
        return super().add_once(v)

    def eml_content(self, part, name, _):
        return part.get_content()

    def eml_register(self, name, vs):
        vs = self.add_once(vs)
        try:
            os = self[name]
            assert os == vs or nominal(para_join(os)) == nominal(para_join(vs))
        except KeyError:
            if vs:
                self[name] = vs
        return vs

    def extract(self, name, raw):
        vs = []
        for i, p in enumerate(raw.walk()):
            if self.check_type(p):
                v = self.eml_content(p, name, i)
                if v:
                    vs.append(v)
        return self.eml_register(name, tuple(vs))


class Texts(Registry):

    _res_path = config.qnar_dst + 'texts.qnr'

    def check_type(self, part):
        return part.get_content_type() == 'text/' + config.PLAIN

    def eml_register(self, name, vs):
        return vs

    def register(self, name, paras):
        super().eml_register(name, tuple(paras))

    def expand(self, name, paras):
        if name in self:
            del self[name]
        self.register(name, paras)


class Htmls(Registry):

    _res_path = config.qnar_dst + 'htmls.qnr'

    def check_type(self, part):
        return part.get_content_type() == 'text/' + config.HTML

    def eml_content(self, part, name, i):
        v = part.get_content()
        if v:
            d = digest(v.encode())
            if d not in self:
                self[d] = 'aaa'
                # print('\nhtml', name, i, d)
            # return d


class Attms(Registry):

    _res_path = config.qnar_dst + 'attms.qnr'

    def check_type(self, part):
        return part.get_content_maintype() not in ('multipart', 'text')

    def eml_content(self, part, name, i):
        try:
            v = part.get_content()
            if v:
                d = digest(v)
                if d not in self:
                    self[d] = 'aaa'
                    t = mimetypes.guess_extension(part.get_content_type())
                    print('\n', t, name, i, d, part.get_content_maintype(),
                          part.get_filename())
        except Exception as e:
            print(e)
            log.error('Error getting content {} {} {}', name,
                      part.get_content_maintype(), part.get_filename())
