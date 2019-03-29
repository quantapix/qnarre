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

from .qnn import Qnn
from .log import Logger
from .base import config
from .args import BArgs, CArgs, MArgs
from .resource import resource
from .dispatch import Dispatch

log = Logger(__name__)


def filt_mbox():
    a = MArgs().parse_args()
    with resource(Dispatch.create(a.base)) as d:
        d.filt_mbox(**a.kw)


def merge_mbox():
    a = MArgs().parse_args()
    with resource(Dispatch.create(a.base)) as d:
        d.merge_mbox(**a.kw)


def strip_mbox():
    a = MArgs().parse_args()
    with resource(Dispatch.create(a.base, a.realm)) as d:
        d.strip_mbox(**a.kw)


def import_from(src):
    a = MArgs().parse_args()
    with resource(Dispatch.create(a.base, a.realm)) as d:
        d.import_from(src, **a.kw)


def import_main():
    import_from(config.main_src)


def import_blog():
    import_from(config.blog_src)


def import_priv():
    import_from(config.priv_src)


def import_docs():
    import_from(config.docs_src)


def import_sbox():
    import_from(config.sbox_src)


def import_mbox():
    import_from(config.mbox_src)


def import_tbox():
    import_from(config.tbox_src)


def import_bbox():
    import_from(config.bbox_src)


def import_pics():
    import_from(config.PICS)


def protect():
    a = BArgs().parse_args()
    Dispatch.create(a.base, config.PROT).protect(**a.kw)


def redact():
    a = BArgs().parse_args()
    Dispatch.create(a.base, config.PUBL).redact(**a.kw)


def obfuscate():
    a = BArgs().parse_args()
    Dispatch.create(a.base, config.OPEN).obfuscate(**a.kw)


def check_recs():
    a = MArgs().parse_args()
    with resource(Dispatch.create(a.base, a.realm)) as d:
        d.check_recs(**a.kw)


def graph_recs():
    a = CArgs().parse_args()
    with resource(Dispatch.create(a.base, a.realm)) as d:
        d.graph_recs()


def qnn_setup():
    a = BArgs().parse_args()
    Qnn.create(a.base, config.PROT).setup(**a.kw)


def qnn_learn():
    a = BArgs().parse_args()
    Qnn.create(a.base, config.PROT).learn(**a.kw)


def qnn_guess():
    a = BArgs().parse_args()
    Qnn.create(a.base, config.PROT).guess(**a.kw)


def export_all(kind):
    a = CArgs().parse_args()
    assert a.realm
    with resource(Dispatch.create(a.base, a.realm)) as d:
        d.export_all(kind, **a.kw)


def export_orgs():
    export_all(config.ORGS)


def export_pngs():
    export_all(config.IMGS)


def export_jpgs():
    export_all(config.PICS)


def export_mbox():
    export_all(config.MBOX)


def export_blog():
    export_all(config.BLOG)
