
__all__ = ('generate',)

import contextlib as cl

from fabric.api import cd, run
from fabric.api import task, settings

from .base import Config
from .corenlp import CoreNLP
from .tboard import TBoard


class Protos(Config):

    def extract(self):
        c = CoreNLP(self.root)
        c.sources
        with cd(self.root + 'corenlp'):
            if not exists('stanford-corenlp-*-sources.jar'):
                TBoard(self.root).extract()

        if names is None:
            return self.all_repos.values()
        return [self.all_repos[n.strip()] for n in names.split(',')]

    def clean(self):
        with cd(self.root):
            for r in repos:
                getattr(r, method)(*args)


@cl.contextmanager
def protos(args, kw):
    ps = Protos(*args, **kw)
    ps.extract()
    yield ps
    ps.clean()


@task
def generate(root=None):
    with settings(output_prefix=False):
        with protos(root) as ps:
            ps.generate()
