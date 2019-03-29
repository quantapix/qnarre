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

from django.apps import apps
from django.views import generic
from django.shortcuts import render
from django.dispatch import receiver
from django.http import HttpResponseRedirect
from django.contrib.auth.signals import user_logged_in, user_logged_out

from ..section import Session

from .forms import LoadForm

sessions = {}


@receiver(user_logged_in)
def create_session(sender, user, **kw):
    return sessions.setdefault(user.username, Session(SectionView.app()))


@receiver(user_logged_out)
def delete_session(sender, user, **kw):
    try:
        del sessions[user.username]
    except KeyError:
        pass


class TemplateView(generic.TemplateView):
    def get_template_names(self):
        return ['qnarre/' + self.section + '.html']

    def get_context_data(self, **kw):
        c = super().get_context_data(**kw)
        c['section'] = sc = self.section
        s = self.request.GET.get('show', None)
        c['actives'] = set((sc, s) if s else (sc, ))
        return c


class AboutView(TemplateView):

    section = 'about'


class HomeView(TemplateView):

    section = 'home'

    def get_context_data(self, **kw):
        c = super().get_context_data(**kw)
        ss = create_session(None, self.request.user)
        b = ss.parts_all['01-02-01_000']
        c['blurbs'] = str(b.body).split('\n\n')
        return c


class SectionView(TemplateView):

    _app = None

    groups_name = 'Topics'
    subgroups_name = 'Subjects'

    @classmethod
    def app(cls):
        if cls._app is None:
            cls._app = apps.get_app_config('qnarre')
        return cls._app

    def subtemplate(self, name='part'):
        return 'qnarre/parts/{}.html'.format(name)

    def get_context_data(self, slug=None, **kw):
        c = super().get_context_data(**kw)
        ss = create_session(None, self.request.user)
        c['subtemplate'] = self.subtemplate()
        sc = getattr(ss, self.section)
        c['groups'] = gs = sc.groups
        c['groups_name'] = self.groups_name
        sg = sc.subgroups
        c['subgroups'] = sg[gs[0]] if len(gs) == 1 else sg
        c['subgroups_name'] = self.subgroups_name
        ps = sc.extent
        if slug:
            s = ss.parts_all[slug]
            try:
                ps = sc.parts[s]
            except KeyError:
                ps = (s, )
        if len(ps) == 1:
            c['part'] = p = ps[0]
            c['subtemplate'] = self.subtemplate(p.get_template())
        else:
            c['parts'] = ps
        return c


class StoryView(SectionView):

    section = 'story'

    def subtemplate(self, name=None):
        return super().subtemplate(name or self.section)


class BlogView(StoryView):

    section = 'blog'


class AgentsView(SectionView):

    section = 'agents'
    subgroups_name = 'Roles'


class DocsView(SectionView):

    section = 'docs'


class PartView(SectionView):
    @property
    def section(self):
        return self.request.GET.get('section', 'docs')


class RegisterView(SectionView):

    section = 'blog'


def load(request):
    if request.method == 'POST':
        f = LoadForm(request.POST)
        if f.is_valid():
            print('***Loading***')
            return HttpResponseRedirect('/')
    else:
        f = LoadForm()
    return render(request, 'qnarre/load.html', {'form': f})
