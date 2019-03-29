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

from django.conf.urls import url

from . import views as vs

app_name = 'qnarre'

urlpatterns = [
    url(r'^$', vs.HomeView.as_view(), name='home'),
    url(r'^story$', vs.StoryView.as_view(), name='story'),
    url(r'^blog$', vs.BlogView.as_view(), name='blog'),
    url(r'^about$', vs.AboutView.as_view(), name='about'),
    url(r'^agents$', vs.AgentsView.as_view(), name='agents'),
    url(r'^docs$', vs.DocsView.as_view(), name='docs'),
    url(r'^p/(?P<slug>[a-z0-9_-]+)$', vs.PartView.as_view(), name='part'),
    url(r'^register/$', vs.RegisterView.as_view(), name='register'),
    url(r'^load/$', vs.load, name='load'),
]
