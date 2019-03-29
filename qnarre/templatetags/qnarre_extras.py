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

from django.template import Library

register = Library()


@register.filter
def get_active(dictionary, key):
    return 'active' if key in dictionary else ''


@register.filter
def get_show(dictionary, key):
    return 'show' if str(key) in dictionary else ''


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


@register.filter
def joinby(vs, sep):
    return sep.join(vs)
