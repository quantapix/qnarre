# Copyright 2018 Quantapix Authors. All Rights Reserved.
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

# import pytest

from qnarre.patch import Patch


def test_patch_1():
    a = '\n'.join(('0', '11', '2', 'a', 'c',
                   'd', 'ee', 'g', 'h', 'h2', 'h3', 'h4',
                   'i', 'j', 'l', 'm', 'n', 'o',
                   '0', '11', '2', 'a', 'c',
                   'd', 'ee', 'g', 'h', 'h2', 'h3', 'h4',
                   'i', 'j', 'l'))
    b = '\n'.join(('0', '1', '2', '3', 'a', 'b', 'c',
                   'd', 'e', 'f', 'g', 'h', 'h2', 'h3', 'h4',
                   'i', 'j', 'k', 'l', 'm', 'n', 'o'
                   '0', '1', '2', '3', 'a', 'b', 'c',
                   'd', 'e', 'f', 'g', 'h', 'h2', 'h3', 'h4',
                   'i', 'j', 'k', 'l'))
    p = Patch.create(a, b)
    # print(p.chunks)
    b2 = p.apply(a)
    print(b2)
    assert b == b2
