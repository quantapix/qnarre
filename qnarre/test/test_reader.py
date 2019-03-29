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

import pathlib as pth

from qnarre.pdf_old.pdf2txt import extract_text
from qnarre.reader import PDF


def a_test_reader_old():
    p = pth.Path('/tmp/qn-reader/test.pdf')
    extract_text((str(p),))


def test_reader():
    p = pth.Path('/tmp/qn-reader/test.pdf')
    pdf = PDF(p)
    pdf.text_lines()
