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

from data.utils import pickle_loads
from data.service import Tokenizer, Service


class Tokenizer(Tokenizer):
    def __init__(self, words=None, limit=None):
        super().__init__(Service(), words, limit)

    def from_blob(self, blob):
        import data.tokens
        for t in pickle_loads(blob, self.words):
            yield t


class Service(Service):

    bin_path = '/home/qpix/tools/spacy/bin/'
    command = bin_path + 'python spacy_server.py'
