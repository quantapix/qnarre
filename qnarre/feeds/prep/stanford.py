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

from data.utils import normalize
from data.tokens import Embeds, Token, Span
from data.service import Tokenizer, Service


class Tokenizer(Tokenizer):

    params = {
        'params': {
            'properties':
            str({
                'inputFormat':
                'text',
                'outputFormat':
                'serialized',
                'annotators':
                'tokenize,ssplit,lemma,pos,ner,depparse',
                'serializer':
                'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer',
            })
        }
    }

    fix = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}'
    }

    def __init__(self, words=None, limit=None):
        super().__init__(Service(), words, limit)

    def from_blob(self, blob):
        for s in self._to_sentences(blob):
            for t in s.token:
                w = t.word
                w = self.words[normalize(self.fix.get(w, w))]
                s = Span(t.beginChar, t.endChar)
                yield Token(
                    word=w,
                    span=s,
                    pos=self.words[normalize(t.pos)],
                    lemma=self.words[normalize(t.lemma)],
                    ner=self.words[normalize(t.ner)],
                    embeds=Embeds())

    def _to_sentences(self, blob):
        from data.CoreNLP_pb2 import Document
        d = Document()
        from google.protobuf.internal.decoder import _DecodeVarint
        s, p = _DecodeVarint(blob, 0)
        d.ParseFromString(blob[p:p + s])
        return d.sentence


class Service(Service):

    class_path = '/home/qpix/tools/corenlp/'
    command = ('java -mx8g -cp ' + class_path + '* ' +
               'edu.stanford.nlp.pipeline.StanfordCoreNLPServer ' +
               '-port 9000 -timeout 15000')
