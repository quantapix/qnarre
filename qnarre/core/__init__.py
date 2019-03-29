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

from .org import Org
from .proof import Proof
from .net import Boot, Net
from .doc import Genre, Doc
from .claim import Claim, Place
from .author import Author, Agent, Authority
from .narrative import Topic, Node, Narrative
from .named import Tagged, Saved, Named, Preset
from .conflict import Inherent, Conceal, Deceive, Fraud, Extort, Repeat
from .judgment import Bias, Confusion, Disregard, Fabrication, Validation
from .activism import (Activism, Exclude, Insinuate, Polarize, Recast, Elevate,
                       Victimize, Exploit, Perpetuate)
from .conjecture import (Reality, Dissent, Misreading, Fairness, Negligence,
                         Proportionality, Contradiction, Isolation, Omission,
                         Distortion)

__all__ = (
    Activism,
    Agent,
    Author,
    Authority,
    Bias,
    Boot,
    Claim,
    Conceal,
    Confusion,
    Contradiction,
    Deceive,
    Disregard,
    Dissent,
    Distortion,
    Doc,
    Elevate,
    Exclude,
    Exploit,
    Extort,
    Fabrication,
    Fairness,
    Fraud,
    Genre,
    Inherent,
    Insinuate,
    Isolation,
    Misreading,
    Named,
    Narrative,
    Negligence,
    Net,
    Node,
    Omission,
    Org,
    Perpetuate,
    Place,
    Polarize,
    Preset,
    Proof,
    Proportionality,
    Reality,
    Recast,
    Repeat,
    Saved,
    Tagged,
    Topic,
    Validation,
    Victimize,
)

all_nodes = (
    Bias,
    Confusion,
    (Contradiction, 'x'),
    Deceive,
    (Disregard, 'g'),
    (Distortion, 's'),
    Elevate,
    Extort,
    Exclude,
    Exploit,
    Fabrication,
    (Fairness, 'a'),
    (Fraud, 'u'),
    Insinuate,
    Isolation,
    Misreading,
    Negligence,
    Omission,
    Perpetuate,
    Polarize,
    Proof,
    (Proportionality, 'y'),
    Reality,
    Recast,
    (Repeat, 't'),
    Validation,
    Victimize,
)

all_genres = (
    'affidavit',
    'deposition',
    'letter',
    'message',
    'motion',
    'order',
    'report',
    'service',
    'trial',
)
