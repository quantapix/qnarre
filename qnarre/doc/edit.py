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

import re
import math as ma
import random as ra

from .base import config
from .date import slugify


def fudge(count=None):
    n = ra.randrange(5, 10) if count is None else count

    def choice():
        s = abs(ma.ceil(ra.normalvariate(4.0, 3.0))) + 1
        return ra.choice(lorem[s])

    return ' '.join(choice() for _ in range(n))


fs = r'(?am)'

r_pat = r'\(:qurl:([a-zA-Z0-9/_.-]+)\)'
r_pat = re.compile(fs + r_pat)


def resolve(txt, ctxt):
    ss = ctxt.sources

    def swap(match):
        return '(/p/{})'.format(slugify(ss.res_ref(match.group(1))))

    return r_pat.sub(swap, txt)


e_pat = r'[a-zA-Z]+'
e_pat = re.compile(fs + e_pat)


def protect(txt):
    def swap(match):
        m = match.group(0)
        if m not in config.alloweds:
            w = ra.choice(lorem[len(m)])
            return w.title() if match.group(0).istitle() else w

    if txt is not None:
        return e_pat.sub(swap, txt)


def redact(txt):
    def swap(match):
        m = match.group(0)
        try:
            return config.substitutes[m]
        except KeyError:
            return m

    if txt is not None:
        return e_pat.sub(swap, txt)


def obfuscate(txt):
    def swap(match):
        w = ra.choice(lorem[len(match.group(0))])
        return w.title() if match.group(0).istitle() else w

    if txt is not None:
        return e_pat.sub(swap, txt)


lorem = {
    1: ['a', 'm'],
    2: [
        'ab', 'ac', 'ad', 'at', 'de', 'do', 'ea', 'eo', 'et', 'eu', 'ex', 'id',
        'in', 'me', 'ne', 'ob', 're', 'se', 'si', 'te'
    ],
    3: [
        'ars', 'aut', 'cum', 'cur', 'dis', 'ego', 'eis', 'eos', 'est', 'eum',
        'hac', 'hic', 'his', 'hoc', 'hos', 'iam', 'mea', 'nam', 'non', 'nos'
    ],
    4: [
        'amet', 'anim', 'apta', 'apud', 'aute', 'bene', 'cara', 'dabo', 'dixi',
        'duis', 'eius', 'elit', 'enim', 'erat', 'erit', 'esse', 'fuga', 'haec',
        'huic', 'illa'
    ],
    5: [
        'aequo', 'alias', 'aliis', 'animi', 'animo', 'artis', 'atque', 'autem',
        'causa', 'credo', 'culpa', 'deero', 'dicam', 'dicta', 'digna', 'dolor',
        'eadem', 'eaque', 'earum', 'error'
    ],
    6: [
        'aetate', 'aliqua', 'artium', 'beatae', 'causae', 'causis', 'cillum',
        'civium', 'cumque', 'custos', 'dolore', 'eisdem', 'etenim', 'evenit',
        'facere', 'fateor', 'frater', 'fugiat', 'genere', 'gloria'
    ],
    7: [
        'actione', 'adsumet', 'aedilis', 'aliquam', 'aliquid', 'aliquip',
        'angulis', 'aperiam', 'artibus', 'athenis', 'ceteros', 'cogebat',
        'cognita', 'commodi', 'commodo', 'communi', 'constet', 'curulis',
        'debitis', 'deditus'
    ],
    8: [
        'adipisci', 'amicitia', 'amicorum', 'arbitror', 'certamen', 'communia',
        'conferam', 'consequi', 'corporis', 'corrupti', 'cuiusdam', 'declarat',
        'delectus', 'deleniti', 'deserunt', 'dicendum', 'dignitas', 'eligendi',
        'expedita', 'expetunt'
    ],
    9: [
        'accusamus', 'aequitate', 'angustiis', 'appellant', 'assumenda',
        'clamabunt', 'cognoscas', 'concedunt', 'concordia', 'confidant',
        'consecuti', 'consequat', 'consulatu', 'contineri', 'cumulatus',
        'cupidatat', 'desinamus', 'devenimus', 'discrimen', 'disputant'
    ],
    10: [
        'adhibeatur', 'architecto', 'asperiores', 'aspernatur', 'blanditiis',
        'ceterosque', 'cognitione', 'concessero', 'consecutus', 'consulatum',
        'consumendi', 'copioseque', 'cupiditate', 'dissentire', 'disserebam',
        'distinctio', 'doloremque', 'eloquentia', 'exciderunt', 'fuerintque'
    ],
    11: [
        'accommodata', 'accusantium', 'adipisicing', 'admirabiles',
        'aristotelem', 'auctoritate', 'cogitatione', 'cognitionem',
        'concesserit', 'consectetur', 'consequatur', 'converteris',
        'dignissimos', 'disciplinae', 'efflorescat', 'elocutionem',
        'eloquentiam', 'eloquentium', 'excellentis', 'exsanguique'
    ],
    12: [
        'cogitatisque', 'consequuntur', 'exercitation', 'exstitissent',
        'immortalibus', 'intellegimus', 'magnitudinem', 'perspiciatis',
        'professioque', 'theophrastum', 'tractationem', 'voluptatibus'
    ],
    13: [
        'asperitatibus', 'clarissimique', 'complectantur', 'conformatione',
        'perfectiusque', 'philosophorum', 'reprehenderit'
    ],
    14: [
        'adulescentulis', 'adulescentulus', 'commentariolis', 'eloquentissimi',
        'exercitationem', 'exercitationis', 'necessitatibus', 'perturbationem',
        'praeclarissima'
    ],
    15: ['disputationibus', 'eruditissimorum'],
    16: ['exercitationibus'],
    17: ['commentaationibus'],
    18: ['exerciriolisonibus'],
    19: ['exercitcommentariol'],
    20: ['exercitationcomments'],
    21: ['exercitationicommebus'],
    22: ['excommentariolisanibus'],
    23: ['commentariolisationibus'],
}
"""
ws = ('a', 'ab', 'ac', 'accommodata', 'accusamus', 'accusantium', 'actione', 'ad', 'adhibeatur', 'adipisci', 'adipisicing', 'admirabiles', 'adsumet', 'adulescentulis', 'adulescentulus', 'aedilis', 'aequitate', 'aequo', 'aetate', 'alias', 'aliis', 'aliqua', 'aliquam', 'aliquid', 'aliquip', 'amet', 'amicitia', 'amicorum', 'angulis', 'angustiis', 'anim', 'animi', 'animo', 'aperiam', 'appellant', 'apta', 'apud', 'arbitror', 'architecto', 'aristotelem', 'ars', 'artibus', 'artis', 'artium', 'asperiores', 'asperitatibus', 'aspernatur', 'assumenda', 'at', 'athenis', 'atque', 'auctoritate', 'aut', 'aute', 'autem', 'beatae', 'bene', 'blanditiis', 'cara', 'causa', 'causae', 'causis', 'certamen', 'ceteros', 'ceterosque', 'cillum', 'civium', 'clamabunt', 'clarissimique', 'cogebat', 'cogitatione', 'cogitatisque', 'cognita', 'cognitione', 'cognitionem', 'cognoscas', 'commentariolis', 'commodi', 'commodo', 'communi', 'communia', 'complectantur', 'concedunt', 'concesserit', 'concessero', 'concordia', 'conferam', 'confidant', 'conformatione', 'consectetur', 'consecuti', 'consecutus', 'consequat', 'consequatur', 'consequi', 'consequuntur', 'constet', 'consulatu', 'consulatum', 'consumendi', 'contineri', 'converteris', 'copioseque', 'corporis', 'corrupti', 'credo', 'cuiusdam', 'culpa', 'cum', 'cumque', 'cumulatus', 'cupidatat', 'cupiditate', 'cur', 'curulis', 'custos', 'dabo', 'de', 'debitis', 'declarat', 'deditus', 'deero', 'delectus', 'deleniti', 'depulsi', 'deserunt', 'desinamus', 'devenimus', 'dicam', 'dicatur', 'dicendi', 'dicendo', 'dicendum', 'dicta', 'digna', 'dignissimos', 'dignitas', 'dis', 'disciplinae', 'discrimen', 'disputant', 'disputationibus', 'dissentire', 'disserant', 'disserebam', 'distinctio', 'dixi', 'diximus', 'dixisti', 'do', 'doctrinae', 'dolor', 'dolore', 'dolorem', 'doloremque', 'dolores', 'doloribus', 'dolorum', 'ducimus', 'duis', 'ea', 'eadem', 'eaque', 'earum', 'efflorescat', 'ego', 'eis', 'eisdem', 'eius', 'eiusmod', 'elaborare', 'elegantia', 'eligendi', 'elit', 'elocutionem', 'eloquentia', 'eloquentiam', 'eloquentissimi', 'eloquentium', 'enim', 'eo', 'eos', 'erat', 'erit', 'error', 'eruditissimorum', 'esse', 'esset', 'est', 'et', 'etenim', 'etiam', 'eu', 'eum', 'eveniet', 'evenit', 'ex', 'excellentis', 'excepteur', 'excepturi', 'exciderunt', 'exercitation', 'exercitationem', 'exercitationibus', 'exercitationis', 'expedita', 'expetunt', 'explicabo', 'explicata', 'explicet', 'exsanguique', 'exstitissent', 'facere', 'faceret', 'facilis', 'fateor', 'fluctibus', 'frater', 'fraus', 'fuerint', 'fuerintque', 'fuga', 'fugiat', 'fugit', 'genere', 'gentium', 'gestu', 'gloria', 'gravis', 'gravitate', 'gymnasia', 'habet', 'hac', 'haec', 'harum', 'hic', 'his', 'histrionum', 'hoc', 'homines', 'hominum', 'hortanti', 'hortemurque', 'hos', 'huic', 'iam', 'id', 'ignorat', 'illa', 'illam', 'illi', 'illis', 'illo', 'illud', 'illum', 'immortalibus', 'impedit', 'imponam', 'in', 'inanem', 'inciderint', 'incididunt', 'incidimus', 'incidunt', 'incohata', 'ingeni', 'ingeniis', 'inimicorum', 'inscribunt', 'intellegimus', 'interesset', 'intuenti', 'inventis', 'inventore', 'ipsa', 'ipsam', 'ipsis', 'ipsos', 'ipsum', 'irure', 'iste', 'isti', 'itaque', 'iucunditate', 'iure', 'iusto', 'labore', 'laborent', 'laboriosam', 'laboris', 'laborum', 'laudantium', 'laude', 'levis', 'libero', 'liberos', 'libros', 'licere', 'loci', 'lorem', 'ludos', 'm', 'magistris', 'magna', 'magnam', 'magnarum', 'magni', 'magnitudine', 'magnitudinem', 'maiores', 'marcellus', 'maximarum', 'maxime', 'me', 'mea', 'mediocrium', 'medium', 'memoria', 'memoriae', 'mentibus', 'mihi', 'minim', 'minima', 'minus', 'mirari', 'mirifice', 'moderanda', 'moderatione', 'modi', 'molestiae', 'molestias', 'mollit', 'mollitia', 'motu', 'motus', 'mutuor', 'nam', 'natus', 'ne', 'necessitatibus', 'nemo', 'neque', 'nescire', 'nesciunt', 'nihil', 'nisi', 'nobis', 'nomine', 'non', 'nos', 'nosmet', 'noster', 'nostris', 'nostro', 'nostros', 'nostrud', 'nostrum', 'nulla', 'numero', 'numquam', 'nunc', 'ob', 'obcaecati', 'obiecimus', 'obsequar', 'occaecat', 'occupatione', 'odio', 'odit', 'officia', 'officiis', 'omne', 'omnes', 'omni', 'omnia', 'omnibus', 'omnino', 'omnis', 'omnium', 'oneris', 'oportet', 'optio', 'oratio', 'orationis', 'orator', 'oratore', 'oratorem', 'oratori', 'oratoribus', 'oratoris', 'oratorum', 'oris', 'ornata', 'ornate', 'oti', 'paene', 'pariatur', 'pauci', 'paucitatis', 'per', 'percepta', 'perfectiusque', 'perferendis', 'peritura', 'permagnum', 'permultos', 'persaepe', 'perspiciatis', 'pertinere', 'perturbationem', 'peste', 'philosophis', 'philosophorum', 'pietate', 'placeat', 'plura', 'plures', 'plus', 'politius', 'polliceri', 'ponendam', 'porro', 'posse', 'possimus', 'post', 'poterit', 'potest', 'potissimum', 'potius', 'praeceptis', 'praeclarissima', 'praeditos', 'praesentium', 'praesertim', 'prima', 'profecto', 'proferri', 'professioque', 'proident', 'prope', 'proposita', 'propria', 'proprium', 'provident', 'publica', 'puerilem', 'pueris', 'putem', 'putes', 'qua', 'quae', 'quaecumque', 'quaerat', 'quaerendum', 'quam', 'quamquam', 'quandam', 'quanta', 'quantum', 'quas', 'quasi', 'quem', 'qui', 'quia', 'quibus', 'quibusdam', 'quid', 'quidem', 'quis', 'quisquam', 'quo', 'quocumque', 'quod', 'quodam', 'quoniam', 'quoque', 'quorum', 'quos', 'ratione', 're', 'rebus', 'recordatio', 'recusandae', 'redundarent', 'redundet', 'rei', 'reiciendis', 'rem', 'repellat', 'repellendus', 'repetenda', 'reprehenderit', 'repudiandae', 'requiris', 'rerum', 'res', 'rhetoricos', 'roganti', 'rudia', 'saepe', 'sane', 'sapiente', 'satis', 'scaena', 'scaevola', 'scholae', 'scientia', 'scientiam', 'scribendum', 'scripsisse', 'se', 'sed', 'segregandam', 'senserint', 'sensibus', 'sententia', 'sequi', 'sermone', 'sermoni', 'si', 'sibi', 'similique', 'sine', 'singulis', 'sint', 'sit', 'sola', 'solesque', 'solum', 'soluta', 'spectare', 'statuam', 'studiis', 'sua', 'suae', 'summis', 'summos', 'sumus', 'sunt', 'suscipere', 'suscipit', 'tamen', 'tanta', 'tantisque', 'tantum', 'te', 'temperantia', 'tempor', 'tempora', 'tempore', 'temporibus', 'temporis', 'tempus', 'tenetur', 'tenui', 'theophrastum', 'thesauro', 'tibi', 'tot', 'totam', 'totum', 'tractationem', 'tribuam', 'tribuet', 'tu', 'tum', 'ullam', 'ullamco', 'unde', 'universis', 'urbis', 'usu', 'ut', 'utuntur', 'valere', 'varietate', 'vel', 'velit', 'veniam', 'verbis', 'veritatis', 'vero', 'veteris', 'vide', 'videbis', 'videtur', 'viri', 'virtutis', 'vis', 'visum', 'vitae', 'vix', 'vocis', 'voluerint', 'voluntate', 'voluptas', 'voluptate', 'voluptatem', 'voluptates', 'voluptatibus', 'voluptatum', 'vultu'
)

l1 = {}

for w in ws:
    l1.setdefault(len(w), []).append(w)

l2 = {i: l[:20] for i, l in l1.items()}
"""
