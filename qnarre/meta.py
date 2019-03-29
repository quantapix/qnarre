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


class converter:

    by_name = {}

    @classmethod
    def convert(cls, src, **kw):
        return cls.by_name[type(src).__name__].convert(src, **kw)

    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        self.by_name[self.name] = cls
        return cls


class with_current:
    def __init__(self):
        pass

    def __call__(self, cls):
        setattr(cls, 'current', cls())
        return cls


class with_class_init:
    def __init__(self):
        pass

    def __call__(self, cls):
        cls.init()
        return cls


class with_property:

    default = None

    def __init__(self, name, creator, default=None):
        self.name = name
        self.multi = name.endswith('s')
        self.creator = creator
        if default is not None:
            self.default = default
        elif self.multi:
            self.default = ()

    def __call__(self, cls):
        n = '_' + self.name
        setattr(cls, n, self.default)

        def getter(self):
            return getattr(self, n)

        c = self.creator

        if self.multi:

            def setter(self, vs):
                if vs:
                    setattr(self, n, tuple(c(vs)))
                else:
                    self.__dict__.pop(n, None)
        else:

            def setter(self, v):
                if v:
                    setattr(self, n, c(v))
                else:
                    self.__dict__.pop(n, None)

        setattr(cls, self.name, property(getter, setter))
        return cls


if __name__ == '__main__':

    @with_current()
    class A:
        def __init__(self):
            self.a = 'a'

    assert A.current.a == 'a'
    print('0 passed')

    class Name:
        @classmethod
        def create(cls, v):
            return v

    @with_property('name', Name.create, default='')
    class A:
        pass

    a = A()
    assert a.name is ''
    assert '_name' not in vars(a)
    a.name = 'b'
    assert '_name' in vars(a)
    assert a.name == 'b'
    a.name = ''
    assert a.name is ''
    assert '_name' not in vars(a)
    print('A passed')

    class Link:
        @classmethod
        def create(cls, v):
            return v

    @with_property('link', Link.create)
    class B:
        pass

    a = B()
    assert a.link is None
    assert '_link' not in vars(a)
    a.link = 'b'
    assert '_link' in vars(a)
    assert a.link == 'b'
    a.link = None
    assert a.link is None
    assert '_link' not in vars(a)
    print('B passed')

    class Value:
        @classmethod
        def creator(cls, vs):
            for v in vs:
                yield v

    @with_property('values', Value.creator)
    class C:
        pass

    a = C()
    assert a.values is ()
    assert '_values' not in vars(a)
    a.values = ('b', 'c')
    assert '_values' in vars(a)
    assert a.values == ('b', 'c')
    a.values = ()
    assert a.values is ()
    assert '_values' not in vars(a)
    print('C passed')

    class Other:
        def meth(self, v):
            return v

    Other.obj = Other()

    @with_property('extra', creator=Other.obj.meth)
    class D:
        pass

    a = D()
    assert a.extra is None
    assert '_extra' not in vars(a)
    a.extra = 'b'
    assert '_extra' in vars(a)
    assert a.extra == 'b'
    a.extra = None
    assert a.extra is None
    assert '_extra' not in vars(a)
    print('D passed')
