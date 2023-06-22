// bindgen-flags: --enable-cxx-namespaces --opaque-type
// 'zoidberg::Template<zoidberg::Bar>'  --with-derive-hash
// --with-derive-partialeq --with-derive-eq -- -std=c++14

namespace zoidberg {

template <typename T> class Template {
  T member;
};

struct Foo {
  char c;
};

struct Bar {
  int i;
};

class ContainsInstantiation {
  Template<Foo> not_opaque;
};

class ContainsOpaqueInstantiation {
  Template<Bar> opaque;
};

} // namespace zoidberg
