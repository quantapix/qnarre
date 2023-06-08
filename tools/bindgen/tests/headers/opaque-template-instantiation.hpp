// bindgen-flags: --opaque-type 'Template<int>'  --with-derive-hash
// --with-derive-partialeq --with-derive-eq -- -std=c++14

template <typename T> class Template {
  T member;
};

class ContainsInstantiation {
  Template<char> not_opaque;
};

class ContainsOpaqueInstantiation {
  Template<int> opaque;
};
