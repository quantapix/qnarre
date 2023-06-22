// bindgen-flags: --with-derive-hash --with-derive-partialord --with-derive-ord
// --with-derive-partialeq --with-derive-eq
//
template <typename T> struct foo {
  T data;
};

struct IntStr {
  foo<int> a;
};

struct FloatStr {
  foo<float> a;
};
