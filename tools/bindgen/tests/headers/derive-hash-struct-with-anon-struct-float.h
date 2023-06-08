// bindgen-flags: --with-derive-hash --with-derive-partialord --with-derive-ord
// --with-derive-partialeq --with-derive-eq
//
struct foo {
  struct {
    float a;
    float b;
  } bar;
};
