// bindgen-flags: --rust-target 1.0 --with-derive-partialeq --impl-partialeq

union ShouldDerivePartialEq {
  char a[150];
  int b;
};
