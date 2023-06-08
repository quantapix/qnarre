// bindgen-flags: --with-derive-partialeq --impl-partialeq

union ShouldNotDerivePartialEq {
  char a;
  int b;
};
