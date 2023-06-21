// bindgen-flags: --with-derive-partialeq --impl-partialeq --impl-debug
// --rust-target 1.40

struct Foo {
  int large[33];
  char type_ : 3;
  unsigned : 8;
  char type();
  void set_type_(char c);
  void set_type(char c);
};
