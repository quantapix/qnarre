// bindgen-flags: --enable-cxx-namespaces

namespace foo {
struct Bar {
  int baz;
};
} // namespace foo

namespace bar {
struct Bar_Replacement {
  int bazz;
};
}; // namespace bar

typedef foo::Bar ReferencesBar;
