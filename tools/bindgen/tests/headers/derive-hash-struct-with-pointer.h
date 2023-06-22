// bindgen-flags: --with-derive-hash --with-derive-partialord --with-derive-ord
// --with-derive-partialeq --with-derive-eq
//
struct ConstPtrMutObj {
  int *const bar;
};

struct MutPtrMutObj {
  int *bar;
};

struct MutPtrConstObj {
  const int *bar;
};

struct ConstPtrConstObj {
  const int *const bar;
};
