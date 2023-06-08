// bindgen-flags: --opaque-type 'OpaqueTemplate' --with-derive-hash
// --with-derive-partialeq --with-derive-eq

template <typename T> class OpaqueTemplate {
  T mData;
};

class ContainsOpaqueTemplate {
  OpaqueTemplate<int> mBlah;
  int mBaz;
};

class InheritsOpaqueTemplate : public OpaqueTemplate<bool> {
  char *wow;
};
