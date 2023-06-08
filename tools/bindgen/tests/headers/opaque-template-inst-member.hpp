// bindgen-flags: --opaque-type 'OpaqueTemplate' --with-derive-hash
// --with-derive-partialeq --impl-partialeq --with-derive-eq --rust-target 1.40

template <typename T> class OpaqueTemplate {
  T mData;
  bool mCannotDebug[400];
};

class ContainsOpaqueTemplate {
  OpaqueTemplate<int> mBlah;
  int mBaz;
};

class InheritsOpaqueTemplate : public OpaqueTemplate<bool> {
  char *wow;
};
