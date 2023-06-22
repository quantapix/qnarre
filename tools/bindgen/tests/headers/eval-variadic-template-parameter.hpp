// bindgen-flags: -- -std=c++11

template <typename... T> struct B {
  static const long c = sizeof...(T);
};
