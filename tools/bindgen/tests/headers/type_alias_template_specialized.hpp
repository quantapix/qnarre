// bindgen-flags: --allowlist-type Rooted -- -std=c++14

template <typename a> using MaybeWrapped = a;
class Rooted {
  MaybeWrapped<int> ptr;
};

template <typename a> using replaces_MaybeWrapped = a;
