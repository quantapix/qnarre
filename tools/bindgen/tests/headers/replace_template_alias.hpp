// bindgen-flags: -- --std=c++14

namespace JS {
namespace detail {

template <typename T> using MaybeWrapped = int;

}

template <typename T> class Rooted {
  detail::MaybeWrapped<T> ptr;
};

} // namespace JS

///
template <typename T> using replaces_MaybeWrapped = T;
