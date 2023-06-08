// bindgen-flags: -- -std=c++14

// Small test that we handle virtual tables correctly when deriving from a
// template instantiation. This wasn't previously handled at all. Note that when
// inheriting from a template parameter, the type that is instantiated might or
// might not have a virtual table, and we have no way of knowing. We don't
// handle that yet, so no test for it here.

template <class T> class BaseWithVtable {
  T t;

  virtual void hello();
};

class DerivedWithNoVirtualMethods : public BaseWithVtable<char *> {};

class DerivedWithVirtualMethods : public BaseWithVtable<char *> {
  virtual void zoidberg();
};

template <class U> class BaseWithoutVtable {
  U u;
};

class DerivedWithVtable : public BaseWithoutVtable<char *> {
  virtual void leela();
};

class DerivedWithoutVtable : public BaseWithoutVtable<char *> {};
