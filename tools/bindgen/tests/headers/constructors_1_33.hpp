// bindgen-flags: --rust-target 1.33

class TestOverload {
  TestOverload();

public:
  TestOverload(int);
  TestOverload(double);
};

class TestPublicNoArgs {
public:
  TestPublicNoArgs();
};
