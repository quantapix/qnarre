macro_rules! test {
    () => {
        cfg!(test) && ::std::env::var("_CLANG_TEST").is_ok()
    };
}

macro_rules! target_os {
    ($os:expr) => {
        if cfg!(test) && ::std::env::var("_CLANG_TEST").is_ok() {
            let y = ::std::env::var("_CLANG_TEST_OS");
            y.map_or(false, |x| x == $os)
        } else {
            cfg!(target_os = $os)
        }
    };
}

macro_rules! target_pointer_width {
    ($pointer_width:expr) => {
        if cfg!(test) && ::std::env::var("_CLANG_TEST").is_ok() {
            let y = ::std::env::var("_CLANG_TEST_POINTER_WIDTH");
            y.map_or(false, |x| x == $pointer_width)
        } else {
            cfg!(target_pointer_width = $pointer_width)
        }
    };
}

macro_rules! target_env {
    ($env:expr) => {
        if cfg!(test) && ::std::env::var("_CLANG_TEST").is_ok() {
            let y = ::std::env::var("_CLANG_TEST_ENV");
            y.map_or(false, |x| x == $env)
        } else {
            cfg!(target_env = $env)
        }
    };
}
