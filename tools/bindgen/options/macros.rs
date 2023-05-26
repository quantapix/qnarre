macro_rules! regex_opt {
    ($(#[$attrs:meta])* pub fn $($tokens:tt)*) => {
        $(#[$attrs])*
        pub fn $($tokens)*
    };
}

macro_rules! default {
    () => {
        Default::default()
    };
    ($expr:expr) => {
        $expr
    };
}

macro_rules! as_args {
    ($flag:literal) => {
        |field, args| AsArgs::as_args(field, args, $flag)
    };
    ($expr:expr) => {
        $expr
    };
}

pub(super) fn ignore<T>(_: &T, _: &mut Vec<String>) {}
