#[macro_export]
macro_rules! format_to {
    ($x:expr) => ();
    ($x:expr, $y:literal $($z:tt)*) => {
        { use ::std::fmt::Write as _; let _ = ::std::write!($x, $y $($z)*); }
    };
}

#[macro_export]
macro_rules! impl_from {
    ($($variant:ident $(($($sub_variant:ident),*))?),* for $enum:ident) => {
        $(
            impl From<$variant> for $enum {
                fn from(x: $variant) -> $enum {
                    $enum::$variant(x)
                }
            }
            $($(
                impl From<$sub_variant> for $enum {
                    fn from(x: $sub_variant) -> $enum {
                        $enum::$variant($variant::$sub_variant(x))
                    }
                }
            )*)?
        )*
    };
    ($($variant:ident$(<$V:ident>)?),* for $enum:ident) => {
        $(
            impl$(<$V>)? From<$variant$(<$V>)?> for $enum$(<$V>)? {
                fn from(x: $variant$(<$V>)?) -> $enum$(<$V>)? {
                    $enum::$variant(x)
                }
            }
        )*
    }
}

#[macro_export]
macro_rules! assert_eq_text {
    ($left:expr, $right:expr) => {
        assert_eq_text!($left, $right,)
    };
    ($left:expr, $right:expr, $($tt:tt)*) => {{
        let left = $left;
        let right = $right;
        if left != right {
            if left.trim() == right.trim() {
                eprintln!("Left:\n{:?}\n\nRight:\n{:?}\n\nWhitespace difference\n", left, right);
            }
            eprintln!($($tt)*);
            panic!("text differs");
        }
    }};
}
