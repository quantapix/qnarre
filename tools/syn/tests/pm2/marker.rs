#![allow(clippy::extra_unused_type_parameters)]

use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

macro_rules! assert_impl {
    ($ty:ident is $($marker:ident) and +) => {
        #[test]
        #[allow(non_snake_case)]
        fn $ty() {
            fn assert_implemented<T: $($marker +)+>() {}
            assert_implemented::<$ty>();
        }
    };

    ($ty:ident is not $($marker:ident) or +) => {
        #[test]
        #[allow(non_snake_case)]
        fn $ty() {
            $(
                {
                    trait IsNotImplemented {
                        fn assert_not_implemented() {}
                    }
                    impl<T: $marker> IsNotImplemented for T {}
                    trait IsImplemented {
                        fn assert_not_implemented() {}
                    }
                    impl IsImplemented for $ty {}
                    <$ty>::assert_not_implemented();
                }
            )+
        }
    };
}

assert_impl!(Delimiter is Send and Sync);
assert_impl!(Spacing is Send and Sync);

assert_impl!(Group is not Send or Sync);
assert_impl!(Ident is not Send or Sync);
assert_impl!(LexError is not Send or Sync);
assert_impl!(Literal is not Send or Sync);
assert_impl!(Punct is not Send or Sync);
assert_impl!(Span is not Send or Sync);
assert_impl!(TokenStream is not Send or Sync);
assert_impl!(TokenTree is not Send or Sync);

mod unwind_safe {
    use proc_macro2::{Delimiter, Group, Ident, LexError, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
    use std::panic::{RefUnwindSafe, UnwindSafe};

    macro_rules! assert_unwind_safe {
        ($($types:ident)*) => {
            $(
                assert_impl!($types is UnwindSafe and RefUnwindSafe);
            )*
        };
    }

    assert_unwind_safe! {
        Delimiter
        Group
        Ident
        LexError
        Literal
        Punct
        Spacing
        Span
        TokenStream
        TokenTree
    }
}
