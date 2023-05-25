#[macro_export]
macro_rules! extra_assert {
    ( $cond:expr ) => {
        if cfg!(feature = "__testing_only_extra_assertions") {
            assert!($cond);
        }
    };
    ( $cond:expr , $( $arg:tt )+ ) => {
        if cfg!(feature = "__testing_only_extra_assertions") {
            assert!($cond, $( $arg )* )
        }
    };
}

#[macro_export]
macro_rules! extra_assert_eq {
    ( $lhs:expr , $rhs:expr ) => {
        if cfg!(feature = "__testing_only_extra_assertions") {
            assert_eq!($lhs, $rhs);
        }
    };
    ( $lhs:expr , $rhs:expr , $( $arg:tt )+ ) => {
        if cfg!(feature = "__testing_only_extra_assertions") {
            assert!($lhs, $rhs, $( $arg )* );
        }
    };
}
