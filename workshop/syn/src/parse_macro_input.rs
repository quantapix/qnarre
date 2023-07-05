#[macro_export]
macro_rules! parse_macro_input {
    ($tokenstream:ident as $ty:ty) => {
        match $crate::parse::<$ty>($tokenstream) {
            $crate::__private::Ok(data) => data,
            $crate::__private::Err(err) => {
                return $crate::__private::TokenStream::from(err.to_compile_error());
            },
        }
    };
    ($tokenstream:ident with $parser:path) => {
        match $crate::parse::Parser::parse($parser, $tokenstream) {
            $crate::__private::Ok(data) => data,
            $crate::__private::Err(err) => {
                return $crate::__private::TokenStream::from(err.to_compile_error());
            },
        }
    };
    ($tokenstream:ident) => {
        $crate::parse_macro_input!($tokenstream as _)
    };
}
