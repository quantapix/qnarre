pub use proc_macro2::TokenStream as TokenStream2;
pub use syn::{parse_str, DeriveInput};

#[cfg(all(
    not(all(target_arch = "wasm32", any(target_os = "unknown", target_os = "wasi"))),
    feature = "proc-macro"
))]
pub use proc_macro::TokenStream;
#[cfg(all(
    not(all(target_arch = "wasm32", any(target_os = "unknown", target_os = "wasi"))),
    feature = "proc-macro"
))]
pub use syn::parse;

#[cfg(all(
    not(all(target_arch = "wasm32", any(target_os = "unknown", target_os = "wasi"))),
    feature = "proc-macro"
))]
#[macro_export]
macro_rules! decl_derive {
    ([$derives:ident $($derive_t:tt)*] => $(#[$($attrs:tt)*])* $inner:path) => {
        #[proc_macro_derive($derives $($derive_t)*)]
        #[allow(non_snake_case)]
        $(#[$($attrs)*])*
        pub fn $derives(
            i: $crate::macros::TokenStream
        ) -> $crate::macros::TokenStream {
            match $crate::macros::parse::<$crate::macros::DeriveInput>(i) {
                ::core::result::Result::Ok(p) => {
                    match $crate::Structure::try_new(&p) {
                        ::core::result::Result::Ok(s) => $crate::MacroResult::into_stream($inner(s)),
                        ::core::result::Result::Err(e) => {
                            ::core::convert::Into::into(e.to_compile_error())
                        }
                    }
                }
                ::core::result::Result::Err(e) => {
                    ::core::convert::Into::into(e.to_compile_error())
                }
            }
        }
    };
}

#[cfg(all(
    not(all(target_arch = "wasm32", any(target_os = "unknown", target_os = "wasi"))),
    feature = "proc-macro"
))]
#[macro_export]
macro_rules! decl_attribute {
    ([$attribute:ident] => $(#[$($attrs:tt)*])* $inner:path) => {
        #[proc_macro_attribute]
        $(#[$($attrs)*])*
        pub fn $attribute(
            attr: $crate::macros::TokenStream,
            i: $crate::macros::TokenStream,
        ) -> $crate::macros::TokenStream {
            match $crate::macros::parse::<$crate::macros::DeriveInput>(i) {
                ::core::result::Result::Ok(p) => match $crate::Structure::try_new(&p) {
                    ::core::result::Result::Ok(s) => {
                        $crate::MacroResult::into_stream(
                            $inner(::core::convert::Into::into(attr), s)
                        )
                    }
                    ::core::result::Result::Err(e) => {
                        ::core::convert::Into::into(e.to_compile_error())
                    }
                },
                ::core::result::Result::Err(e) => {
                    ::core::convert::Into::into(e.to_compile_error())
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_derive {
    ($name:path { $($i:tt)* } expands to { $($o:tt)* }) => {
        {
            #[allow(dead_code)]
            fn ensure_compiles() {
                $($i)*
                $($o)*
            }

            $crate::test_derive!($name { $($i)* } expands to { $($o)* } no_build);
        }
    };

    ($name:path { $($i:tt)* } expands to { $($o:tt)* } no_build) => {
        {
            let i = ::core::stringify!( $($i)* );
            let parsed = $crate::macros::parse_str::<$crate::macros::DeriveInput>(i)
                .expect(::core::concat!(
                    "Failed to parse input to `#[derive(",
                    ::core::stringify!($name),
                    ")]`",
                ));

            let raw_res = $name($crate::Structure::new(&parsed));
            let res = $crate::MacroResult::into_result(raw_res)
                .expect(::core::concat!(
                    "Procedural macro failed for `#[derive(",
                    ::core::stringify!($name),
                    ")]`",
                ));

            let expected = ::core::stringify!( $($o)* )
                .parse::<$crate::macros::TokenStream2>()
                .expect("output should be a valid TokenStream");
            let mut expected_toks = <$crate::macros::TokenStream2
                as ::core::convert::From<$crate::macros::TokenStream2>>::from(expected);
            if <$crate::macros::TokenStream2 as ::std::string::ToString>::to_string(&res)
                != <$crate::macros::TokenStream2 as ::std::string::ToString>::to_string(&expected_toks)
            {
                panic!("\
test_derive failed:
expected:
```
{}
```

got:
```
{}
```\n",
                    $crate::unpretty_print(&expected_toks),
                    $crate::unpretty_print(&res),
                );
            }
        }
    };
}
