#[macro_export]
macro_rules! custom_keyword {
    ($ident:ident) => {
        #[allow(non_camel_case_types)]
        pub struct $ident {
            pub span: $crate::__private::Span,
        }
        #[allow(dead_code, non_snake_case)]
        pub fn $ident<__S: $crate::__private::IntoSpans<$crate::__private::Span>>(span: __S) -> $ident {
            $ident {
                span: $crate::__private::IntoSpans::into_spans(span),
            }
        }
        const _: () = {
            impl $crate::__private::Default for $ident {
                fn default() -> Self {
                    $ident {
                        span: $crate::__private::Span::call_site(),
                    }
                }
            }
            $crate::impl_parse_for_custom_keyword!($ident);
            $crate::impl_to_tokens_for_custom_keyword!($ident);
            $crate::impl_clone_for_custom_keyword!($ident);
            $crate::impl_extra_traits_for_custom_keyword!($ident);
        };
    };
}
#[macro_export]
macro_rules! impl_parse_for_custom_keyword {
    ($ident:ident) => {
        impl $crate::token::CustomToken for $ident {
            fn peek(cursor: $crate::buffer::Cursor) -> $crate::__private::bool {
                if let $crate::__private::Some((ident, _rest)) = cursor.ident() {
                    ident == $crate::__private::stringify!($ident)
                } else {
                    false
                }
            }
            fn display() -> &'static $crate::__private::str {
                $crate::__private::concat!("`", $crate::__private::stringify!($ident), "`")
            }
        }
        impl $crate::parse::Parse for $ident {
            fn parse(input: $crate::parse::ParseStream) -> $crate::parse::Result<$ident> {
                input.step(|cursor| {
                    if let $crate::__private::Some((ident, rest)) = cursor.ident() {
                        if ident == $crate::__private::stringify!($ident) {
                            return $crate::__private::Ok(($ident { span: ident.span() }, rest));
                        }
                    }
                    $crate::__private::Err(cursor.error($crate::__private::concat!(
                        "expected `",
                        $crate::__private::stringify!($ident),
                        "`",
                    )))
                })
            }
        }
    };
}
#[cfg(not(feature = "parsing"))]
#[macro_export]
macro_rules! impl_parse_for_custom_keyword {
    ($ident:ident) => {};
}
#[macro_export]
macro_rules! impl_to_tokens_for_custom_keyword {
    ($ident:ident) => {
        impl $crate::__private::ToTokens for $ident {
            fn to_tokens(&self, tokens: &mut $crate::__private::TokenStream2) {
                let ident = $crate::Ident::new($crate::__private::stringify!($ident), self.span);
                $crate::__private::TokenStreamExt::append(tokens, ident);
            }
        }
    };
}
#[cfg(not(feature = "printing"))]
#[macro_export]
macro_rules! impl_to_tokens_for_custom_keyword {
    ($ident:ident) => {};
}
#[macro_export]
macro_rules! impl_clone_for_custom_keyword {
    ($ident:ident) => {
        impl $crate::__private::Copy for $ident {}
        #[allow(clippy::expl_impl_clone_on_copy)]
        impl $crate::__private::Clone for $ident {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
#[cfg(not(feature = "clone-impls"))]
#[macro_export]
macro_rules! impl_clone_for_custom_keyword {
    ($ident:ident) => {};
}
#[macro_export]
macro_rules! impl_extra_traits_for_custom_keyword {
    ($ident:ident) => {
        impl $crate::__private::Debug for $ident {
            fn fmt(&self, f: &mut $crate::__private::Formatter) -> $crate::__private::fmt::Result {
                $crate::__private::Formatter::write_str(
                    f,
                    $crate::__private::concat!("Keyword [", $crate::__private::stringify!($ident), "]",),
                )
            }
        }
        impl $crate::__private::Eq for $ident {}
        impl $crate::__private::PartialEq for $ident {
            fn eq(&self, _other: &Self) -> $crate::__private::bool {
                true
            }
        }
        impl $crate::__private::Hash for $ident {
            fn hash<__H: $crate::__private::Hasher>(&self, _state: &mut __H) {}
        }
    };
}
#[cfg(not(feature = "extra-traits"))]
#[macro_export]
macro_rules! impl_extra_traits_for_custom_keyword {
    ($ident:ident) => {};
}
