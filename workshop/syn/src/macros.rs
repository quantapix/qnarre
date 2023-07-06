#[cfg_attr(
    not(any(feature = "full", feature = "derive")),
    allow(unknown_lints, unused_macro_rules)
)]
macro_rules! ast_struct {
    (
        [$($attrs_pub:tt)*]
        struct $name:ident #full $($rest:tt)*
    ) => {
                $($attrs_pub)* struct $name $($rest)*
        #[cfg(not(feature = "full"))]
        $($attrs_pub)* struct $name {
            _noconstruct: ::std::marker::PhantomData<::proc_macro2::Span>,
        }
        #[cfg(all(not(feature = "full"), feature = "printing"))]
        impl ::quote::ToTokens for $name {
            fn to_tokens(&self, _: &mut ::proc_macro2::TokenStream) {
                unreachable!()
            }
        }
    };
    (
        [$($attrs_pub:tt)*]
        struct $name:ident $($rest:tt)*
    ) => {
        $($attrs_pub)* struct $name $($rest)*
    };
    ($($t:tt)*) => {
        strip_attrs_pub!(ast_struct!($($t)*));
    };
}
macro_rules! ast_enum {
    (
        [$($attrs_pub:tt)*]
        enum $name:ident #no_visit $($rest:tt)*
    ) => (
        ast_enum!([$($attrs_pub)*] enum $name $($rest)*);
    );
    (
        [$($attrs_pub:tt)*]
        enum $name:ident $($rest:tt)*
    ) => (
        $($attrs_pub)* enum $name $($rest)*
    );
    ($($t:tt)*) => {
        strip_attrs_pub!(ast_enum!($($t)*));
    };
}
macro_rules! ast_enum_of_structs {
    (
        $(#[$enum_attr:meta])*
        $pub:ident $enum:ident $name:ident $body:tt
        $($remaining:tt)*
    ) => {
        ast_enum!($(#[$enum_attr])* $pub $enum $name $body);
        ast_enum_of_structs_impl!($pub $enum $name $body $($remaining)*);
    };
}
macro_rules! ast_enum_of_structs_impl {
    (
        $pub:ident $enum:ident $name:ident {
            $(
                $(#[cfg $cfg_attr:tt])*
                $(#[doc $($doc_attr:tt)*])*
                $variant:ident $( ($($member:ident)::+) )*,
            )*
        }
    ) => {
        check_keyword_matches!(pub $pub);
        check_keyword_matches!(enum $enum);
        $($(
            ast_enum_from_struct!($name::$variant, $($member)::+);
        )*)*
                generate_to_tokens! {
            ()
            tokens
            $name {
                $(
                    $(#[cfg $cfg_attr])*
                    $(#[doc $($doc_attr)*])*
                    $variant $($($member)::+)*,
                )*
            }
        }
    };
}
macro_rules! ast_enum_from_struct {
    ($name:ident::Verbatim, $member:ident) => {};
    ($name:ident::$variant:ident, $member:ident) => {
        impl From<$member> for $name {
            fn from(e: $member) -> $name {
                $name::$variant(e)
            }
        }
    };
}
macro_rules! generate_to_tokens {
    (
        ($($arms:tt)*) $tokens:ident $name:ident {
            $(#[cfg $cfg_attr:tt])*
            $(#[doc $($doc_attr:tt)*])*
            $variant:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $(#[cfg $cfg_attr])* $name::$variant => {})
            $tokens $name { $($next)* }
        );
    };
    (
        ($($arms:tt)*) $tokens:ident $name:ident {
            $(#[cfg $cfg_attr:tt])*
            $(#[doc $($doc_attr:tt)*])*
            $variant:ident $member:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $(#[cfg $cfg_attr])* $name::$variant(_e) => _e.to_tokens($tokens),)
            $tokens $name { $($next)* }
        );
    };
    (($($arms:tt)*) $tokens:ident $name:ident {}) => {
        impl ::quote::ToTokens for $name {
            fn to_tokens(&self, $tokens: &mut ::proc_macro2::TokenStream) {
                match self {
                    $($arms)*
                }
            }
        }
    };
}
macro_rules! strip_attrs_pub {
    ($mac:ident!($(#[$m:meta])* $pub:ident $($t:tt)*)) => {
        check_keyword_matches!(pub $pub);
        $mac!([$(#[$m])* $pub] $($t)*);
    };
}
macro_rules! check_keyword_matches {
    (enum enum) => {};
    (pub pub) => {};
}

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

#[macro_export]
macro_rules! custom_punctuation {
    ($ident:ident, $($tt:tt)+) => {
        pub struct $ident {
            pub spans: $crate::custom_punctuation_repr!($($tt)+),
        }
                #[allow(dead_code, non_snake_case)]
        pub fn $ident<__S: $crate::__private::IntoSpans<$crate::custom_punctuation_repr!($($tt)+)>>(
            spans: __S,
        ) -> $ident {
            let _validate_len = 0 $(+ $crate::custom_punctuation_len!(strict, $tt))*;
            $ident {
                spans: $crate::__private::IntoSpans::into_spans(spans)
            }
        }
        const _: () = {
            impl $crate::__private::Default for $ident {
                fn default() -> Self {
                    $ident($crate::__private::Span::call_site())
                }
            }
            $crate::impl_parse_for_custom_punctuation!($ident, $($tt)+);
            $crate::impl_to_tokens_for_custom_punctuation!($ident, $($tt)+);
            $crate::impl_clone_for_custom_punctuation!($ident, $($tt)+);
            $crate::impl_extra_traits_for_custom_punctuation!($ident, $($tt)+);
        };
    };
}
#[macro_export]
macro_rules! impl_parse_for_custom_punctuation {
    ($ident:ident, $($tt:tt)+) => {
        impl $crate::token::CustomToken for $ident {
            fn peek(cursor: $crate::buffer::Cursor) -> bool {
                $crate::__private::peek_punct(cursor, $crate::stringify_punct!($($tt)+))
            }
            fn display() -> &'static $crate::__private::str {
                $crate::__private::concat!("`", $crate::stringify_punct!($($tt)+), "`")
            }
        }
        impl $crate::parse::Parse for $ident {
            fn parse(input: $crate::parse::ParseStream) -> $crate::parse::Result<$ident> {
                let spans: $crate::custom_punctuation_repr!($($tt)+) =
                    $crate::__private::parse_punct(input, $crate::stringify_punct!($($tt)+))?;
                Ok($ident(spans))
            }
        }
    };
}
#[macro_export]
macro_rules! impl_to_tokens_for_custom_punctuation {
    ($ident:ident, $($tt:tt)+) => {
        impl $crate::__private::ToTokens for $ident {
            fn to_tokens(&self, tokens: &mut $crate::__private::TokenStream2) {
                $crate::__private::print_punct($crate::stringify_punct!($($tt)+), &self.spans, tokens)
            }
        }
    };
}
#[macro_export]
macro_rules! impl_clone_for_custom_punctuation {
    ($ident:ident, $($tt:tt)+) => {
        impl $crate::__private::Copy for $ident {}
        #[allow(clippy::expl_impl_clone_on_copy)]
        impl $crate::__private::Clone for $ident {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
#[macro_export]
macro_rules! impl_extra_traits_for_custom_punctuation {
    ($ident:ident, $($tt:tt)+) => {
        impl $crate::__private::Debug for $ident {
            fn fmt(&self, f: &mut $crate::__private::Formatter) -> $crate::__private::fmt::Result {
                $crate::__private::Formatter::write_str(f, $crate::__private::stringify!($ident))
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
#[macro_export]
macro_rules! custom_punctuation_repr {
    ($($tt:tt)+) => {
        [$crate::__private::Span; 0 $(+ $crate::custom_punctuation_len!(lenient, $tt))+]
    };
}
#[macro_export]
#[rustfmt::skip]
macro_rules! custom_punctuation_len {
    ($mode:ident, +)     => { 1 };
    ($mode:ident, +=)    => { 2 };
    ($mode:ident, &)     => { 1 };
    ($mode:ident, &&)    => { 2 };
    ($mode:ident, &=)    => { 2 };
    ($mode:ident, @)     => { 1 };
    ($mode:ident, !)     => { 1 };
    ($mode:ident, ^)     => { 1 };
    ($mode:ident, ^=)    => { 2 };
    ($mode:ident, :)     => { 1 };
    ($mode:ident, ::)    => { 2 };
    ($mode:ident, ,)     => { 1 };
    ($mode:ident, /)     => { 1 };
    ($mode:ident, /=)    => { 2 };
    ($mode:ident, .)     => { 1 };
    ($mode:ident, ..)    => { 2 };
    ($mode:ident, ...)   => { 3 };
    ($mode:ident, ..=)   => { 3 };
    ($mode:ident, =)     => { 1 };
    ($mode:ident, ==)    => { 2 };
    ($mode:ident, >=)    => { 2 };
    ($mode:ident, >)     => { 1 };
    ($mode:ident, <=)    => { 2 };
    ($mode:ident, <)     => { 1 };
    ($mode:ident, *=)    => { 2 };
    ($mode:ident, !=)    => { 2 };
    ($mode:ident, |)     => { 1 };
    ($mode:ident, |=)    => { 2 };
    ($mode:ident, ||)    => { 2 };
    ($mode:ident, #)     => { 1 };
    ($mode:ident, ?)     => { 1 };
    ($mode:ident, ->)    => { 2 };
    ($mode:ident, <-)    => { 2 };
    ($mode:ident, %)     => { 1 };
    ($mode:ident, %=)    => { 2 };
    ($mode:ident, =>)    => { 2 };
    ($mode:ident, ;)     => { 1 };
    ($mode:ident, <<)    => { 2 };
    ($mode:ident, <<=)   => { 3 };
    ($mode:ident, >>)    => { 2 };
    ($mode:ident, >>=)   => { 3 };
    ($mode:ident, *)     => { 1 };
    ($mode:ident, -)     => { 1 };
    ($mode:ident, -=)    => { 2 };
    ($mode:ident, ~)     => { 1 };
    (lenient, $tt:tt)    => { 0 };
    (strict, $tt:tt)     => {{ $crate::custom_punctuation_unexpected!($tt); 0 }};
}
#[macro_export]
macro_rules! custom_punctuation_unexpected {
    () => {};
}
#[macro_export]
macro_rules! stringify_punct {
    ($($tt:tt)+) => {
        $crate::__private::concat!($($crate::__private::stringify!($tt)),+)
    };
}
