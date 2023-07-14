use super::*;

pub struct Mac {
    pub path: Path,
    pub bang: Token![!],
    pub delim: tok::Delim,
    pub toks: pm2::Stream,
}
impl Mac {
    pub fn parse_body<T: Parse>(&self) -> Res<T> {
        self.parse_body_with(T::parse)
    }
    pub fn parse_body_with<T: Parser>(&self, f: T) -> Res<T::Output> {
        let scope = self.delim.span().close();
        parse::parse_scoped(f, scope, self.toks.clone())
    }
}
impl Parse for Mac {
    fn parse(x: Stream) -> Res<Self> {
        let toks;
        Ok(Mac {
            path: x.call(Path::parse_mod_style)?,
            bang: x.parse()?,
            delim: {
                let (y, x) = parse_delim(x)?;
                toks = x;
                y
            },
            toks,
        })
    }
}
impl ToTokens for Mac {
    fn to_tokens(&self, ys: &mut Stream) {
        self.path.to_tokens(ys);
        self.bang.to_tokens(ys);
        self.delim.surround(ys, self.toks.clone());
    }
}

pub fn parse_delim(s: Stream) -> Res<(tok::Delim, pm2::Stream)> {
    s.step(|c| {
        if let Some((pm2::Tree::Group(x), rest)) = c.token_tree() {
            let s = x.delim_span();
            let delim = match x.delimiter() {
                pm2::Delim::Parenthesis => tok::Delim::Paren(Paren(s)),
                pm2::Delim::Brace => tok::Delim::Brace(Brace(s)),
                pm2::Delim::Bracket => tok::Delim::Bracket(Bracket(s)),
                pm2::Delim::None => {
                    return Err(c.err("expected delimiter"));
                },
            };
            Ok(((delim, x.stream()), rest))
        } else {
            Err(c.err("expected delimiter"))
        }
    })
}

macro_rules! ast_enum_of_structs {
    (
        pub enum $name:ident $body:tt
        $($rest:tt)*
    ) => {
        pub enum $name $body
        ast_enum_of_structs_impl!($name $body $($rest)*);
    };
}
macro_rules! ast_enum_of_structs_impl {
    (
        $name:ident {
            $(
                $variant:ident $( ($($member:ident)::+) )*,
            )*
        }
    ) => {
        $($(
            ast_enum_from_struct!($name::$variant, $($member)::+);
        )*)*
        generate_to_tokens! {
            ()
            tokens
            $name {
                $(
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
        ($($arms:tt)*) $ys:ident $name:ident {
            $variant:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $name::$variant => {})
            $ys $name { $($next)* }
        );
    };
    (
        ($($arms:tt)*) $ys:ident $name:ident {
            $variant:ident $member:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $name::$variant(_e) => _e.to_tokens($ys),)
            $ys $name { $($next)* }
        );
    };
    (($($arms:tt)*) $ys:ident $name:ident {}) => {
        impl ::quote::ToTokens for $name {
            fn to_tokens(&self, $ys: &mut crate::pm2::Stream) {
                match self {
                    $($arms)*
                }
            }
        }
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
        pub fn $ident<__S: $crate::IntoSpans<$crate::__private::Span>>(span: __S) -> $ident {
            $ident {
                span: $crate::IntoSpans::into_spans(span),
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
        impl $crate::tok::Custom for $ident {
            fn peek(cursor: $crate::Cursor) -> $crate::__private::bool {
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
            fn parse(input: $crate::parse::Stream) -> $crate::Res<$ident> {
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
            fn to_tokens(&self, ys: &mut $crate::__private::TokenStream2) {
                let ident = $crate::Ident::new($crate::__private::stringify!($ident), self.span);
                $crate::__private::TokenStreamExt::append(ys, ident);
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
        pub fn $ident<__S: $crate::IntoSpans<$crate::custom_punctuation_repr!($($tt)+)>>(
            spans: __S,
        ) -> $ident {
            let _validate_len = 0 $(+ $crate::custom_punctuation_len!(strict, $tt))*;
            $ident {
                spans: $crate::IntoSpans::into_spans(spans)
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
        impl $crate::tok::Custom for $ident {
            fn peek(cursor: $crate::Cursor) -> bool {
                $crate::__private::peek_punct(cursor, $crate::stringify_punct!($($tt)+))
            }
            fn display() -> &'static $crate::__private::str {
                $crate::__private::concat!("`", $crate::stringify_punct!($($tt)+), "`")
            }
        }
        impl $crate::parse::Parse for $ident {
            fn parse(input: $crate::parse::Stream) -> $crate::Res<$ident> {
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
            fn to_tokens(&self, ys: &mut $crate::__private::TokenStream2) {
                $crate::__private::lower_punct($crate::stringify_punct!($($tt)+), &self.spans, ys)
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

#[macro_export]
macro_rules! parse_quote {
    ($($tt:tt)*) => {
        $crate::__private::parse_quote_fn($crate::__private::quote::quote!($($tt)*))
    };
}
#[macro_export]
macro_rules! parse_quote_spanned {
    ($span:expr=> $($tt:tt)*) => {
        $crate::__private::parse_quote_fn($crate::__private::quote::quote_spanned!($span=> $($tt)*))
    };
}

#[macro_export]
macro_rules! parse_macro_input {
    ($ys:ident as $ty:ty) => {
        match $crate::parse::<$ty>($ys) {
            $crate::__private::Ok(data) => data,
            $crate::__private::Err(err) => {
                return $crate::__private::pm2::Stream::from(err.to_compile_error());
            },
        }
    };
    ($ys:ident with $parser:path) => {
        match $crate::parse::Parser::parse($parser, $ys) {
            $crate::__private::Ok(data) => data,
            $crate::__private::Err(err) => {
                return $crate::__private::pm2::Stream::from(err.to_compile_error());
            },
        }
    };
    ($ys:ident) => {
        $crate::parse_macro_input!($ys as _)
    };
}

#[macro_export]
macro_rules! Token {
    [abstract]    => { $crate::tok::Abstract };
    [as]          => { $crate::tok::As };
    [async]       => { $crate::tok::Async };
    [auto]        => { $crate::tok::Auto };
    [await]       => { $crate::tok::Await };
    [become]      => { $crate::tok::Become };
    [box]         => { $crate::tok::Box };
    [break]       => { $crate::tok::Break };
    [const]       => { $crate::tok::Const };
    [continue]    => { $crate::tok::Continue };
    [crate]       => { $crate::tok::Crate };
    [default]     => { $crate::tok::Default };
    [do]          => { $crate::tok::Do };
    [dyn]         => { $crate::tok::Dyn };
    [else]        => { $crate::tok::Else };
    [enum]        => { $crate::tok::Enum };
    [extern]      => { $crate::tok::Extern };
    [final]       => { $crate::tok::Final };
    [fn]          => { $crate::tok::Fn };
    [for]         => { $crate::tok::For };
    [if]          => { $crate::tok::If };
    [impl]        => { $crate::tok::Impl };
    [in]          => { $crate::tok::In };
    [let]         => { $crate::tok::Let };
    [loop]        => { $crate::tok::Loop };
    [macro]       => { $crate::tok::Macro };
    [match]       => { $crate::tok::Match };
    [mod]         => { $crate::tok::Mod };
    [move]        => { $crate::tok::Move };
    [mut]         => { $crate::tok::Mut };
    [override]    => { $crate::tok::Override };
    [priv]        => { $crate::tok::Priv };
    [pub]         => { $crate::tok::Pub };
    [ref]         => { $crate::tok::Ref };
    [return]      => { $crate::tok::Return };
    [Self]        => { $crate::tok::SelfType };
    [self]        => { $crate::tok::SelfValue };
    [static]      => { $crate::tok::Static };
    [struct]      => { $crate::tok::Struct };
    [super]       => { $crate::tok::Super };
    [trait]       => { $crate::tok::Trait };
    [try]         => { $crate::tok::Try };
    [type]        => { $crate::tok::Type };
    [typeof]      => { $crate::tok::Typeof };
    [union]       => { $crate::tok::Union };
    [unsafe]      => { $crate::tok::Unsafe };
    [unsized]     => { $crate::tok::Unsized };
    [use]         => { $crate::tok::Use };
    [virtual]     => { $crate::tok::Virtual };
    [where]       => { $crate::tok::Where };
    [while]       => { $crate::tok::While };
    [yield]       => { $crate::tok::Yield };
    [&]           => { $crate::tok::And };
    [&&]          => { $crate::tok::AndAnd };
    [&=]          => { $crate::tok::AndEq };
    [@]           => { $crate::tok::At };
    [^]           => { $crate::tok::Caret };
    [^=]          => { $crate::tok::CaretEq };
    [:]           => { $crate::tok::Colon };
    [,]           => { $crate::tok::Comma };
    [$]           => { $crate::tok::Dollar };
    [.]           => { $crate::tok::Dot };
    [..]          => { $crate::tok::DotDot };
    [...]         => { $crate::tok::DotDotDot };
    [..=]         => { $crate::tok::DotDotEq };
    [=]           => { $crate::tok::Eq };
    [==]          => { $crate::tok::EqEq };
    [=>]          => { $crate::tok::FatArrow };
    [>=]          => { $crate::tok::Ge };
    [>]           => { $crate::tok::Gt };
    [<-]          => { $crate::tok::LArrow };
    [<=]          => { $crate::tok::Le };
    [<]           => { $crate::tok::Lt };
    [-]           => { $crate::tok::Minus };
    [-=]          => { $crate::tok::MinusEq };
    [!=]          => { $crate::tok::Ne };
    [!]           => { $crate::tok::Not };
    [|]           => { $crate::tok::Or };
    [|=]          => { $crate::tok::OrEq };
    [||]          => { $crate::tok::OrOr };
    [::]          => { $crate::tok::PathSep };
    [%]           => { $crate::tok::Percent };
    [%=]          => { $crate::tok::PercentEq };
    [+]           => { $crate::tok::Plus };
    [+=]          => { $crate::tok::PlusEq };
    [#]           => { $crate::tok::Pound };
    [?]           => { $crate::tok::Question };
    [->]          => { $crate::tok::RArrow };
    [;]           => { $crate::tok::Semi };
    [<<]          => { $crate::tok::Shl };
    [<<=]         => { $crate::tok::ShlEq };
    [>>]          => { $crate::tok::Shr };
    [>>=]         => { $crate::tok::ShrEq };
    [/]           => { $crate::tok::Slash };
    [/=]          => { $crate::tok::SlashEq };
    [*]           => { $crate::tok::Star };
    [*=]          => { $crate::tok::StarEq };
    [~]           => { $crate::tok::Tilde };
    [_]           => { $crate::tok::Underscore };
}

#[macro_export]
macro_rules! parenthesized {
    ($content:ident in $cur:expr) => {
        match $crate::parse::parse_parens(&$cur) {
            $crate::__private::Ok(x) => {
                $content = x.content;
                x.token
            },
            $crate::__private::Err(x) => {
                return $crate::__private::Err(x);
            },
        }
    };
}

#[macro_export]
macro_rules! braced {
    ($content:ident in $cur:expr) => {
        match $crate::parse::parse_braces(&$cur) {
            $crate::__private::Ok(x) => {
                $content = x.content;
                x.token
            },
            $crate::__private::Err(x) => {
                return $crate::__private::Err(x);
            },
        }
    };
}

#[macro_export]
macro_rules! bracketed {
    ($content:ident in $cur:expr) => {
        match $crate::parse::parse_brackets(&$cur) {
            $crate::__private::Ok(x) => {
                $content = x.content;
                x.token
            },
            $crate::__private::Err(x) => {
                return $crate::__private::Err(x);
            },
        }
    };
}
