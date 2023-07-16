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
                pm2::Delim::Parenthesis => tok::Delim::Paren(tok::Paren(s)),
                pm2::Delim::Brace => tok::Delim::Brace(tok::Brace(s)),
                pm2::Delim::Bracket => tok::Delim::Bracket(tok::Bracket(s)),
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
        pub enum $n:ident $tt:tt
        $($rest:tt)*
    ) => {
        pub enum $n $tt
        ast_enum_of_structs_impl!($n $tt $($rest)*);
    };
}
macro_rules! ast_enum_of_structs_impl {
    (
        $n:ident {
            $(
                $variant:ident $( ($($m:ident)::+) )*,
            )*
        }
    ) => {
        $($(
            ast_enum_from_struct!($n::$variant, $($m)::+);
        )*)*
        generate_to_tokens! {
            ()
            tokens
            $n {
                $(
                    $variant $($($m)::+)*,
                )*
            }
        }
    };
}
macro_rules! ast_enum_from_struct {
    ($n:ident::Stream, $m:ident) => {};
    ($n:ident::$variant:ident, $m:ident) => {
        impl From<$m> for $n {
            fn from(e: $m) -> $n {
                $n::$variant(e)
            }
        }
    };
}
macro_rules! generate_to_tokens {
    (
        ($($arms:tt)*) $ys:ident $n:ident {
            $variant:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $n::$variant => {})
            $ys $n { $($next)* }
        );
    };
    (
        ($($arms:tt)*) $ys:ident $n:ident {
            $variant:ident $m:ident,
            $($next:tt)*
        }
    ) => {
        generate_to_tokens!(
            ($($arms)* $n::$variant(_e) => _e.to_tokens($ys),)
            $ys $n { $($next)* }
        );
    };
    (($($arms:tt)*) $ys:ident $n:ident {}) => {
        impl crate::quote::ToTokens for $n {
            fn to_tokens(&self, $ys: &mut crate::pm2::Stream) {
                match self {
                    $($arms)*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! custom_kw {
    ($n:ident) => {
        #[allow(non_camel_case_types)]
        pub struct $n<'a> {
            pub span: $crate::pm2::Span,
        }
        #[allow(dead_code, non_snake_case)]
        pub fn $n<__S: $crate::IntoSpans<$crate::pm2::Span>>(span: __S) -> $n {
            $n {
                span: $crate::IntoSpans::into_spans(span),
            }
        }
        const _: () = {
            impl<'a> Default for $n<'a> {
                fn default() -> Self {
                    $n {
                        span: $crate::pm2::Span::call_site(),
                    }
                }
            }
            $crate::impl_parse_for_custom_kw!($n);
            $crate::impl_to_tokens_for_custom_kw!($n);
            $crate::impl_clone_for_custom_kw!($n);
            $crate::impl_traits_for_custom_kw!($n);
        };
    };
}
#[macro_export]
macro_rules! impl_parse_for_custom_kw {
    ($n:ident) => {
        impl<'a> $crate::tok::Custom for $n<'a> {
            fn peek(x: $crate::Cursor) -> bool {
                if let Some((x, _)) = x.ident() {
                    x == std::stringify!($n)
                } else {
                    false
                }
            }
            fn display() -> &'static str {
                std::concat!("`", std::stringify!($n), "`")
            }
        }
        impl<'a> $crate::parse::Parse for $n<'a> {
            fn parse(s: $crate::parse::Stream) -> $crate::Res<$n> {
                s.step(|c| {
                    if let Some((x, rest)) = c.ident() {
                        if x == std::stringify!($n) {
                            return Ok(($n { span: x.span() }, rest));
                        }
                    }
                    Err(c.error(std::concat!("expected `", std::stringify!($n), "`",)))
                })
            }
        }
    };
}
#[macro_export]
macro_rules! impl_to_tokens_for_custom_kw {
    ($n:ident) => {
        impl<'a> $crate::quote::ToTokens for $n<'a> {
            fn to_tokens(&self, ys: &mut $crate::pm2::TokenStream) {
                let y = $crate::Ident::new(std::stringify!($n), self.span);
                $crate::quote::TokenStreamExt::append(ys, y);
            }
        }
    };
}
#[macro_export]
macro_rules! impl_clone_for_custom_kw {
    ($n:ident) => {
        impl<'a> Copy for $n<'a> {}
        #[allow(clippy::expl_impl_clone_on_copy)]
        impl<'a> Clone for $n<'a> {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
#[macro_export]
macro_rules! impl_traits_for_custom_kw {
    ($n:ident) => {
        impl<'a> std::fmt::Debug for $n<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Formatter::write_str(f, std::concat!("Keyword [", std::stringify!($n), "]",))
            }
        }
        impl<'a> Eq for $n<'a> {}
        impl<'a> PartialEq for $n<'a> {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }
        impl<'a> std::hash::Hash for $n<'a> {
            fn hash<__H: std::hash::Hasher>(&self, _: &mut __H) {}
        }
    };
}

#[macro_export]
macro_rules! custom_punctuation {
    ($n:ident, $($tt:tt)+) => {
        pub struct $n {
            pub spans: $crate::custom_punctuation_repr!($($tt)+),
        }
        #[allow(dead_code, non_snake_case)]
        pub fn $n<__S: $crate::IntoSpans<$crate::custom_punctuation_repr!($($tt)+)>>(
            spans: __S,
        ) -> $n {
            let _ = 0 $(+ $crate::custom_punctuation_len!(strict, $tt))*;
            $n {
                spans: $crate::IntoSpans::into_spans(spans)
            }
        }
        const _: () = {
            impl<'a> Default for $n<'a> {
                fn default() -> Self {
                    $n($crate::pm2::Span::call_site())
                }
            }
            $crate::impl_parse_for_custom_punct!($n, $($tt)+);
            $crate::impl_to_tokens_for_custom_punct!($n, $($tt)+);
            $crate::impl_clone_for_custom_punct!($n, $($tt)+);
            $crate::impl_traits_for_custom_punct!($n, $($tt)+);
        };
    };
}
#[macro_export]
macro_rules! impl_parse_for_custom_punct {
    ($n:ident, $($tt:tt)+) => {
        impl $crate::tok::Custom for $n {
            fn peek(x: $crate::Cursor) -> bool {
                $crate::tok::peek_punct(x, $crate::stringify_punct!($($tt)+))
            }
            fn display() -> &'static str {
                std::concat!("`", $crate::stringify_punct!($($tt)+), "`")
            }
        }
        impl $crate::parse::Parse for $n {
            fn parse(s: $crate::parse::Stream) -> $crate::Res<$n> {
                let ys: $crate::custom_punctuation_repr!($($tt)+) =
                    $crate::tok::parse_punct(s, $crate::stringify_punct!($($tt)+))?;
                Ok($n(ys))
            }
        }
    };
}
#[macro_export]
macro_rules! impl_to_tokens_for_custom_punct {
    ($n:ident, $($tt:tt)+) => {
        impl $crate::quote::ToTokens for $n {
            fn to_tokens(&self, ys: &mut $crate::pm2::TokenStream) {
                $crate::tok::punct_to_tokens($crate::stringify_punct!($($tt)+), &self.spans, ys)
            }
        }
    };
}
#[macro_export]
macro_rules! impl_clone_for_custom_punct {
    ($n:ident, $($tt:tt)+) => {
        impl<'a> Copy for $n<'a> {}
        #[allow(clippy::expl_impl_clone_on_copy)]
        impl<'a> Clone for $n<'a> {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
#[macro_export]
macro_rules! impl_traits_for_custom_punct {
    ($n:ident, $($tt:tt)+) => {
        impl<'a> std::fmt::Debug for $n<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Formatter::write_str(f, std::stringify!($n))
            }
        }
        impl<'a> Eq for $n<'a> {}
        impl<'a> PartialEq for $n<'a> {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }
        impl<'a> std::hash::Hash for $n<'a> {
            fn hash<__H: std::hash::Hasher>(&self, _: &mut __H) {}
        }
    };
}
#[macro_export]
macro_rules! custom_punctuation_repr {
    ($($tt:tt)+) => {
        [$crate::pm2::Span; 0 $(+ $crate::custom_punctuation_len!(lenient, $tt))+]
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
        std::concat!($(std::stringify!($tt)),+)
    };
}

#[macro_export]
macro_rules! parse_quote {
    ($($tt:tt)*) => {
        $crate::parse::parse_quote_fn($crate::quote::quote!($($tt)*))
    };
}
#[macro_export]
macro_rules! parse_quote_spanned {
    ($span:expr=> $($tt:tt)*) => {
        $crate::parse::parse_quote_fn($crate::quote::quote_spanned!($span=> $($tt)*))
    };
}

#[macro_export]
macro_rules! parse_macro_input {
    ($n:ident as $ty:ty) => {
        match $crate::parse::<$ty>($n) {
            Ok(x) => x,
            Err(x) => {
                return $crate::pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($n:ident with $p:path) => {
        match $crate::parse::Parser::parse($p, $n) {
            Ok(x) => x,
            Err(x) => {
                return $crate::pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($n:ident) => {
        $crate::parse_macro_input!($n as _)
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
    ($n:ident in $s:expr) => {
        match $crate::parse::parse_parens(&$s) {
            Ok(x) => {
                $n = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}

#[macro_export]
macro_rules! braced {
    ($n:ident in $s:expr) => {
        match $crate::parse::parse_braces(&$s) {
            Ok(x) => {
                $n = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}

#[macro_export]
macro_rules! bracketed {
    ($n:ident in $s:expr) => {
        match $crate::parse::parse_brackets(&$s) {
            Ok(x) => {
                $n = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}
