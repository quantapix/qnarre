use super::*;
use std::{
    collections::btree_set::{self, BTreeSet},
    iter,
    ops::BitOr,
    slice,
};

macro_rules! Token {
    [abstract]    => { tok::Abstract };
    [as]          => { tok::As };
    [async]       => { tok::Async };
    [auto]        => { tok::Auto };
    [await]       => { tok::Await };
    [become]      => { tok::Become };
    [box]         => { tok::Box };
    [break]       => { tok::Break };
    [const]       => { tok::Const };
    [continue]    => { tok::Continue };
    [crate]       => { tok::Crate };
    [default]     => { tok::Default };
    [do]          => { tok::Do };
    [dyn]         => { tok::Dyn };
    [else]        => { tok::Else };
    [enum]        => { tok::Enum };
    [extern]      => { tok::Extern };
    [final]       => { tok::Final };
    [fn]          => { tok::Fn };
    [for]         => { tok::For };
    [if]          => { tok::If };
    [impl]        => { tok::Impl };
    [in]          => { tok::In };
    [let]         => { tok::Let };
    [loop]        => { tok::Loop };
    [macro]       => { tok::Macro };
    [match]       => { tok::Match };
    [mod]         => { tok::Mod };
    [move]        => { tok::Move };
    [mut]         => { tok::Mut };
    [override]    => { tok::Override };
    [priv]        => { tok::Priv };
    [pub]         => { tok::Pub };
    [ref]         => { tok::Ref };
    [return]      => { tok::Return };
    [Self]        => { tok::SelfType };
    [self]        => { tok::SelfValue };
    [static]      => { tok::Static };
    [struct]      => { tok::Struct };
    [super]       => { tok::Super };
    [trait]       => { tok::Trait };
    [try]         => { tok::Try };
    [type]        => { tok::Type };
    [typeof]      => { tok::Typeof };
    [union]       => { tok::Union };
    [unsafe]      => { tok::Unsafe };
    [unsized]     => { tok::Unsized };
    [use]         => { tok::Use };
    [virtual]     => { tok::Virtual };
    [where]       => { tok::Where };
    [while]       => { tok::While };
    [yield]       => { tok::Yield };
    [&]           => { tok::And };
    [&&]          => { tok::AndAnd };
    [&=]          => { tok::AndEq };
    [@]           => { tok::At };
    [^]           => { tok::Caret };
    [^=]          => { tok::CaretEq };
    [:]           => { tok::Colon };
    [,]           => { tok::Comma };
    [$]           => { tok::Dollar };
    [.]           => { tok::Dot };
    [..]          => { tok::DotDot };
    [...]         => { tok::DotDotDot };
    [..=]         => { tok::DotDotEq };
    [=]           => { tok::Eq };
    [==]          => { tok::EqEq };
    [=>]          => { tok::FatArrow };
    [>=]          => { tok::Ge };
    [>]           => { tok::Gt };
    [<-]          => { tok::LArrow };
    [<=]          => { tok::Le };
    [<]           => { tok::Lt };
    [-]           => { tok::Minus };
    [-=]          => { tok::MinusEq };
    [!=]          => { tok::Ne };
    [!]           => { tok::Not };
    [|]           => { tok::Or };
    [|=]          => { tok::OrEq };
    [||]          => { tok::OrOr };
    [::]          => { tok::PathSep };
    [%]           => { tok::Percent };
    [%=]          => { tok::PercentEq };
    [+]           => { tok::Plus };
    [+=]          => { tok::PlusEq };
    [#]           => { tok::Pound };
    [?]           => { tok::Question };
    [->]          => { tok::RArrow };
    [;]           => { tok::Semi };
    [<<]          => { tok::Shl };
    [<<=]         => { tok::ShlEq };
    [>>]          => { tok::Shr };
    [>>=]         => { tok::ShrEq };
    [/]           => { tok::Slash };
    [/=]          => { tok::SlashEq };
    [*]           => { tok::Star };
    [*=]          => { tok::StarEq };
    [~]           => { tok::Tilde };
    [_]           => { tok::Underscore };
}

macro_rules! impl_From {
    ($n:ident::Verbatim, $($y:ident)::+) => {};
    ($n:ident::$x:ident, $($y:ident)::+) => {
        impl From<$($y)::+> for $n {
            fn from(x: $($y)::+) -> $n {
                $n::$x(x)
            }
        }
    };
}
macro_rules! impl_Lower {
    (
        ($($ys:tt)*)
        $n:ident {
            $x:ident ()
            $($xs:tt)*
        }
    ) => {
        impl_Lower!(
            ($($ys)* $n::$x => {})
            $n { $($xs)* }
        );
    };
    (
        ($($ys:tt)*)
        $n:ident {
            $x:ident ( $($s:ident)::+, )
            $($xs:tt)*
        }
    ) => {
        impl_Lower!(
            ($($ys)* $n::$x(x) => x.lower(s), )
            $n { $($xs)* }
        );
    };
    (($($ys:tt)*) $n:ident {}) => {
        impl Lower for $n {
            fn lower(&self, s: &mut pm2::Stream) {
                match self {
                    $($ys)*
                }
            }
        }
    };
}
macro_rules! impl_enum {
    (
        $n:ident { $( $x:ident $( ($($xs:ident)::+) )*, )* }
    ) => {
        $( $( impl_From!($n::$x, $($xs)::+); )* )*
        impl_Lower!{ () $n { $( $x ( $( $($xs)::+, )* ) )* } }
    };
}
macro_rules! enum_of_structs {
    (
        pub enum $n:ident $x:tt $($xs:tt)*
    ) => {
        pub enum $n $x
        impl_enum!($n $x $($xs)*);
    };
}

macro_rules! braced {
    ($y:ident in $s:expr) => {
        match parse::parse_braces(&$s) {
            Ok(x) => {
                $y = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}
macro_rules! bracketed {
    ($y:ident in $s:expr) => {
        match parse::parse_brackets(&$s) {
            Ok(x) => {
                $y = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}
macro_rules! parenthed {
    ($y:ident in $s:expr) => {
        match parse::parse_parenths(&$s) {
            Ok(x) => {
                $y = x.buf;
                x.tok
            },
            Err(x) => {
                return Err(x);
            },
        }
    };
}

macro_rules! parse_quote {
    ($($tt:tt)*) => {
        parse::parse_quote(quote!($($tt)*))
    };
}
macro_rules! parse_quote_spanned {
    ($s:expr=> $($tt:tt)*) => {
        parse::parse_quote(quote_spanned!($s => $($tt)*))
    };
}
macro_rules! parse_mac_input {
    ($y:ident as $ty:ty) => {
        match parse::<$ty>($y) {
            Ok(x) => x,
            Err(x) => {
                return pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($y:ident with $p:path) => {
        match Parser::parse($p, $y) {
            Ok(x) => x,
            Err(x) => {
                return pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($y:ident) => {
        parse_mac_input!($y as _)
    };
}

macro_rules! format_ident_impl {
    ([$s:expr, $($fmt:tt)*]) => {
        ident::make(
            format!($($fmt)*),
            $s,
        )
    };
    ([$old:expr, $($fmt:tt)*] span = $s:expr) => {
        format_ident_impl!([$old, $($fmt)*] span = $s,)
    };
    ([$old:expr, $($fmt:tt)*] span = $s:expr, $($rest:tt)*) => {
        format_ident_impl!([
            Option::Some::<Span>($s),
            $($fmt)*
        ] $($rest)*)
    };
    ([$s:expr, $($fmt:tt)*] $name:ident = $arg:expr) => {
        format_ident_impl!([$s, $($fmt)*] $name = $arg,)
    };
    ([$s:expr, $($fmt:tt)*] $name:ident = $arg:expr, $($rest:tt)*) => {
        match lower::FragAdaptor(&$arg) {
            arg => format_ident_impl!([$s.or(arg.span()), $($fmt)*, $name = arg] $($rest)*),
        }
    };
    ([$s:expr, $($fmt:tt)*] $arg:expr) => {
        format_ident_impl!([$s, $($fmt)*] $arg,)
    };
    ([$s:expr, $($fmt:tt)*] $arg:expr, $($rest:tt)*) => {
        match lower::FragAdaptor(&$arg) {
            arg => format_ident_impl!([$s.or(arg.span()), $($fmt)*, arg] $($rest)*),
        }
    };
}
macro_rules! format_ident {
    ($x:expr) => {
        format_ident_impl!([
            Option::None,
            $x
        ])
    };
    ($x:expr, $($xs:tt)*) => {
        format_ident_impl!([
            Option::None,
            $x
        ] $($xs)*)
    };
}

macro_rules! clone_for_kw {
    ($n:ident) => {
        impl Copy for $n {}
        impl Clone for $n {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
macro_rules! traits_for_kw {
    ($n:ident) => {
        impl Debug for $n {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Formatter::write_str(f, std::concat!("Keyword [", std::stringify!($n), "]",))
            }
        }
        impl Eq for $n {}
        impl PartialEq for $n {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }
        impl<H: Hasher> Hash for $n {
            fn hash(&self, _: &mut H) {}
        }
    };
}
macro_rules! parse_for_kw {
    ($n:ident) => {
        impl tok::Custom for $n {
            fn peek(x: Cursor) -> bool {
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
        impl parse::Parse for $n {
            fn parse(s: parse::Stream) -> Res<$n> {
                s.step(|x| {
                    if let Some((x, rest)) = x.ident() {
                        if x == std::stringify!($n) {
                            return Ok(($n { span: x.span() }, rest));
                        }
                    }
                    Err(x.error(std::concat!("expected `", std::stringify!($n), "`",)))
                })
            }
        }
    };
}
macro_rules! lower_for_kw {
    ($n:ident) => {
        impl Lower for $n {
            fn lower(&self, s: &mut pm2::Stream) {
                let y = Ident::new(std::stringify!($n), self.span);
                mac::StreamExt::append(s, y);
            }
        }
    };
}
macro_rules! custom_kw {
    ($n:ident) => {
        pub struct $n {
            pub span: Span,
        }
        pub fn $n<S: pm2::IntoSpans<Span>>(s: S) -> $n {
            $n { span: s.into_spans() }
        }
        const _: () = {
            impl Default for $n {
                fn default() -> Self {
                    $n {
                        span: Span::call_site(),
                    }
                }
            }
            clone_for_kw!($n);
            traits_for_kw!($n);
            parse_for_kw!($n);
            lower_for_kw!($n);
        };
    };
}

#[allow(unused_macros)]
macro_rules! punct_unexpected {
    () => {};
}
#[rustfmt::skip]
macro_rules! punct_len {
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
    (strict, $tt:tt)     => {{ punct_unexpected!($tt); 0 }};
}
macro_rules! punct_repr {
    ($($tt:tt)+) => {
        [Span; 0 $(+ punct_len!(lenient, $tt))+]
    };
}
macro_rules! stringify_punct {
    ($($tt:tt)+) => {
        std::concat!($(std::stringify!($tt)),+)
    };
}

macro_rules! clone_for_punct {
    ($n:ident, $($tt:tt)+) => {
        impl Copy for $n {}
        impl Clone for $n {
            fn clone(&self) -> Self {
                *self
            }
        }
    };
}
macro_rules! traits_for_punct {
    ($n:ident, $($tt:tt)+) => {
        impl Debug for $n {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Formatter::write_str(f, std::stringify!($n))
            }
        }
        impl Eq for $n {}
        impl PartialEq for $n {
            fn eq(&self, _: &Self) -> bool {
                true
            }
        }
        impl<H: Hasher> Hash for $n {
            fn hash(&self, _: &mut H) {}
        }
    };
}
macro_rules! parse_for_punct {
    ($n:ident, $($tt:tt)+) => {
        impl tok::Custom for $n {
            fn peek(x: Cursor) -> bool {
                tok::peek_punct(x, stringify_punct!($($tt)+))
            }
            fn display() -> &'static str {
                std::concat!("`", stringify_punct!($($tt)+), "`")
            }
        }
        impl parse::Parse for $n {
            fn parse(s: parse::Stream) -> Res<$n> {
                let ys: punct_repr!($($tt)+) =
                    tok::parse_punct(s, stringify_punct!($($tt)+))?;
                Ok($n(ys))
            }
        }
    };
}
macro_rules! lower_for_punct {
    ($n:ident, $($tt:tt)+) => {
        impl Lower for $n {
            fn lower(&self, s: &mut pm2::Stream) {
                tok::punct_lower(stringify_punct!($($tt)+), &self.spans, s)
            }
        }
    };
}
macro_rules! custom_punct {
    ($n:ident, $($tt:tt)+) => {
        pub struct $n {
            pub spans: punct_repr!($($tt)+),
        }
        pub fn $n<S: pm2::IntoSpans<punct_repr!($($tt)+)>>(
            s: S,
        ) -> $n {
            let _ = 0 $(+ punct_len!(strict, $tt))*;
            $n {
                spans: s.into_spans()
            }
        }
        const _: () = {
            impl Default for $n {
                fn default() -> Self {
                    $n(Span::call_site())
                }
            }
            parse_for_punct!($n, $($tt)+);
            lower_for_punct!($n, $($tt)+);
            clone_for_punct!($n, $($tt)+);
            traits_for_punct!($n, $($tt)+);
        };
    };
}

macro_rules! decl_derive {
    ([$n:ident $($tt:tt)*] => $(#[$($attrs:tt)*])* $inner:path) => {
        #[proc_macro_derive($n $(tt)*)]
        #[allow(non_snake_case)]
        $(#[$($attrs)*])*
        pub fn $n(
            s: Stream
        ) -> Stream {
            match parse::<data::Input>(s) {
                Res::Ok(x) => {
                    match Structure::try_new(&x) {
                        Res::Ok(x) => MacroResult::into_stream($inner(x)),
                        Res::Err(x) => {
                            ::core::convert::Into::into(x.to_compile_error())
                        }
                    }
                }
                Err(x) => {
                    ::core::convert::Into::into(x.to_compile_error())
                }
            }
        }
    };
}
macro_rules! decl_attr {
    ([$attribute:ident] => $(#[$($attrs:tt)*])* $inner:path) => {
        #[proc_macro_attribute]
        $(#[$($attrs)*])*
        pub fn $attribute(
            attr: Stream,
            s: Stream,
        ) -> Stream {
            match parse::<data::Input>(s) {
                Res::Ok(x) => match Structure::try_new(&x) {
                    Res::Ok(x) => {
                        MacroResult::into_stream(
                            $inner(::core::convert::Into::into(attr), x)
                        )
                    }
                    Res::Err(x) => {
                        ::core::convert::Into::into(x.to_compile_error())
                    }
                },
                Res::Err(x) => {
                    ::core::convert::Into::into(x.to_compile_error())
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_derive {
    ($n:path { $($i:tt)* } expands to { $($o:tt)* }) => {
        {
            #[allow(dead_code)]
            fn ensure_compiles() {
                $($i)*
                $($o)*
            }
            test_derive!($n { $($i)* } expands to { $($o)* } no_build);
        }
    };
    ($n:path { $($i:tt)* } expands to { $($o:tt)* } no_build) => {
        {
            let i = ::core::stringify!( $($i)* );
            let y = parse_str::<data::Input>(i)
                .expect(::core::concat!(
                    "Failed to parse input to `#[derive(",
                    ::core::stringify!($n),
                    ")]`",
                ));
            let y = $n(Structure::new(&y));
            let y = $MacroResult::into_result(y)
                .expect(::core::concat!(
                    "Procedural macro failed for `#[derive(",
                    ::core::stringify!($n),
                    ")]`",
                ));
            let ys = ::core::stringify!( $($o)* )
                .parse::<pm2::Stream>()
                .expect("output should be a valid TokenStream");
            let mut ys = <pm2::Stream
                as ::core::convert::From<pm2::Stream>>::from(ys);
            if <pm2::Stream as ::std::string::ToString>::to_string(&y)
                != <pm2::Stream as ::std::string::ToString>::to_string(&ys)
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
                    $crate::unpretty_print(&ys),
                    $crate::unpretty_print(&y),
                );
            }
        }
    };
}

#[macro_export]
macro_rules! bind_into_iter {
    ($has_iter:ident $x:ident) => {
        #[allow(unused_mut)]
        let (mut $x, y) = $x.quote_into_iter();
        let $has_iter = $has_iter | y;
    };
}
#[macro_export]
macro_rules! bind_next_or_break {
    ($x:ident) => {
        let $x = match $x.next() {
            Some(x) => mac::RepInterp(x),
            None => break,
        };
    };
}

macro_rules! pounded_names_with_context {
    ($call:ident! $x:tt ($($b1:tt)*) ($($curr:tt)*)) => {
        $(
            pounded_with_context!{$call! $x $b1 $curr}
        )*
    };
}
macro_rules! pounded_names {
    ($call:ident! $x:tt $($xs:tt)*) => {
        pounded_names_with_context!{$call! $x
            (@ $($xs)*)
            ($($xs)* @)
        }
    };
}
#[macro_export]
macro_rules! pounded_with_context {
    ($call:ident! $x:tt $b1:tt ( $($inner:tt)* )) => {
        pounded_names!{$call! $x $($inner)*}
    };
    ($call:ident! $x:tt $b1:tt [ $($inner:tt)* ]) => {
        pounded_names!{$call! $x $($inner)*}
    };
    ($call:ident! $x:tt $b1:tt { $($inner:tt)* }) => {
        pounded_names!{$call! $x $($inner)*}
    };
    ($call:ident!($($x:tt)*) # $var:ident) => {
        $crate::$call!($($x)* $var);
    };
    ($call:ident! $x:tt $b1:tt $curr:tt) => {};
}

macro_rules! quote_token {
    ($x:ident $ys:ident) => {
        lower::push_ident(&mut $ys, stringify!($x));
    };
    (:: $ys:ident) => {
        lower::push_colon2(&mut $ys);
    };
    (( $($x:tt)* ) $ys:ident) => {
        lower::push_group(
            &mut $ys,
            pm2::Delim::Parenthesis,
            quote!($($x)*),
        );
    };
    ([ $($x:tt)* ] $ys:ident) => {
        lower::push_group(
            &mut $ys,
            pm2::Delim::Bracket,
            quote!($($x)*),
        );
    };
    ({ $($x:tt)* } $ys:ident) => {
        lower::push_group(
            &mut $ys,
            pm2::Delim::Brace,
            quote!($($x)*),
        );
    };
    (# $ys:ident) => {
        lower::push_pound(&mut $ys);
    };
    (, $ys:ident) => {
        lower::push_comma(&mut $ys);
    };
    (. $ys:ident) => {
        lower::push_dot(&mut $ys);
    };
    (; $ys:ident) => {
        lower::push_semi(&mut $ys);
    };
    (: $ys:ident) => {
        lower::push_colon(&mut $ys);
    };
    (+ $ys:ident) => {
        lower::push_add(&mut $ys);
    };
    (+= $ys:ident) => {
        lower::push_add_eq(&mut $ys);
    };
    (& $ys:ident) => {
        lower::push_and(&mut $ys);
    };
    (&& $ys:ident) => {
        lower::push_and_and(&mut $ys);
    };
    (&= $ys:ident) => {
        lower::push_and_eq(&mut $ys);
    };
    (@ $ys:ident) => {
        lower::push_at(&mut $ys);
    };
    (! $ys:ident) => {
        lower::push_bang(&mut $ys);
    };
    (^ $ys:ident) => {
        lower::push_caret(&mut $ys);
    };
    (^= $ys:ident) => {
        lower::push_caret_eq(&mut $ys);
    };
    (/ $ys:ident) => {
        lower::push_div(&mut $ys);
    };
    (/= $ys:ident) => {
        lower::push_div_eq(&mut $ys);
    };
    (.. $ys:ident) => {
        lower::push_dot2(&mut $ys);
    };
    (... $ys:ident) => {
        lower::push_dot3(&mut $ys);
    };
    (..= $ys:ident) => {
        lower::push_dot_dot_eq(&mut $ys);
    };
    (= $ys:ident) => {
        lower::push_eq(&mut $ys);
    };
    (== $ys:ident) => {
        lower::push_eq_eq(&mut $ys);
    };
    (>= $ys:ident) => {
        lower::push_ge(&mut $ys);
    };
    (> $ys:ident) => {
        lower::push_gt(&mut $ys);
    };
    (<= $ys:ident) => {
        lower::push_le(&mut $ys);
    };
    (< $ys:ident) => {
        lower::push_lt(&mut $ys);
    };
    (*= $ys:ident) => {
        lower::push_mul_eq(&mut $ys);
    };
    (!= $ys:ident) => {
        lower::push_ne(&mut $ys);
    };
    (| $ys:ident) => {
        lower::push_or(&mut $ys);
    };
    (|= $ys:ident) => {
        lower::push_or_eq(&mut $ys);
    };
    (|| $ys:ident) => {
        lower::push_or_or(&mut $ys);
    };
    (? $ys:ident) => {
        lower::push_question(&mut $ys);
    };
    (-> $ys:ident) => {
        lower::push_rarrow(&mut $ys);
    };
    (<- $ys:ident) => {
        lower::push_larrow(&mut $ys);
    };
    (% $ys:ident) => {
        lower::push_rem(&mut $ys);
    };
    (%= $ys:ident) => {
        lower::push_rem_eq(&mut $ys);
    };
    (=> $ys:ident) => {
        lower::push_fat_arrow(&mut $ys);
    };
    (<< $ys:ident) => {
        lower::push_shl(&mut $ys);
    };
    (<<= $ys:ident) => {
        lower::push_shl_eq(&mut $ys);
    };
    (>> $ys:ident) => {
        lower::push_shr(&mut $ys);
    };
    (>>= $ys:ident) => {
        lower::push_shr_eq(&mut $ys);
    };
    (* $ys:ident) => {
        lower::push_star(&mut $ys);
    };
    (- $ys:ident) => {
        lower::push_sub(&mut $ys);
    };
    (-= $ys:ident) => {
        lower::push_sub_eq(&mut $ys);
    };
    ($x:lifetime $ys:ident) => {
        lower::push_lifetime(&mut $ys, stringify!($x));
    };
    (_ $ys:ident) => {
        lower::push_underscore(&mut $ys);
    };
    ($x:tt $ys:ident) => {
        parse(&mut $ys, stringify!($x));
    };
}
macro_rules! quote_with_context {
    ($ys:ident $b3:tt $b2:tt $b1:tt @ $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $b3:tt $b2:tt $b1:tt (#) ( $($x:tt)* ) * $a3:tt) => {{
        let has_iter = mac::HasNoIter;
        pounded_names!{bind_into_iter!(has_iter) () $($x)*}
        let _: mac::HasIter = has_iter;
        while true {
            pounded_names!{bind_next_or_break!() () $($x)*}
            quote_each!{$ys $($x)*}
        }
    }};
    ($ys:ident $b3:tt $b2:tt # (( $($x:tt)* )) * $a2:tt $a3:tt) => {};
    ($ys:ident $b3:tt # ( $($x:tt)* ) (*) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $b3:tt $b2:tt $b1:tt (#) ( $($x:tt)* ) $sep:tt *) => {{
        let mut _i = 0usize;
        let has_iter = mac::HasNoIter;
        pounded_names!{bind_into_iter!(has_iter) () $($x)*}
        let _: mac::HasIter = has_iter;
        while true {
            pounded_names!{bind_next_or_break!() () $($x)*}
            if _i > 0 {
                quote_token!{$sep $ys}
            }
            _i += 1;
            quote_each!{$ys $($x)*}
        }
    }};
    ($ys:ident $b3:tt $b2:tt # (( $($x:tt)* )) $sep:tt * $a3:tt) => {};
    ($ys:ident $b3:tt # ( $($x:tt)* ) ($sep:tt) * $a2:tt $a3:tt) => {};
    ($ys:ident # ( $($x:tt)* ) * (*) $a1:tt $a2:tt $a3:tt) => {
        quote_token!{* $ys}
    };
    ($ys:ident # ( $($x:tt)* ) $sep:tt (*) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $b3:tt $b2:tt $b1:tt (#) $x:ident $a2:tt $a3:tt) => {
        Lower::lower(&$x, &mut $ys);
    };
    ($ys:ident $b3:tt $b2:tt # ($x:ident) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $b3:tt $b2:tt $b1:tt ($x:tt) $a1:tt $a2:tt $a3:tt) => {
        quote_token!{$x $ys}
    };
}
macro_rules! quote_all {
    ($ys:ident
        ($($b3:tt)*) ($($b2:tt)*) ($($b1:tt)*)
        ($($x:tt)*)
        ($($a1:tt)*) ($($a2:tt)*) ($($a3:tt)*)
    ) => {
        $(
            quote_with_context!{$ys $b3 $b2 $b1 $x $a1 $a2 $a3}
        )*
    };
}
macro_rules! quote_each {
    ($ys:ident $($xs:tt)*) => {
        quote_all!{$ys
            (@ @ @ @ @ @ $($xs)*)
            (@ @ @ @ @ $($xs)* @)
            (@ @ @ @ $($xs)* @ @)
            (@ @ @ $(($xs))* @ @ @)
            (@ @ $($xs)* @ @ @ @)
            (@ $($xs)* @ @ @ @ @)
            ($($xs)* @ @ @ @ @ @)
        }
    };
}

macro_rules! quote_token_spanned {
    ($x:ident $ys:ident $s:ident) => {
        lower::push_ident_spanned(&mut $ys, $s, stringify!($x));
    };
    (:: $ys:ident $s:ident) => {
        lower::push_colon2_spanned(&mut $ys, $s);
    };
    (( $($x:tt)* ) $ys:ident $s:ident) => {
        lower::push_group_spanned(
            &mut $ys,
            $s,
            pm2::Delim::Parenthesis,
            quote_spanned!($s=> $($x)*),
        );
    };
    ([ $($x:tt)* ] $ys:ident $s:ident) => {
        lower::push_group_spanned(
            &mut $ys,
            $s,
            pm2::Delim::Bracket,
            quote_spanned!($s=> $($x)*),
        );
    };
    ({ $($x:tt)* } $ys:ident $s:ident) => {
        lower::push_group_spanned(
            &mut $ys,
            $s,
            pm2::Delim::Brace,
            quote_spanned!($s=> $($x)*),
        );
    };
    (# $ys:ident $s:ident) => {
        lower::push_pound_spanned(&mut $ys, $s);
    };
    (, $ys:ident $s:ident) => {
        lower::push_comma_spanned(&mut $ys, $s);
    };
    (. $ys:ident $s:ident) => {
        lower::push_dot_spanned(&mut $ys, $s);
    };
    (; $ys:ident $s:ident) => {
        lower::push_semi_spanned(&mut $ys, $s);
    };
    (: $ys:ident $s:ident) => {
        lower::push_colon_spanned(&mut $ys, $s);
    };
    (+ $ys:ident $s:ident) => {
        lower::push_add_spanned(&mut $ys, $s);
    };
    (+= $ys:ident $s:ident) => {
        lower::push_add_eq_spanned(&mut $ys, $s);
    };
    (& $ys:ident $s:ident) => {
        lower::push_and_spanned(&mut $ys, $s);
    };
    (&& $ys:ident $s:ident) => {
        lower::push_and_and_spanned(&mut $ys, $s);
    };
    (&= $ys:ident $s:ident) => {
        lower::push_and_eq_spanned(&mut $ys, $s);
    };
    (@ $ys:ident $s:ident) => {
        lower::push_at_spanned(&mut $ys, $s);
    };
    (! $ys:ident $s:ident) => {
        lower::push_bang_spanned(&mut $ys, $s);
    };
    (^ $ys:ident $s:ident) => {
        lower::push_caret_spanned(&mut $ys, $s);
    };
    (^= $ys:ident $s:ident) => {
        lower::push_caret_eq_spanned(&mut $ys, $s);
    };
    (/ $ys:ident $s:ident) => {
        lower::push_div_spanned(&mut $ys, $s);
    };
    (/= $ys:ident $s:ident) => {
        lower::push_div_eq_spanned(&mut $ys, $s);
    };
    (.. $ys:ident $s:ident) => {
        lower::push_dot2_spanned(&mut $ys, $s);
    };
    (... $ys:ident $s:ident) => {
        lower::push_dot3_spanned(&mut $ys, $s);
    };
    (..= $ys:ident $s:ident) => {
        lower::push_dot_dot_eq_spanned(&mut $ys, $s);
    };
    (= $ys:ident $s:ident) => {
        lower::push_eq_spanned(&mut $ys, $s);
    };
    (== $ys:ident $s:ident) => {
        lower::push_eq_eq_spanned(&mut $ys, $s);
    };
    (>= $ys:ident $s:ident) => {
        lower::push_ge_spanned(&mut $ys, $s);
    };
    (> $ys:ident $s:ident) => {
        lower::push_gt_spanned(&mut $ys, $s);
    };
    (<= $ys:ident $s:ident) => {
        lower::push_le_spanned(&mut $ys, $s);
    };
    (< $ys:ident $s:ident) => {
        lower::push_lt_spanned(&mut $ys, $s);
    };
    (*= $ys:ident $s:ident) => {
        lower::push_mul_eq_spanned(&mut $ys, $s);
    };
    (!= $ys:ident $s:ident) => {
        lower::push_ne_spanned(&mut $ys, $s);
    };
    (| $ys:ident $s:ident) => {
        lower::push_or_spanned(&mut $ys, $s);
    };
    (|= $ys:ident $s:ident) => {
        lower::push_or_eq_spanned(&mut $ys, $s);
    };
    (|| $ys:ident $s:ident) => {
        lower::push_or_or_spanned(&mut $ys, $s);
    };
    (? $ys:ident $s:ident) => {
        lower::push_question_spanned(&mut $ys, $s);
    };
    (-> $ys:ident $s:ident) => {
        lower::push_rarrow_spanned(&mut $ys, $s);
    };
    (<- $ys:ident $s:ident) => {
        lower::push_larrow_spanned(&mut $ys, $s);
    };
    (% $ys:ident $s:ident) => {
        lower::push_rem_spanned(&mut $ys, $s);
    };
    (%= $ys:ident $s:ident) => {
        lower::push_rem_eq_spanned(&mut $ys, $s);
    };
    (=> $ys:ident $s:ident) => {
        lower::push_fat_arrow_spanned(&mut $ys, $s);
    };
    (<< $ys:ident $s:ident) => {
        lower::push_shl_spanned(&mut $ys, $s);
    };
    (<<= $ys:ident $s:ident) => {
        lower::push_shl_eq_spanned(&mut $ys, $s);
    };
    (>> $ys:ident $s:ident) => {
        lower::push_shr_spanned(&mut $ys, $s);
    };
    (>>= $ys:ident $s:ident) => {
        lower::push_shr_eq_spanned(&mut $ys, $s);
    };
    (* $ys:ident $s:ident) => {
        lower::push_star_spanned(&mut $ys, $s);
    };
    (- $ys:ident $s:ident) => {
        lower::push_sub_spanned(&mut $ys, $s);
    };
    (-= $ys:ident $s:ident) => {
        lower::push_sub_eq_spanned(&mut $ys, $s);
    };
    ($x:lifetime $ys:ident $s:ident) => {
        lower::push_lifetime_spanned(&mut $ys, $s, stringify!($x));
    };
    (_ $ys:ident $s:ident) => {
        lower::push_underscore_spanned(&mut $ys, $s);
    };
    ($x:tt $ys:ident $s:ident) => {
        $crate::parse_spanned(&mut $ys, $s, stringify!($x));
    };
}
macro_rules! quote_with_context_spanned {
    ($ys:ident $s:ident $b3:tt $b2:tt $b1:tt @ $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt $b2:tt $b1:tt (#) ( $($x:tt)* ) * $a3:tt) => {{
        let has_iter = mac::HasNoIter;
        pounded_names!{bind_into_iter!(has_iter) () $($x)*}
        let _: mac::HasIter = has_iter;
        while true {
            pounded_names!{bind_next_or_break!() () $($x)*}
            quote_each_spanned!{$ys $s $($x)*}
        }
    }};
    ($ys:ident $s:ident $b3:tt $b2:tt # (( $($x:tt)* )) * $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt # ( $($x:tt)* ) (*) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt $b2:tt $b1:tt (#) ( $($x:tt)* ) $sep:tt *) => {{
        let mut _i = 0usize;
        let has_iter = mac::HasNoIter;
        pounded_names!{bind_into_iter!(has_iter) () $($x)*}
        let _: mac::HasIter = has_iter;
        while true {
            pounded_names!{bind_next_or_break!() () $($x)*}
            if _i > 0 {
                quote_token_spanned!{$sep $ys $s}
            }
            _i += 1;
            quote_each_spanned!{$ys $s $($x)*}
        }
    }};
    ($ys:ident $s:ident $b3:tt $b2:tt # (( $($x:tt)* )) $sep:tt * $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt # ( $($x:tt)* ) ($sep:tt) * $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident # ( $($x:tt)* ) * (*) $a1:tt $a2:tt $a3:tt) => {
        quote_token_spanned!{* $ys $s}
    };
    ($ys:ident $s:ident # ( $($x:tt)* ) $sep:tt (*) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt $b2:tt $b1:tt (#) $var:ident $a2:tt $a3:tt) => {
        Lower::lower(&$var, &mut $ys);
    };
    ($ys:ident $s:ident $b3:tt $b2:tt # ($var:ident) $a1:tt $a2:tt $a3:tt) => {};
    ($ys:ident $s:ident $b3:tt $b2:tt $b1:tt ($curr:tt) $a1:tt $a2:tt $a3:tt) => {
        quote_token_spanned!{$curr $ys $s}
    };
}
macro_rules! quote_all_spanned {
    ($ys:ident $s:ident
        ($($b3:tt)*) ($($b2:tt)*) ($($b1:tt)*)
        ($($curr:tt)*)
        ($($a1:tt)*) ($($a2:tt)*) ($($a3:tt)*)
    ) => {
        $(
            quote_with_context_spanned!{$ys $s $b3 $b2 $b1 $curr $a1 $a2 $a3}
        )*
    };
}
macro_rules! quote_each_spanned {
    ($ys:ident $s:ident $($xs:tt)*) => {
        quote_all_spanned!{$ys $s
            (@ @ @ @ @ @ $($xs)*)
            (@ @ @ @ @ $($xs)* @)
            (@ @ @ @ $($xs)* @ @)
            (@ @ @ $(($xs))* @ @ @)
            (@ @ $($xs)* @ @ @ @)
            (@ $($xs)* @ @ @ @ @)
            ($($xs)* @ @ @ @ @ @)
        }
    };
}

#[macro_export]
macro_rules! quote {
    () => {
        Stream::new()
    };
    ($x:tt) => {{
        let mut ys = Stream::new();
        quote_token!{$x ys}
        ys
    }};
    (# $x:ident) => {{
        let mut ys = Stream::new();
        Lower::lower(&$x, &mut ys);
        ys
    }};
    ($x1:tt $x2:tt) => {{
        let mut ys = Stream::new();
        quote_token!{$x1 ys}
        quote_token!{$x2 ys}
        ys
    }};
    ($($x:tt)*) => {{
        let mut ys = Stream::new();
        quote_each!{ys $($x)*}
        ys
    }};
}
#[macro_export]
macro_rules! quote_spanned {
    ($s:expr=>) => {{
        let _: Span = lower::get_span($s).__into_span();
        Stream::new()
    }};
    ($s:expr=> $x:tt) => {{
        let mut ys = Stream::new();
        let y: Span = lower::get_span($s).__into_span();
        quote_token_spanned!{$x ys y}
        ys
    }};
    ($s:expr=> # $x:ident) => {{
        let mut ys = Stream::new();
        let _: Span = lower::get_span($s).__into_span();
        Lower::lower(&$x, &mut ys);
        ys
    }};
    ($s:expr=> $x1:tt $x2:tt) => {{
        let mut ys = Stream::new();
        let y: Span = lower::get_span($s).__into_span();
        quote_token_spanned!{$x1 ys y}
        quote_token_spanned!{$x2 ys y}
        ys
    }};
    ($s:expr=> $($x:tt)*) => {{
        let mut ys = Stream::new();
        let y: Span = lower::get_span($s).__into_span();
        quote_each_spanned!{ys y $($x)*}
        ys
    }};
}

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
    fn parse(s: Stream) -> Res<Self> {
        let toks;
        Ok(Mac {
            path: s.call(Path::parse_mod_style)?,
            bang: s.parse()?,
            delim: {
                let (y, x) = tok::parse_delim(s)?;
                toks = x;
                y
            },
            toks,
        })
    }
}
impl Lower for Mac {
    fn lower(&self, s: &mut Stream) {
        self.path.lower(s);
        self.bang.lower(s);
        self.delim.surround(s, self.toks.clone());
    }
}
impl Clone for Mac {
    fn clone(&self) -> Self {
        Mac {
            path: self.path.clone(),
            bang: self.bang.clone(),
            delim: self.delim.clone(),
            toks: self.toks.clone(),
        }
    }
}
impl Debug for Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Macro");
        f.field("path", &self.path);
        f.field("bang", &self.bang);
        f.field("delimiter", &self.delim);
        f.field("toks", &self.toks);
        f.finish()
    }
}
impl Eq for Mac {}
impl PartialEq for Mac {
    fn eq(&self, x: &Self) -> bool {
        self.path == x.path && self.delim == x.delim && StreamHelper(&self.toks) == StreamHelper(&x.toks)
    }
}
impl Pretty for Mac {
    fn pretty_with_args(&self, p: &mut Print, x: &Option<pretty::Args>) {
        let Some(x, semi) = pretty::Args::ident_semi(x);
        if self.path.is_ident("macro_rules") {
            if let Some(x) = x {
                p.macro_rules(x, &self.toks);
                return;
            }
        }
        if x.is_none() && p.standard_library_macro(self, semi) {
            return;
        }
        &self.path.pretty_with_args(p, path::Kind::Simple);
        p.word("!");
        if let Some(x) = x {
            p.nbsp();
            x.pretty(x);
        }
        use tok::Delim::*;
        let (open, close, f) = match self.delim {
            Brace(_) => (" {", "}", Print::hardbreak as fn(&mut Self)),
            Bracket(_) => ("[", "]", Print::zerobreak as fn(&mut Self)),
            Parenth(_) => ("(", ")", Print::zerobreak as fn(&mut Self)),
        };
        p.word(open);
        if !self.toks.is_empty() {
            p.cbox(INDENT);
            f(p);
            p.ibox(0);
            p.macro_rules_tokens(self.toks.clone(), false);
            p.end();
            f(p);
            p.offset(-INDENT);
            p.end();
        }
        p.word(close);
        if semi {
            match self.delim {
                Parenth(_) | Bracket(_) => p.word(";"),
                Brace(_) => {},
            }
        }
    }
}
impl<F: Folder + ?Sized> Fold for Mac {
    fn fold(&self, f: &mut F) {
        Mac {
            path: self.path.fold(f),
            bang: self.bang,
            delim: self.delim.fold(f),
            toks: self.toks,
        }
    }
}
impl<H: Hasher> Hash for Mac {
    fn hash(&self, h: &mut H) {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Mac {
    fn visit(&self, v: &mut V) {
        &self.path.visit(v);
        &self.delim.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.path.visit_mut(v);
        &mut self.delim.visit_mut(v);
    }
}

pub trait StreamExt {
    fn append<U>(&mut self, x: U)
    where
        U: Into<pm2::Tree>;
    fn append_all<I>(&mut self, xs: I)
    where
        I: IntoIterator,
        I::Item: Lower;
    fn append_sep<I, U>(&mut self, xs: I, y: U)
    where
        I: IntoIterator,
        I::Item: Lower,
        U: Lower;
    fn append_term<I, U>(&mut self, xs: I, y: U)
    where
        I: IntoIterator,
        I::Item: Lower,
        U: Lower;
}
impl StreamExt for pm2::Stream {
    fn append<U>(&mut self, x: U)
    where
        U: Into<pm2::Tree>,
    {
        self.extend(iter::once(x.into()));
    }
    fn append_all<I>(&mut self, xs: I)
    where
        I: IntoIterator,
        I::Item: Lower,
    {
        for x in xs {
            x.lower(self);
        }
    }
    fn append_sep<I, U>(&mut self, xs: I, y: U)
    where
        I: IntoIterator,
        I::Item: Lower,
        U: Lower,
    {
        for (i, x) in xs.into_iter().enumerate() {
            if i > 0 {
                y.lower(self);
            }
            x.lower(self);
        }
    }
    fn append_term<I, U>(&mut self, xs: I, y: U)
    where
        I: IntoIterator,
        I::Item: Lower,
        U: Lower,
    {
        for x in xs {
            x.lower(self);
            y.lower(self);
        }
    }
}

pub struct HasIter;
impl BitOr<HasIter> for HasIter {
    type Output = HasIter;
    fn bitor(self, _: HasIter) -> HasIter {
        HasIter
    }
}
impl BitOr<HasNoIter> for HasIter {
    type Output = HasIter;
    fn bitor(self, _: HasNoIter) -> HasIter {
        HasIter
    }
}

pub struct HasNoIter;
impl BitOr<HasIter> for HasNoIter {
    type Output = HasIter;
    fn bitor(self, _: HasIter) -> HasIter {
        HasIter
    }
}
impl BitOr<HasNoIter> for HasNoIter {
    type Output = HasNoIter;
    fn bitor(self, _: HasNoIter) -> HasNoIter {
        HasNoIter
    }
}

pub trait RepIterExt: Iterator + Sized {
    fn quote_into_iter(self) -> (Self, HasIter) {
        (self, HasIter)
    }
}
impl<T: Iterator> RepIterExt for T {}

pub trait RepLowerExt {
    fn next(&self) -> Option<&Self> {
        Some(self)
    }
    fn quote_into_iter(&self) -> (&Self, HasNoIter) {
        (self, HasNoIter)
    }
}
impl<T: Lower + ?Sized> RepLowerExt for T {}

#[derive(Copy, Clone)]
pub struct RepInterp<T>(pub T);
impl<T> RepInterp<T> {
    pub fn next(self) -> Option<T> {
        Some(self.0)
    }
}
impl<T: Iterator> Iterator for RepInterp<T> {
    type Item = T::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
impl<T: Lower> Lower for RepInterp<T> {
    fn lower(&self, s: &mut Stream) {
        self.0.lower(s);
    }
}

pub trait RepAsIterExt<'q> {
    type Iter: Iterator;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter);
}
impl<'q, 'a, T: RepAsIterExt<'q> + ?Sized> RepAsIterExt<'q> for &'a T {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        <T as RepAsIterExt>::quote_into_iter(*self)
    }
}
impl<'q, 'a, T: RepAsIterExt<'q> + ?Sized> RepAsIterExt<'q> for &'a mut T {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        <T as RepAsIterExt>::quote_into_iter(*self)
    }
}
impl<'q, T: 'q> RepAsIterExt<'q> for [T] {
    type Iter = slice::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: 'q> RepAsIterExt<'q> for Vec<T> {
    type Iter = slice::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: 'q> RepAsIterExt<'q> for BTreeSet<T> {
    type Iter = btree_set::Iter<'q, T>;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        (self.iter(), HasIter)
    }
}
impl<'q, T: RepAsIterExt<'q>> RepAsIterExt<'q> for RepInterp<T> {
    type Iter = T::Iter;
    fn quote_into_iter(&'q self) -> (Self::Iter, HasIter) {
        self.0.quote_into_iter()
    }
}
