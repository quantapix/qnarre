use super::*;
use std::iter;

#[macro_export]
macro_rules! braced {
    ($n:ident in $s:expr) => {
        match parse::parse_braces(&$s) {
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
        match parse::parse_brackets(&$s) {
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
macro_rules! parenthed {
    ($n:ident in $s:expr) => {
        match parse::parse_parenths(&$s) {
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
macro_rules! parse_quote {
    ($($tt:tt)*) => {
        parse::parse_quote(quote!($($tt)*))
    };
}
#[macro_export]
macro_rules! parse_quote_spanned {
    ($s:expr=> $($tt:tt)*) => {
        parse::parse_quote(quote_spanned!($s => $($tt)*))
    };
}
#[macro_export]
macro_rules! parse_mac_input {
    ($n:ident as $ty:ty) => {
        match parse::<$ty>($n) {
            Ok(x) => x,
            Err(x) => {
                return pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($n:ident with $p:path) => {
        match Parser::parse($p, $n) {
            Ok(x) => x,
            Err(x) => {
                return pm2::Stream::from(x.to_compile_error());
            },
        }
    };
    ($n:ident) => {
        parse_mac_input!($n as _)
    };
}

#[macro_export]
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
                let (y, x) = parse_delim(s)?;
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
impl Eq for Mac {}
impl PartialEq for Mac {
    fn eq(&self, x: &Self) -> bool {
        self.path == x.path && self.delim == x.delim && StreamHelper(&self.toks) == StreamHelper(&x.toks)
    }
}
impl<H> Hash for Mac
where
    H: Hasher,
{
    fn hash(&self, h: &mut H) {
        self.path.hash(h);
        self.delim.hash(h);
        StreamHelper(&self.toks).hash(h);
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
impl<V> Visit for Mac
where
    V: Visitor + ?Sized,
{
    fn visit(&self, v: &mut V) {
        &self.path.visit(v);
        &self.delim.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        &mut self.path.visit_mut(v);
        &mut self.delim.visit_mut(v);
    }
}

pub fn parse_delim(s: Stream) -> Res<(tok::Delim, pm2::Stream)> {
    s.step(|x| {
        if let Some((pm2::Tree::Group(x), rest)) = x.token_tree() {
            let s = x.delim_span();
            let delim = match x.delim() {
                pm2::Delim::Parenth => tok::Delim::Parenth(tok::Parenth(s)),
                pm2::Delim::Brace => tok::Delim::Brace(tok::Brace(s)),
                pm2::Delim::Bracket => tok::Delim::Bracket(tok::Bracket(s)),
                pm2::Delim::None => {
                    return Err(x.err("expected delim"));
                },
            };
            Ok(((delim, x.stream()), rest))
        } else {
            Err(x.err("expected delim"))
        }
    })
}

macro_rules! enum_from {
    ($n:ident::Verbatim, $($s:ident)::+) => {};
    ($n:ident::$v:ident, $($s:ident)::+) => {
        impl From<$($s)::+> for $n {
            fn from(x: $($s)::+) -> $n {
                $n::$v(x)
            }
        }
    };
}
macro_rules! lower_of {
    (
        ($($x:tt)*) $n:ident {
            $v:ident,
            $($rest:tt)*
        }
    ) => {
        lower_of!(
            ($($x)* $n::$v => {})
            $n { $($rest)* }
        );
    };
    (
        ($($x:tt)*) $n:ident {
            $v:ident $($s:ident)::+,
            $($rest:tt)*
        }
    ) => {
        lower_of!(
            ($($x)* $n::$v(x) => x.lower(s),)
            $n { $($rest)* }
        );
    };
    (($($x:tt)*) $n:ident {}) => {
        impl Lower for $n {
            fn lower(&self, s: &mut pm2::Stream) {
                match self {
                    $($x)*
                }
            }
        }
    };
}
macro_rules! enum_of_impl {
    (
        $n:ident {
            $(
                $v:ident $( ($($s:ident)::+) )*,
            )*
        }
    ) => {
        $($(
            enum_from!($n::$v, $($s)::+);
        )*)*
        $(lower_of! {
            ()
            $n {
                $(
                    $v $($s)::+,
                )*
            }
        })*
    };
}
#[macro_export]
macro_rules! enum_of_structs {
    (
        pub enum $n:ident $tt:tt
        $($rest:tt)*
    ) => {
        pub enum $n $tt
        enum_of_impl!($n $tt $($rest)*);
    };
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
        impl Hash for $n {
            fn hash<H: Hasher>(&self, _: &mut H) {}
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
            pub span: pm2::Span,
        }
        pub fn $n<S: pm2::IntoSpans<pm2::Span>>(s: S) -> $n {
            $n { span: s.into_spans() }
        }
        const _: () = {
            impl Default for $n {
                fn default() -> Self {
                    $n {
                        span: pm2::Span::call_site(),
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
        [$crate::pm2::Span; 0 $(+ punct_len!(lenient, $tt))+]
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
        impl Hash for $n {
            fn hash<H: Hasher>(&self, _: &mut H) {}
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
                    $n(pm2::Span::call_site())
                }
            }
            parse_for_punct!($n, $($tt)+);
            lower_for_punct!($n, $($tt)+);
            clone_for_punct!($n, $($tt)+);
            traits_for_punct!($n, $($tt)+);
        };
    };
}

#[macro_export]
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
#[macro_export]
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

            $crate::test_derive!($n { $($i)* } expands to { $($o)* } no_build);
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
