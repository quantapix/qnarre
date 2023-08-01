use super::*;
use std::{iter, slice};

pub enum Style {
    Outer,
    Inner(Token![!]),
}

pub struct Attr {
    pub pound: Token![#],
    pub style: Style,
    pub bracket: tok::Bracket,
    pub meta: Meta,
}
impl Attr {
    pub fn path(&self) -> &Path {
        self.meta.path()
    }
    pub fn parse_args<T: Parse>(&self) -> Res<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, p: T) -> Res<T::Output> {
        use Meta::*;
        match &self.meta {
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!(
                    "expected args in parentheses: {}[{}(...)]",
                    DisplayStyle(&self.style),
                    path::DisplayPath(x),
                ),
            )),
            NameValue(x) => Err(Err::new(
                x.eq.span,
                format_args!(
                    "expected parentheses: {}[{}(...)]",
                    DisplayStyle(&self.style),
                    path::DisplayPath(&x.name),
                ),
            )),
            List(x) => x.parse_args_with(p),
        }
    }
    pub fn parse_nested(&self, x: impl FnMut(Nested) -> Res<()>) -> Res<()> {
        self.parse_args_with(parser(x))
    }
    pub fn parse_inners(s: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        parse_inners(s, &mut y)?;
        Ok(y)
    }
    pub fn parse_outers(s: Stream) -> Res<Vec<Self>> {
        let mut y = Vec::new();
        while s.peek(Token![#]) {
            y.push(s.call(parse_one_outer)?);
        }
        Ok(y)
    }
    fn value(&self, name: &str) -> Option<String> {
        let y = match &self.meta {
            Meta::NameValue(x) if x.name.is_ident(name) => &x.value,
            _ => return None,
        };
        let y = match y {
            expr::Expr::Lit(x) if x.attrs.is_empty() => &x.lit,
            _ => return None,
        };
        match y {
            lit::Lit::Str(x) => Some(x.value()),
            _ => None,
        }
    }
}
impl Lower for Attr {
    fn lower(&self, s: &mut Stream) {
        self.pound.lower(s);
        if let Style::Inner(x) = &self.style {
            x.lower(s);
        }
        self.bracket.surround(s, |s| {
            self.meta.lower(s);
        });
    }
}
impl Pretty for Attr {
    fn pretty(&self, p: &mut Print) {
        use Style::*;
        if let Some(mut x) = self.value("doc") {
            if !x.contains('\n')
                && match self.style {
                    Outer => !x.starts_with('/'),
                    Inner(_) => true,
                }
            {
                trim_trailing(&mut x);
                p.word(match self.style {
                    Outer => "///",
                    Inner(_) => "//!",
                });
                p.word(x);
                p.hardbreak();
                return;
            } else if is_blocklike(&x)
                && match self.style {
                    Outer => !x.starts_with(&['*', '/'][..]),
                    Inner(_) => true,
                }
            {
                trim_interior(&mut x);
                p.word(match self.style {
                    Outer => "/**",
                    Inner(_) => "/*!",
                });
                p.word(x);
                p.word("*/");
                p.hardbreak();
                return;
            }
        } else if let Some(mut x) = self.value("comment") {
            if !x.contains('\n') {
                trim_trailing(&mut x);
                p.word("//");
                p.word(x);
                p.hardbreak();
                return;
            } else if is_blocklike(&x) && !x.starts_with(&['*', '!'][..]) {
                trim_interior(&mut x);
                p.word("/*");
                p.word(x);
                p.word("*/");
                p.hardbreak();
                return;
            }
        }
        p.word(match self.style {
            Outer => "#",
            Inner(_) => "#!",
        });
        p.word("[");
        &self.meta.pretty(p);
        p.word("]");
        p.space();
    }
}
fn trim_trailing(x: &mut String) {
    x.truncate(x.trim_end_matches(' ').len());
}
fn trim_interior(x: &mut String) {
    if !x.contains(" \n") {
        return;
    }
    let mut y = String::with_capacity(x.len());
    let mut xs = x.split('\n').peekable();
    while let Some(x) = xs.next() {
        if xs.peek().is_some() {
            y.push_str(x.trim_end_matches(' '));
            y.push('\n');
        } else {
            y.push_str(x);
        }
    }
    *x = y;
}
fn is_blocklike(x: &str) -> bool {
    let mut depth = 0usize;
    let xs = x.as_bytes();
    let mut i = 0usize;
    let upper = xs.len() - 1;
    while i < upper {
        if xs[i] == b'/' && xs[i + 1] == b'*' {
            depth += 1;
            i += 2;
        } else if xs[i] == b'*' && xs[i + 1] == b'/' {
            if depth == 0 {
                return false;
            }
            depth -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    depth == 0
}

impl Print {
    pub fn outer_attrs(&mut self, xs: &[attr::Attr]) {
        for x in xs {
            if let attr::Style::Outer = x.style {
                x.pretty(self);
            }
        }
    }
    pub fn inner_attrs(&mut self, xs: &[attr::Attr]) {
        for x in xs {
            if let attr::Style::Inner(_) = x.style {
                x.pretty(self);
            }
        }
    }
}

pub trait Filter<'a> {
    type Ret: Iterator<Item = &'a Attr>;
    fn inners(self) -> Self::Ret;
    fn outers(self) -> Self::Ret;
}
impl<'a> Filter<'a> for &'a [Attr] {
    type Ret = iter::Filter<slice::Iter<'a, Attr>, fn(&&Attr) -> bool>;
    fn inners(self) -> Self::Ret {
        fn is_inner(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Inner(_) => true,
                Outer => false,
            }
        }
        self.iter().filter(is_inner)
    }
    fn outers(self) -> Self::Ret {
        fn is_outer(x: &&Attr) -> bool {
            use Style::*;
            match x.style {
                Outer => true,
                Inner(_) => false,
            }
        }
        self.iter().filter(is_outer)
    }
}

pub fn has_inner(xs: &[Attr]) -> bool {
    for x in xs {
        if let Style::Inner(_) = x.style {
            return true;
        }
    }
    false
}
pub fn has_outer(xs: &[Attr]) -> bool {
    for x in xs {
        if let Style::Outer = x.style {
            return true;
        }
    }
    false
}

pub fn parse_inners(s: Stream, ys: &mut Vec<Attr>) -> Res<()> {
    while s.peek(Token![#]) && s.peek2(Token![!]) {
        ys.push(s.call(parse_one_inner)?);
    }
    Ok(())
}
pub fn parse_one_inner(s: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: s.parse()?,
        style: Style::Inner(s.parse()?),
        bracket: bracketed!(y in s),
        meta: y.parse()?,
    })
}
pub fn parse_one_outer(s: Stream) -> Res<Attr> {
    let y;
    Ok(Attr {
        pound: s.parse()?,
        style: Style::Outer,
        bracket: bracketed!(y in s),
        meta: y.parse()?,
    })
}

pub fn lower_inners(xs: &[Attr], s: &mut Stream) {
    s.append_all(xs.inners());
}
pub fn lower_outers(xs: &[Attr], s: &mut Stream) {
    s.append_all(xs.outers());
}

pub struct DisplayStyle<'a>(pub &'a Style);
impl<'a> Display for DisplayStyle<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self.0 {
            Style::Outer => "#",
            Style::Inner(_) => "#!",
        })
    }
}

ast_enum_of_structs! {
    pub enum Meta {
        List(List),
        NameValue(NameValue),
        Path(Path),
    }
}
impl Meta {
    pub fn path(&self) -> &Path {
        use Meta::*;
        match self {
            List(x) => &x.path,
            NameValue(x) => &x.name,
            Path(x) => x,
        }
    }
    pub fn require_path_only(&self) -> Res<&Path> {
        use Meta::*;
        let y = match self {
            List(x) => x.delim.span().open(),
            NameValue(x) => x.eq.span,
            Path(x) => return Ok(x),
        };
        Err(Err::new(y, "unexpected token in attribute"))
    }
    pub fn require_list(&self) -> Res<&List> {
        use Meta::*;
        match self {
            List(x) => Ok(x),
            NameValue(x) => Err(Err::new(x.eq.span, "expected `(`")),
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!("expected args in parentheses: `{}(...)`", path::DisplayPath(x),),
            )),
        }
    }
    pub fn require_name_value(&self) -> Res<&NameValue> {
        use Meta::*;
        match self {
            List(x) => Err(Err::new(x.delim.span().open(), "expected `=`")),
            NameValue(x) => Ok(x),
            Path(x) => Err(err::new2(
                x.segs.first().unwrap().ident.span(),
                x.segs.last().unwrap().ident.span(),
                format!("expected value for attribute: `{} = ...`", path::DisplayPath(x),),
            )),
        }
    }
}
impl Parse for Meta {
    fn parse(s: Stream) -> Res<Self> {
        let y = s.call(Path::parse_mod_style)?;
        parse_after_path(y, s)
    }
}
impl Pretty for Meta {
    fn pretty(&self, p: &mut Print) {
        use Meta::*;
        match self {
            List(x) => x.pretty(p),
            NameValue(x) => x.pretty(p),
            Path(x) => p.path(x, path::Kind::Simple),
        }
    }
}

pub struct List {
    pub path: Path,
    pub delim: tok::Delim,
    pub toks: pm2::Stream,
}
impl List {
    pub fn parse_args<T: Parse>(&self) -> Res<T> {
        self.parse_args_with(T::parse)
    }
    pub fn parse_args_with<T: Parser>(&self, x: T) -> Res<T::Output> {
        let y = self.delim.span().close();
        parse::parse_scoped(x, y, self.toks.clone())
    }
    pub fn parse_nested(&self, x: impl FnMut(Nested) -> Res<()>) -> Res<()> {
        self.parse_args_with(parser(x))
    }
}
impl Parse for List {
    fn parse(s: Stream) -> Res<Self> {
        let y = s.call(Path::parse_mod_style)?;
        parse_list_after_path(y, s)
    }
}
impl Lower for List {
    fn lower(&self, s: &mut Stream) {
        self.path.lower(s);
        self.delim.surround(s, self.toks.clone());
    }
}
impl Pretty for List {
    fn pretty(&self, p: &mut Print) {
        &self.path.pretty(p, path::Kind::Simple);
        use tok::Delim::*;
        let delim = match self.delim {
            Brace(_) => pm2::Delim::Brace,
            Bracket(_) => pm2::Delim::Bracket,
            Paren(_) => pm2::Delim::Paren,
        };
        let y = pm2::Group::new(delim, self.toks.clone());
        attr_tokens(p, Stream::from(pm2::Tree::Group(y)));
        use pm2::{Delim, Tree};
        fn attr_tokens(p: &mut Print, tokens: Stream) {
            let mut stack = Vec::new();
            stack.push((tokens.into_iter().peekable(), Delim::None));
            let mut space = Print::nbsp as fn(&mut Print);
            #[derive(PartialEq)]
            enum State {
                Word,
                Punct,
                TrailingComma,
            }
            use State::*;
            let mut state = Word;
            while let Some((tokens, delim)) = stack.last_mut() {
                match tokens.next() {
                    Some(Tree::Ident(x)) => {
                        if let Word = state {
                            space(p);
                        }
                        p.ident(&x);
                        state = Word;
                    },
                    Some(Tree::Punct(x)) => {
                        let x = x.as_char();
                        if let (Word, '=') = (state, x) {
                            p.nbsp();
                        }
                        if x == ',' && tokens.peek().is_none() {
                            p.trailing_comma(true);
                            state = TrailingComma;
                        } else {
                            p.token_punct(x);
                            if x == '=' {
                                p.nbsp();
                            } else if x == ',' {
                                space(p);
                            }
                            state = Punct;
                        }
                    },
                    Some(Tree::Literal(x)) => {
                        if let Word = state {
                            space(p);
                        }
                        p.token_literal(&x);
                        state = Word;
                    },
                    Some(Tree::Group(x)) => {
                        let delim = x.delim();
                        let stream = x.stream();
                        use Delim::*;
                        match delim {
                            Paren => {
                                p.word("(");
                                p.cbox(INDENT);
                                p.zerobreak();
                                state = Punct;
                            },
                            Brace => {
                                p.word("{");
                                state = Punct;
                            },
                            Bracket => {
                                p.word("[");
                                state = Punct;
                            },
                            None => {},
                        }
                        stack.push((stream.into_iter().peekable(), delim));
                        space = Print::space;
                    },
                    None => {
                        use Delim::*;
                        match delim {
                            Paren => {
                                if state != TrailingComma {
                                    p.zerobreak();
                                }
                                p.offset(-INDENT);
                                p.end();
                                p.word(")");
                                state = Punct;
                            },
                            Brace => {
                                p.word("}");
                                state = Punct;
                            },
                            Bracket => {
                                p.word("]");
                                state = Punct;
                            },
                            None => {},
                        }
                        stack.pop();
                        if stack.is_empty() {
                            space = Print::nbsp;
                        }
                    },
                }
            }
        }
    }
}

pub struct NameValue {
    pub name: Path,
    pub eq: Token![=],
    pub val: expr::Expr,
}
impl Parse for NameValue {
    fn parse(s: Stream) -> Res<Self> {
        let y = s.call(Path::parse_mod_style)?;
        parse_name_value_after_path(y, s)
    }
}
impl Lower for NameValue {
    fn lower(&self, s: &mut Stream) {
        self.name.lower(s);
        self.eq.lower(s);
        self.val.lower(s);
    }
}
impl Pretty for NameValue {
    fn pretty(&self, p: &mut Print) {
        p.path(&self.name, path::Kind::Simple);
        p.word(" = ");
        p.expr(&self.val);
    }
}

pub struct Nested<'a> {
    pub path: Path,
    pub ins: Stream<'a>,
}
impl<'a> Nested<'a> {
    pub fn val(&self) -> Res<Stream<'a>> {
        self.ins.parse::<Token![=]>()?;
        Ok(self.ins)
    }
    pub fn parse(&self, f: impl FnMut(Nested) -> Res<()>) -> Res<()> {
        let y;
        parenthesized!(y in self.ins);
        parse_nested(&y, f)
    }
    pub fn err(&self, x: impl Display) -> Err {
        let beg = self.path.segs[0].ident.span();
        let end = self.ins.cursor().prev_span();
        err::new2(beg, end, x)
    }
}

pub fn parser(f: impl FnMut(Nested) -> Res<()>) -> impl Parser<Output = ()> {
    |x: Stream| {
        if x.is_empty() {
            Ok(())
        } else {
            parse_nested(x, f)
        }
    }
}

pub fn parse_nested(ins: Stream, mut f: impl FnMut(Nested) -> Res<()>) -> Res<()> {
    loop {
        let path = ins.call(parse_path)?;
        f(Nested { path, ins })?;
        if ins.is_empty() {
            return Ok(());
        }
        ins.parse::<Token![,]>()?;
        if ins.is_empty() {
            return Ok(());
        }
    }
}
pub fn parse_after_path(p: Path, s: Stream) -> Res<Meta> {
    if s.peek(tok::Paren) || s.peek(tok::Bracket) || s.peek(tok::Brace) {
        parse_list_after_path(p, s).map(Meta::List)
    } else if s.peek(Token![=]) {
        parse_name_value_after_path(p, s).map(Meta::NameValue)
    } else {
        Ok(Meta::Path(p))
    }
}
pub fn parse_list_after_path(path: Path, s: Stream) -> Res<List> {
    let (delim, toks) = mac::parse_delim(s)?;
    Ok(List { path, delim, toks })
}
pub fn parse_name_value_after_path(path: Path, s: Stream) -> Res<NameValue> {
    let eq: Token![=] = s.parse()?;
    let ahead = s.fork();
    let lit: Option<lit::Lit> = ahead.parse()?;
    let expr = if let (Some(lit), true) = (lit, ahead.is_empty()) {
        s.advance_to(&ahead);
        expr::Expr::Lit(expr::Lit { attrs: Vec::new(), lit })
    } else if s.peek(Token![#]) && s.peek2(tok::Bracket) {
        return Err(s.error("unexpected attribute inside of attribute"));
    } else {
        s.parse()?
    };
    Ok(NameValue {
        name: path,
        eq,
        val: expr,
    })
}

fn parse_path(s: Stream) -> Res<Path> {
    Ok(Path {
        colon: s.parse()?,
        segs: {
            let mut ys = Puncted::new();
            if s.peek(Ident::peek_any) {
                let y = Ident::parse_any(s)?;
                ys.push_value(path::Segment::from(y));
            } else if s.is_empty() {
                return Err(s.error("expected nested attribute"));
            } else if s.peek(lit::Lit) {
                return Err(s.error("unexpected literal in nested attribute, expected ident"));
            } else {
                return Err(s.error("unexpected token in nested attribute, expected ident"));
            }
            while s.peek(Token![::]) {
                let y = s.parse()?;
                ys.push_punct(y);
                let y = Ident::parse_any(s)?;
                ys.push_value(path::Segment::from(y));
            }
            ys
        },
    })
}
