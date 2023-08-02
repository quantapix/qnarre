use super::pm2::{Delim, Group, Lit, Spacing, Stream, Tree};
use super::{expr::Expr, *};
use std::{
    borrow::Cow,
    cmp,
    collections::VecDeque,
    iter,
    iter::Peekable,
    ops::{Deref, Index, IndexMut},
};

pub enum Args {
    Kind(path::Kind),
    BegLine(bool),
    IdentSemi(&Ident, bool),
}
impl Args {
    pub fn beg_line(x: &Option<Args>) {
        match x {
            Some(x) => match x {
                Args::BegLine(x) => x,
                _ => &false,
            },
            _ => &false,
        }
    }
    pub fn ident_semi(x: &Option<Args>) {
        match x {
            Some(x) => match x {
                Args::IdentSemi(x, semi) => Some((x, semi)),
                _ => None,
            },
            _ => None,
        }
    }
}

pub trait Pretty {
    fn pretty(&self, p: &mut Print) {}
    fn pretty_with_args(&self, p: &mut Print, x: &Option<Args>) {}
}

pub struct Print {
    buf: Buffer<Entry>,
    frames: Vec<Frame>,
    indent: usize,
    left: isize,
    out: String,
    pending: usize,
    right: isize,
    scans: VecDeque<usize>,
    space: isize,
}
impl Print {
    pub fn new() -> Self {
        Print {
            buf: Buffer::new(),
            frames: Vec::new(),
            indent: 0,
            left: 0,
            out: String::new(),
            pending: 0,
            right: 0,
            scans: VecDeque::new(),
            space: MARGIN,
        }
    }
    pub fn eof(mut self) -> String {
        if !self.scans.is_empty() {
            self.check_stack(0);
            self.advance_left();
        }
        self.out
    }
    pub fn scan_begin(&mut self, token: Begin) {
        if self.scans.is_empty() {
            self.left = 1;
            self.right = 1;
            self.buf.clear();
        }
        let right = self.buf.push(Entry {
            tok: Token::Begin(token),
            size: -self.right,
        });
        self.scans.push_back(right);
    }
    pub fn scan_end(&mut self) {
        if self.scans.is_empty() {
            self.print_end();
        } else {
            if !self.buf.is_empty() {
                if let Token::Break(break_token) = self.buf.last().tok {
                    if self.buf.len() >= 2 {
                        if let Token::Begin(_) = self.buf.second_last().tok {
                            self.buf.pop_last();
                            self.buf.pop_last();
                            self.scans.pop_back();
                            self.scans.pop_back();
                            self.right -= break_token.blank_space as isize;
                            return;
                        }
                    }
                    if break_token.if_nonempty {
                        self.buf.pop_last();
                        self.scans.pop_back();
                        self.right -= break_token.blank_space as isize;
                    }
                }
            }
            let right = self.buf.push(Entry {
                tok: Token::End,
                size: -1,
            });
            self.scans.push_back(right);
        }
    }
    pub fn scan_break(&mut self, token: Break) {
        if self.scans.is_empty() {
            self.left = 1;
            self.right = 1;
            self.buf.clear();
        } else {
            self.check_stack(0);
        }
        let right = self.buf.push(Entry {
            tok: Token::Break(token),
            size: -self.right,
        });
        self.scans.push_back(right);
        self.right += token.blank_space as isize;
    }
    pub fn scan_string(&mut self, string: Cow<'static, str>) {
        if self.scans.is_empty() {
            self.print_string(string);
        } else {
            let len = string.len() as isize;
            self.buf.push(Entry {
                tok: Token::String(string),
                size: len,
            });
            self.right += len;
            self.check_stream();
        }
    }
    pub fn offset(&mut self, offset: isize) {
        match &mut self.buf.last_mut().tok {
            Token::Break(token) => token.offset += offset,
            Token::Begin(_) => {},
            Token::String(_) | Token::End => unreachable!(),
        }
    }
    pub fn end_with_max_width(&mut self, max: isize) {
        let mut depth = 1;
        for &index in self.scans.iter().rev() {
            let entry = &self.buf[index];
            match entry.tok {
                Token::Begin(_) => {
                    depth -= 1;
                    if depth == 0 {
                        if entry.size < 0 {
                            let actual_width = entry.size + self.right;
                            if actual_width > max {
                                self.buf.push(Entry {
                                    tok: Token::String(Cow::Borrowed("")),
                                    size: SIZE_INFINITY,
                                });
                                self.right += SIZE_INFINITY;
                            }
                        }
                        break;
                    }
                },
                Token::End => depth += 1,
                Token::Break(_) => {},
                Token::String(_) => unreachable!(),
            }
        }
        self.scan_end();
    }
    fn check_stream(&mut self) {
        while self.right - self.left > self.space {
            if *self.scans.front().unwrap() == self.buf.idx_of_first() {
                self.scans.pop_front().unwrap();
                self.buf.first_mut().size = SIZE_INFINITY;
            }
            self.advance_left();
            if self.buf.is_empty() {
                break;
            }
        }
    }
    fn advance_left(&mut self) {
        while self.buf.first().size >= 0 {
            let left = self.buf.pop_first();
            match left.tok {
                Token::String(string) => {
                    self.left += left.size;
                    self.print_string(string);
                },
                Token::Break(token) => {
                    self.left += token.blank_space as isize;
                    self.print_break(token, left.size);
                },
                Token::Begin(token) => self.print_begin(token, left.size),
                Token::End => self.print_end(),
            }
            if self.buf.is_empty() {
                break;
            }
        }
    }
    fn check_stack(&mut self, mut depth: usize) {
        while let Some(&i) = self.scans.back() {
            let entry = &mut self.buf[i];
            match entry.tok {
                Token::Begin(_) => {
                    if depth == 0 {
                        break;
                    }
                    self.scans.pop_back().unwrap();
                    entry.size += self.right;
                    depth -= 1;
                },
                Token::End => {
                    self.scans.pop_back().unwrap();
                    entry.size = 1;
                    depth += 1;
                },
                Token::Break(_) => {
                    self.scans.pop_back().unwrap();
                    entry.size += self.right;
                    if depth == 0 {
                        break;
                    }
                },
                Token::String(_) => unreachable!(),
            }
        }
    }
    fn get_top(&self) -> Frame {
        const OUTER: Frame = Frame::Broken(0, Breaks::Inconsistent);
        self.frames.last().map_or(OUTER, Frame::clone)
    }
    fn print_begin(&mut self, token: Begin, size: isize) {
        if cfg!(prettyplease_debug) {
            self.out.push(match token.breaks {
                Breaks::Consistent => '«',
                Breaks::Inconsistent => '‹',
            });
            if cfg!(prettyplease_debug_indent) {
                self.out.extend(token.off.to_string().chars().map(|ch| match ch {
                    '0'..='9' => ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'][(ch as u8 - b'0') as usize],
                    '-' => '₋',
                    _ => unreachable!(),
                }));
            }
        }
        if size > self.space {
            self.frames.push(Frame::Broken(self.indent, token.breaks));
            self.indent = usize::try_from(self.indent as isize + token.off).unwrap();
        } else {
            self.frames.push(Frame::Fits(token.breaks));
        }
    }
    fn print_end(&mut self) {
        let breaks = match self.frames.pop().unwrap() {
            Frame::Broken(indent, breaks) => {
                self.indent = indent;
                breaks
            },
            Frame::Fits(breaks) => breaks,
        };
        if cfg!(prettyplease_debug) {
            self.out.push(match breaks {
                Breaks::Consistent => '»',
                Breaks::Inconsistent => '›',
            });
        }
    }
    fn print_break(&mut self, token: Break, size: isize) {
        let fits = token.never
            || match self.get_top() {
                Frame::Fits(..) => true,
                Frame::Broken(.., Breaks::Consistent) => false,
                Frame::Broken(.., Breaks::Inconsistent) => size <= self.space,
            };
        if fits {
            self.pending += token.blank_space;
            self.space -= token.blank_space as isize;
            if let Some(no_break) = token.no_break {
                self.out.push(no_break);
                self.space -= no_break.len_utf8() as isize;
            }
            if cfg!(prettyplease_debug) {
                self.out.push('·');
            }
        } else {
            if let Some(pre_break) = token.pre {
                self.print_indent();
                self.out.push(pre_break);
            }
            if cfg!(prettyplease_debug) {
                self.out.push('·');
            }
            self.out.push('\n');
            let indent = self.indent as isize + token.off;
            self.pending = usize::try_from(indent).unwrap();
            self.space = cmp::max(MARGIN - indent, MIN_SPACE);
            if let Some(post_break) = token.post {
                self.print_indent();
                self.out.push(post_break);
                self.space -= post_break.len_utf8() as isize;
            }
        }
    }
    fn print_string(&mut self, string: Cow<'static, str>) {
        self.print_indent();
        self.out.push_str(&string);
        self.space -= string.len() as isize;
    }
    fn print_indent(&mut self) {
        self.out.reserve(self.pending);
        self.out.extend(iter::repeat(' ').take(self.pending));
        self.pending = 0;
    }
    pub fn ibox(&mut self, indent: isize) {
        self.scan_begin(Begin {
            off: indent,
            breaks: Breaks::Inconsistent,
        });
    }
    pub fn cbox(&mut self, indent: isize) {
        self.scan_begin(Begin {
            off: indent,
            breaks: Breaks::Consistent,
        });
    }
    pub fn end(&mut self) {
        self.scan_end();
    }
    pub fn word<S: Into<Cow<'static, str>>>(&mut self, wrd: S) {
        let s = wrd.into();
        self.scan_string(s);
    }
    fn spaces(&mut self, n: usize) {
        self.scan_break(Break {
            blank_space: n,
            ..Break::default()
        });
    }
    pub fn zerobreak(&mut self) {
        self.spaces(0);
    }
    pub fn space(&mut self) {
        self.spaces(1);
    }
    pub fn nbsp(&mut self) {
        self.word(" ");
    }
    pub fn hardbreak(&mut self) {
        self.spaces(SIZE_INFINITY as usize);
    }
    pub fn space_if_nonempty(&mut self) {
        self.scan_break(Break {
            blank_space: 1,
            if_nonempty: true,
            ..Break::default()
        });
    }
    pub fn hardbreak_if_nonempty(&mut self) {
        self.scan_break(Break {
            blank_space: SIZE_INFINITY as usize,
            if_nonempty: true,
            ..Break::default()
        });
    }
    pub fn trailing_comma(&mut self, is_last: bool) {
        if is_last {
            self.scan_break(Break {
                pre: Some(','),
                ..Break::default()
            });
        } else {
            self.word(",");
            self.space();
        }
    }
    pub fn trailing_comma_or_space(&mut self, is_last: bool) {
        if is_last {
            self.scan_break(Break {
                blank_space: 1,
                pre: Some(','),
                ..Break::default()
            });
        } else {
            self.word(",");
            self.space();
        }
    }
    pub fn neverbreak(&mut self) {
        self.scan_break(Break {
            never: true,
            ..Break::default()
        });
    }
    //expr
    pub fn call_args(&mut self, xs: &punct::Puncted<Expr, Token![,]>) {
        let mut iter = xs.iter();
        match (iter.next(), iter.next()) {
            (Some(x), None) if x.is_blocklike() => {
                x.pretty(self);
            },
            _ => {
                self.cbox(INDENT);
                self.zerobreak();
                for x in xs.iter().delimited() {
                    &x.pretty(self);
                    self.trailing_comma(x.is_last);
                }
                self.offset(-INDENT);
                self.end();
            },
        }
    }
    pub fn small_block(&mut self, block: &stmt::Block, attrs: &[attr::Attr]) {
        self.word("{");
        if attr::has_inner(attrs) || !block.stmts.is_empty() {
            self.space();
            self.inner_attrs(attrs);
            match (block.stmts.get(0), block.stmts.get(1)) {
                (Some(stmt::Stmt::Expr(expr, None)), None) if expr.break_after() => {
                    self.ibox(0);
                    expr.pretty_beg_line(self, true);
                    self.end();
                    self.space();
                },
                _ => {
                    for stmt in &block.stmts {
                        self.stmt(stmt);
                    }
                },
            }
            self.offset(-INDENT);
        }
        self.word("}");
    }
    //mac
    pub fn macro_rules(&mut self, name: &Ident, rules: &Stream) {
        enum State {
            Start,
            Matcher,
            Equal,
            Greater,
            Expander,
        }
        use State::*;
        self.word("macro_rules! ");
        self.ident(name);
        self.word(" {");
        self.cbox(INDENT);
        self.hardbreak_if_nonempty();
        let mut state = State::Start;
        for tt in rules.clone() {
            let token = Token::from(tt);
            match (state, token) {
                (Start, Token::Group(delim, stream)) => {
                    delim.pretty_open(self);
                    if !stream.is_empty() {
                        self.cbox(INDENT);
                        self.zerobreak();
                        self.ibox(0);
                        self.macro_rules_tokens(stream, true);
                        self.end();
                        self.zerobreak();
                        self.offset(-INDENT);
                        self.end();
                    }
                    delim.pretty_close(self);
                    state = Matcher;
                },
                (Matcher, Token::Punct('=', Spacing::Joint)) => {
                    self.word(" =");
                    state = Equal;
                },
                (Equal, Token::Punct('>', Spacing::Alone)) => {
                    self.word(">");
                    state = Greater;
                },
                (Greater, Token::Group(_delim, stream)) => {
                    self.word(" {");
                    self.neverbreak();
                    if !stream.is_empty() {
                        self.cbox(INDENT);
                        self.hardbreak();
                        self.ibox(0);
                        self.macro_rules_tokens(stream, false);
                        self.end();
                        self.hardbreak();
                        self.offset(-INDENT);
                        self.end();
                    }
                    self.word("}");
                    state = Expander;
                },
                (Expander, Token::Punct(';', Spacing::Alone)) => {
                    self.word(";");
                    self.hardbreak();
                    state = Start;
                },
                _ => unimplemented!("bad macro_rules syntax"),
            }
        }
        match state {
            Start => {},
            Expander => {
                self.word(";");
                self.hardbreak();
            },
            _ => self.hardbreak(),
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
    }
    pub fn macro_rules_tokens(&mut self, s: Stream, matcher: bool) {
        #[derive(PartialEq)]
        enum State {
            Start,
            Dollar,
            DollarIdent,
            DollarIdentColon,
            DollarParen,
            DollarParenSep,
            Pound,
            PoundBang,
            Dot,
            Colon,
            Colon2,
            Ident,
            IdentBang,
            Delim,
            Other,
        }
        use State::*;
        let mut state = Start;
        let mut previous_is_joint = true;
        for tt in s {
            let token = Token::from(tt);
            let (needs_space, next_state) = match (&state, &token) {
                (Dollar, Token::Ident(_)) => (false, if matcher { DollarIdent } else { Other }),
                (DollarIdent, Token::Punct(':', Spacing::Alone)) => (false, DollarIdentColon),
                (DollarIdentColon, Token::Ident(_)) => (false, Other),
                (DollarParen, Token::Punct('+' | '*' | '?', Spacing::Alone)) => (false, Other),
                (DollarParen, Token::Ident(_) | Token::Lit(_)) => (false, DollarParenSep),
                (DollarParen, Token::Punct(_, Spacing::Joint)) => (false, DollarParen),
                (DollarParen, Token::Punct(_, Spacing::Alone)) => (false, DollarParenSep),
                (DollarParenSep, Token::Punct('+' | '*', _)) => (false, Other),
                (Pound, Token::Punct('!', _)) => (false, PoundBang),
                (Dollar, Token::Group(Delim::Paren, _)) => (false, DollarParen),
                (Pound | PoundBang, Token::Group(Delim::Bracket, _)) => (false, Other),
                (Ident, Token::Group(Delim::Paren | Delim::Bracket, _)) => (false, Delim),
                (Ident, Token::Punct('!', Spacing::Alone)) => (false, IdentBang),
                (IdentBang, Token::Group(Delim::Parent | Delim::Bracket, _)) => (false, Other),
                (Colon, Token::Punct(':', _)) => (false, Colon2),
                (_, Token::Group(Delim::Parent | Delim::Bracket, _)) => (true, Delim),
                (_, Token::Group(Delim::Brace | Delim::None, _)) => (true, Other),
                (_, Token::Ident(ident)) if !is_keyword(ident) => (state != Dot && state != Colon2, Ident),
                (_, Token::Lit(_)) => (state != Dot, Ident),
                (_, Token::Punct(',' | ';', _)) => (false, Other),
                (_, Token::Punct('.', _)) if !matcher => (state != Ident && state != Delim, Dot),
                (_, Token::Punct(':', Spacing::Joint)) => (state != Ident, Colon),
                (_, Token::Punct('$', _)) => (true, Dollar),
                (_, Token::Punct('#', _)) => (true, Pound),
                (_, _) => (true, Other),
            };
            if !previous_is_joint {
                if needs_space {
                    self.space();
                } else if let Token::Punct('.', _) = token {
                    self.zerobreak();
                }
            }
            previous_is_joint = match token {
                Token::Punct(_, Spacing::Joint) | Token::Punct('$', _) => true,
                _ => false,
            };
            self.single_token(
                token,
                if matcher {
                    |printer, stream| printer.macro_rules_tokens(stream, true)
                } else {
                    |printer, stream| printer.macro_rules_tokens(stream, false)
                },
            );
            state = next_state;
        }
    }
    //token
    pub fn single_token(&mut self, token: Token, group_contents: fn(&mut Self, Stream)) {
        match token {
            Token::Group(delim, stream) => self.token_group(delim, stream, group_contents),
            Token::Ident(ident) => self.ident(&ident),
            Token::Punct(ch, _spacing) => self.token_punct(ch),
            Token::Lit(literal) => self.token_literal(&literal),
        }
    }
    fn token_group(&mut self, delim: Delim, stream: Stream, group_contents: fn(&mut Self, Stream)) {
        delim.pretty_open(self);
        if !stream.is_empty() {
            if delim == Delim::Brace {
                self.space();
            }
            group_contents(self, stream);
            if delim == Delim::Brace {
                self.space();
            }
        }
        delim.pretty_close(self);
    }
    pub fn ident(&mut self, ident: &Ident) {
        self.word(ident.to_string());
    }
    pub fn token_punct(&mut self, ch: char) {
        self.word(ch.to_string());
    }
    pub fn token_literal(&mut self, literal: &Lit) {
        self.word(literal.to_string());
    }
}

struct Buffer<T> {
    data: VecDeque<T>,
    off: usize,
}
impl<T> Buffer<T> {
    fn new() -> Self {
        Buffer {
            data: VecDeque::new(),
            off: 0,
        }
    }
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn push(&mut self, x: T) -> usize {
        let y = self.off + self.data.len();
        self.data.push_back(x);
        y
    }
    fn clear(&mut self) {
        self.data.clear();
    }
    fn idx_of_first(&self) -> usize {
        self.off
    }
    fn first(&self) -> &T {
        &self.data[0]
    }
    fn first_mut(&mut self) -> &mut T {
        &mut self.data[0]
    }
    fn pop_first(&mut self) -> T {
        self.off += 1;
        self.data.pop_front().unwrap()
    }
    fn last(&self) -> &T {
        self.data.back().unwrap()
    }
    fn last_mut(&mut self) -> &mut T {
        self.data.back_mut().unwrap()
    }
    fn pop_last(&mut self) {
        self.data.pop_back().unwrap();
    }
    fn second_last(&self) -> &T {
        &self.data[self.data.len() - 2]
    }
}
impl<T> Index<usize> for Buffer<T> {
    type Output = T;
    fn index(&self, x: usize) -> &Self::Output {
        &self.data[x.checked_sub(self.off).unwrap()]
    }
}
impl<T> IndexMut<usize> for Buffer<T> {
    fn index_mut(&mut self, x: usize) -> &mut Self::Output {
        &mut self.data[x.checked_sub(self.off).unwrap()]
    }
}

#[derive(Clone)]
struct Entry {
    tok: Token,
    size: isize,
}

#[derive(Copy, Clone)]
enum Frame {
    Fits(Breaks),
    Broken(usize, Breaks),
}

#[derive(Clone, Copy, PartialEq)]
enum Breaks {
    Consistent,
    Inconsistent,
}

#[derive(Clone)]
pub enum Token {
    String(Cow<'static, str>),
    Break(Break),
    Begin(Begin),
    End,
}

#[derive(Clone, Copy, Default)]
pub struct Break {
    pub off: isize,
    pub blank_space: usize,
    pub pre: Option<char>,
    pub post: Option<char>,
    pub no_break: Option<char>,
    pub if_nonempty: bool,
    pub never: bool,
}

#[derive(Clone, Copy)]
pub struct Begin {
    pub off: isize,
    pub breaks: Breaks,
}

pub const SIZE_INFINITY: isize = 0xffff;

pub struct Delimited<I: Iterator> {
    is_first: bool,
    iter: Peekable<I>,
}
impl<I: Iterator> Iterator for Delimited<I> {
    type Item = IteratorItem<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let item = IteratorItem {
            value: self.iter.next()?,
            is_first: self.is_first,
            is_last: self.iter.peek().is_none(),
        };
        self.is_first = false;
        Some(item)
    }
}

pub trait IterDelimited: Iterator + Sized {
    fn delimited(self) -> Delimited<Self> {
        Delimited {
            is_first: true,
            iter: self.peekable(),
        }
    }
}
impl<I: Iterator> IterDelimited for I {}

pub struct IteratorItem<T> {
    value: T,
    pub is_first: bool,
    pub is_last: bool,
}
impl<T> Deref for IteratorItem<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

fn is_keyword(x: &Ident) -> bool {
    match x.to_string().as_str() {
        "as" | "async" | "await" | "box" | "break" | "const" | "continue" | "crate" | "dyn" | "else" | "enum"
        | "extern" | "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move"
        | "mut" | "pub" | "ref" | "return" | "static" | "struct" | "trait" | "type" | "unsafe" | "use" | "where"
        | "while" | "yield" => true,
        _ => false,
    }
}
mod standard_library {
    use super::*;
    enum KnownMac {
        Expr(Expr),
        Exprs(Vec<Expr>),
        Cfg(Cfg),
        Matches(Matches),
        ThreadLocal(Vec<ThreadLocal>),
        VecArray(Vec<Expr>),
        VecRepeat { elem: Expr, n: Expr },
    }
    enum Cfg {
        Eq(Ident, Option<Lit>),
        Call(Ident, Vec<Cfg>),
    }
    struct Matches {
        expression: Expr,
        pattern: pat::Pat,
        guard: Option<Expr>,
    }
    struct ThreadLocal {
        attrs: Vec<attr::Attr>,
        vis: data::Visibility,
        name: Ident,
        typ: typ::Type,
        init: Expr,
    }
    struct FormatArgs {
        format_string: Expr,
        args: Vec<Expr>,
    }
    impl parse::Parse for FormatArgs {
        fn parse(s: parse::Stream) -> Res<Self> {
            let format_string: Expr = s.parse()?;
            let mut args = Vec::new();
            while !s.is_empty() {
                s.parse::<Token![,]>()?;
                if s.is_empty() {
                    break;
                }
                let arg = if s.peek(Ident::peek_any) && s.peek2(Token![=]) && !s.peek2(Token![==]) {
                    let key = s.call(Ident::parse_any)?;
                    let eq: Token![=] = s.parse()?;
                    let value: Expr = s.parse()?;
                    Expr::Assign(expr::Assign {
                        attrs: Vec::new(),
                        left: Box::new(Expr::Path(expr::Path {
                            attrs: Vec::new(),
                            qself: None,
                            path: Path::from(key),
                        })),
                        eq,
                        right: Box::new(value),
                    })
                } else {
                    s.parse()?
                };
                args.push(arg);
            }
            Ok(FormatArgs { format_string, args })
        }
    }
    impl KnownMac {
        fn parse_expr(s: parse::Stream) -> Res<Self> {
            let y: Expr = s.parse()?;
            Ok(KnownMac::Expr(y))
        }
        fn parse_expr_comma(s: parse::Stream) -> Res<Self> {
            let y: Expr = s.parse()?;
            s.parse::<Option<Token![,]>>()?;
            Ok(KnownMac::Exprs(vec![y]))
        }
        fn parse_exprs(s: parse::Stream) -> Res<Self> {
            let ys = s.parse_terminated(Expr::parse, Token![,])?;
            Ok(KnownMac::Exprs(Vec::from_iter(ys)))
        }
        fn parse_assert(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            let cond: Expr = s.parse()?;
            ys.push(cond);
            if s.parse::<Option<Token![,]>>()?.is_some() && !s.is_empty() {
                let y: FormatArgs = s.parse()?;
                ys.push(y.format_string);
                ys.extend(y.args);
            }
            Ok(KnownMac::Exprs(ys))
        }
        fn parse_assert_cmp(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            let left: Expr = s.parse()?;
            ys.push(left);
            s.parse::<Token![,]>()?;
            let right: Expr = s.parse()?;
            ys.push(right);
            if s.parse::<Option<Token![,]>>()?.is_some() && !s.is_empty() {
                let y: FormatArgs = s.parse()?;
                ys.push(y.format_string);
                ys.extend(y.args);
            }
            Ok(KnownMac::Exprs(ys))
        }
        fn parse_cfg(s: parse::Stream) -> Res<Self> {
            fn parse_single(s: parse::Stream) -> Res<Cfg> {
                let ident: Ident = s.parse()?;
                if s.peek(tok::Paren) && (ident == "all" || ident == "any") {
                    let y;
                    parenthesized!(y in s);
                    let list = y.call(parse_multiple)?;
                    Ok(Cfg::Call(ident, list))
                } else if s.peek(tok::Paren) && ident == "not" {
                    let y;
                    parenthesized!(y in s);
                    let cfg = y.call(parse_single)?;
                    y.parse::<Option<Token![,]>>()?;
                    Ok(Cfg::Call(ident, vec![cfg]))
                } else if s.peek(Token![=]) {
                    s.parse::<Token![=]>()?;
                    let y: lit::Lit = s.parse()?;
                    Ok(Cfg::Eq(ident, Some(y)))
                } else {
                    Ok(Cfg::Eq(ident, None))
                }
            }
            fn parse_multiple(s: parse::Stream) -> Res<Vec<Cfg>> {
                let mut ys = Vec::new();
                while !s.is_empty() {
                    let cfg = s.call(parse_single)?;
                    ys.push(cfg);
                    if s.is_empty() {
                        break;
                    }
                    s.parse::<Token![,]>()?;
                }
                Ok(ys)
            }
            let cfg = s.call(parse_single)?;
            s.parse::<Option<Token![,]>>()?;
            Ok(KnownMac::Cfg(cfg))
        }
        fn parse_env(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            let name: Expr = s.parse()?;
            ys.push(name);
            if s.parse::<Option<Token![,]>>()?.is_some() && !s.is_empty() {
                let y: Expr = s.parse()?;
                ys.push(y);
                s.parse::<Option<Token![,]>>()?;
            }
            Ok(KnownMac::Exprs(ys))
        }
        fn parse_format_args(s: parse::Stream) -> Res<Self> {
            let y: FormatArgs = s.parse()?;
            let mut ys = y.args;
            ys.insert(0, y.format_string);
            Ok(KnownMac::Exprs(ys))
        }
        fn parse_matches(s: parse::Stream) -> Res<Self> {
            let expression: Expr = s.parse()?;
            s.parse::<Token![,]>()?;
            let pattern = s.call(pat::Pat::parse_multi_with_leading_vert)?;
            let guard = if s.parse::<Option<Token![if]>>()?.is_some() {
                Some(s.parse()?)
            } else {
                None
            };
            s.parse::<Option<Token![,]>>()?;
            Ok(KnownMac::Matches(Matches {
                expression,
                pattern,
                guard,
            }))
        }
        fn parse_thread_local(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            while !s.is_empty() {
                let attrs = s.call(attr::Attr::parse_outer)?;
                let vis: data::Visibility = s.parse()?;
                s.parse::<Token![static]>()?;
                let name: Ident = s.parse()?;
                s.parse::<Token![:]>()?;
                let typ: typ::Type = s.parse()?;
                s.parse::<Token![=]>()?;
                let init: Expr = s.parse()?;
                if s.is_empty() {
                    break;
                }
                s.parse::<Token![;]>()?;
                ys.push(ThreadLocal {
                    attrs,
                    vis,
                    name,
                    typ,
                    init,
                });
            }
            Ok(KnownMac::ThreadLocal(ys))
        }
        fn parse_vec(s: parse::Stream) -> Res<Self> {
            if s.is_empty() {
                return Ok(KnownMac::VecArray(Vec::new()));
            }
            let first: Expr = s.parse()?;
            if s.parse::<Option<Token![;]>>()?.is_some() {
                let len: Expr = s.parse()?;
                Ok(KnownMac::VecRepeat { elem: first, n: len })
            } else {
                let mut ys = vec![first];
                while !s.is_empty() {
                    s.parse::<Token![,]>()?;
                    if s.is_empty() {
                        break;
                    }
                    let y: Expr = s.parse()?;
                    ys.push(y);
                }
                Ok(KnownMac::VecArray(ys))
            }
        }
        fn parse_write(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            let dst: Expr = s.parse()?;
            ys.push(dst);
            s.parse::<Token![,]>()?;
            let y: FormatArgs = s.parse()?;
            ys.push(y.format_string);
            ys.extend(y.args);
            Ok(KnownMac::Exprs(ys))
        }
        fn parse_writeln(s: parse::Stream) -> Res<Self> {
            let mut ys = Vec::new();
            let dst: Expr = s.parse()?;
            ys.push(dst);
            if s.parse::<Option<Token![,]>>()?.is_some() && !s.is_empty() {
                let y: FormatArgs = s.parse()?;
                ys.push(y.format_string);
                ys.extend(y.args);
            }
            Ok(KnownMac::Exprs(ys))
        }
    }
    impl Print {
        pub fn standard_library_macro(&mut self, mac: &mac::Mac, mut semi: bool) -> bool {
            let name = mac.path.segs.last().unwrap().ident.to_string();
            let parser = match name.as_str() {
                "addr_of" | "addr_of_mut" => KnownMac::parse_expr,
                "assert" | "debug_assert" => KnownMac::parse_assert,
                "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne" => KnownMac::parse_assert_cmp,
                "cfg" => KnownMac::parse_cfg,
                "compile_error" | "include" | "include_bytes" | "include_str" | "option_env" => {
                    KnownMac::parse_expr_comma
                },
                "concat" | "concat_bytes" | "dbg" => KnownMac::parse_exprs,
                "const_format_args" | "eprint" | "eprintln" | "format" | "format_args" | "format_args_nl" | "panic"
                | "print" | "println" | "todo" | "unimplemented" | "unreachable" => KnownMac::parse_format_args,
                "env" => KnownMac::parse_env,
                "matches" => KnownMac::parse_matches,
                "thread_local" => KnownMac::parse_thread_local,
                "vec" => KnownMac::parse_vec,
                "write" => KnownMac::parse_write,
                "writeln" => KnownMac::parse_writeln,
                _ => return false,
            };
            let known_macro = match parser.parse2(mac.toks.clone()) {
                Ok(known_macro) => known_macro,
                Err(_) => return false,
            };
            self.path(&mac.path, path::Kind::Simple);
            self.word("!");
            match &known_macro {
                KnownMac::Expr(expr) => {
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.expr(expr);
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                },
                KnownMac::Exprs(exprs) => {
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    for elem in exprs.iter().delimited() {
                        self.expr(&elem);
                        self.trailing_comma(elem.is_last);
                    }
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                },
                KnownMac::Cfg(cfg) => {
                    self.word("(");
                    self.cfg(cfg);
                    self.word(")");
                },
                KnownMac::Matches(matches) => {
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.expr(&matches.expression);
                    self.word(",");
                    self.space();
                    self.pat(&matches.pattern);
                    if let Some(guard) = &matches.guard {
                        self.space();
                        self.word("if ");
                        self.expr(guard);
                    }
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                },
                KnownMac::ThreadLocal(items) => {
                    self.word(" {");
                    self.cbox(INDENT);
                    self.hardbreak_if_nonempty();
                    for item in items {
                        self.outer_attrs(&item.attrs);
                        self.cbox(0);
                        &item.vis.pretty(self);
                        self.word("static ");
                        self.ident(&item.name);
                        self.word(": ");
                        self.ty(&item.typ);
                        self.word(" = ");
                        self.neverbreak();
                        self.expr(&item.init);
                        self.word(";");
                        self.end();
                        self.hardbreak();
                    }
                    self.offset(-INDENT);
                    self.end();
                    self.word("}");
                    semi = false;
                },
                KnownMac::VecArray(vec) => {
                    self.word("[");
                    self.cbox(INDENT);
                    self.zerobreak();
                    for elem in vec.iter().delimited() {
                        self.expr(&elem);
                        self.trailing_comma(elem.is_last);
                    }
                    self.offset(-INDENT);
                    self.end();
                    self.word("]");
                },
                KnownMac::VecRepeat { elem, n } => {
                    self.word("[");
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.expr(elem);
                    self.word(";");
                    self.space();
                    self.expr(n);
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                    self.word("]");
                },
            }
            if semi {
                self.word(";");
            }
            true
        }
        fn cfg(&mut self, cfg: &Cfg) {
            match cfg {
                Cfg::Eq(ident, value) => {
                    self.ident(ident);
                    if let Some(value) = value {
                        self.word(" = ");
                        self.lit(value);
                    }
                },
                Cfg::Call(ident, args) => {
                    self.ident(ident);
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    for arg in args.iter().delimited() {
                        self.cfg(&arg);
                        self.trailing_comma(arg.is_last);
                    }
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                },
            }
        }
    }
}

pub enum Token {
    Group(Delim, Stream),
    Ident(Ident),
    Lit(Lit),
    Punct(char, Spacing),
}
impl From<Tree> for Token {
    fn from(x: Tree) -> Self {
        match x {
            Tree::Group(x) => Token::Group(x.delim(), x.stream()),
            Tree::Ident(x) => Token::Ident(x),
            Tree::Punct(x) => Token::Punct(x.as_char(), x.spacing()),
            Tree::Lit(x) => Token::Lit(x),
        }
    }
}

pub fn unparse(file: &item::File) -> String {
    let mut p = Print::new();
    p.file(file);
    p.eof()
}
