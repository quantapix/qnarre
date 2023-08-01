use super::pm2::{Delim, Group, Ident, Literal, Spacing, Stream, Tree};
use super::{item::File, Expr, Ident, Index, Macro, Path, Stmt, Token, *};
use std::{
    borrow::Cow,
    cmp,
    collections::VecDeque,
    iter,
    iter::Peekable,
    ops::{Deref, Index, IndexMut},
    ptr,
};

pub trait Pretty {
    fn pretty(&self, p: &mut Print);
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
        while let Some(&index) = self.scans.back() {
            let entry = &mut self.buf[index];
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
        let fits = token.never_break
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
            if let Some(pre_break) = token.pre_break {
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
            if let Some(post_break) = token.post_break {
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
                pre_break: Some(','),
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
                pre_break: Some(','),
                ..Break::default()
            });
        } else {
            self.word(",");
            self.space();
        }
    }
    pub fn neverbreak(&mut self) {
        self.scan_break(Break {
            never_break: true,
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
                (Some(Stmt::Expr(expr, None)), None) if expr.break_after() => {
                    self.ibox(0);
                    expr.pretty_beg_of_line(self, true);
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
    fn macro_rules(&mut self, name: &Ident, rules: &Stream) {
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
    pub fn macro_rules_tokens(&mut self, stream: Stream, matcher: bool) {
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
        for tt in stream {
            let token = Token::from(tt);
            let (needs_space, next_state) = match (&state, &token) {
                (Dollar, Token::Ident(_)) => (false, if matcher { DollarIdent } else { Other }),
                (DollarIdent, Token::Punct(':', Spacing::Alone)) => (false, DollarIdentColon),
                (DollarIdentColon, Token::Ident(_)) => (false, Other),
                (DollarParen, Token::Punct('+' | '*' | '?', Spacing::Alone)) => (false, Other),
                (DollarParen, Token::Ident(_) | Token::Literal(_)) => (false, DollarParenSep),
                (DollarParen, Token::Punct(_, Spacing::Joint)) => (false, DollarParen),
                (DollarParen, Token::Punct(_, Spacing::Alone)) => (false, DollarParenSep),
                (DollarParenSep, Token::Punct('+' | '*', _)) => (false, Other),
                (Pound, Token::Punct('!', _)) => (false, PoundBang),
                (Dollar, Token::Group(Delim::Parenthesis, _)) => (false, DollarParen),
                (Pound | PoundBang, Token::Group(Delim::Bracket, _)) => (false, Other),
                (Ident, Token::Group(Delim::Parenthesis | Delim::Bracket, _)) => (false, Delim),
                (Ident, Token::Punct('!', Spacing::Alone)) => (false, IdentBang),
                (IdentBang, Token::Group(Delim::Parenthesis | Delim::Bracket, _)) => (false, Other),
                (Colon, Token::Punct(':', _)) => (false, Colon2),
                (_, Token::Group(Delim::Parenthesis | Delim::Bracket, _)) => (true, Delim),
                (_, Token::Group(Delim::Brace | Delim::None, _)) => (true, Other),
                (_, Token::Ident(ident)) if !is_keyword(ident) => (state != Dot && state != Colon2, Ident),
                (_, Token::Literal(_)) => (state != Dot, Ident),
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
            Token::Literal(literal) => self.token_literal(&literal),
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
    pub fn token_literal(&mut self, literal: &Literal) {
        self.word(literal.to_string());
    }
}

#[derive(Clone)]
struct Entry {
    tok: Token,
    size: isize,
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
    pub pre_break: Option<char>,
    pub post_break: Option<char>,
    pub no_break: Option<char>,
    pub if_nonempty: bool,
    pub never_break: bool,
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

fn is_keyword(ident: &Ident) -> bool {
    match ident.to_string().as_str() {
        "as" | "async" | "await" | "box" | "break" | "const" | "continue" | "crate" | "dyn" | "else" | "enum"
        | "extern" | "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move"
        | "mut" | "pub" | "ref" | "return" | "static" | "struct" | "trait" | "type" | "unsafe" | "use" | "where"
        | "while" | "yield" => true,
        _ => false,
    }
}
mod standard_library {
    use super::*;
    enum KnownMacro {
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
        ty: Type,
        init: Expr,
    }
    struct FormatArgs {
        format_string: Expr,
        args: Vec<Expr>,
    }
    impl parse::Parse for FormatArgs {
        fn parse(input: parse::Stream) -> Res<Self> {
            let format_string: Expr = input.parse()?;
            let mut args = Vec::new();
            while !input.is_empty() {
                input.parse::<Token![,]>()?;
                if input.is_empty() {
                    break;
                }
                let arg = if input.peek(Ident::peek_any) && input.peek2(Token![=]) && !input.peek2(Token![==]) {
                    let key = input.call(Ident::parse_any)?;
                    let eq: Token![=] = input.parse()?;
                    let value: Expr = input.parse()?;
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
                    input.parse()?
                };
                args.push(arg);
            }
            Ok(FormatArgs { format_string, args })
        }
    }
    impl KnownMacro {
        fn parse_expr(input: parse::Stream) -> Res<Self> {
            let expr: Expr = input.parse()?;
            Ok(KnownMacro::Expr(expr))
        }
        fn parse_expr_comma(input: parse::Stream) -> Res<Self> {
            let expr: Expr = input.parse()?;
            input.parse::<Option<Token![,]>>()?;
            Ok(KnownMacro::Exprs(vec![expr]))
        }
        fn parse_exprs(input: parse::Stream) -> Res<Self> {
            let exprs = input.parse_terminated(Expr::parse, Token![,])?;
            Ok(KnownMacro::Exprs(Vec::from_iter(exprs)))
        }
        fn parse_assert(input: parse::Stream) -> Res<Self> {
            let mut exprs = Vec::new();
            let cond: Expr = input.parse()?;
            exprs.push(cond);
            if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
                let format_args: FormatArgs = input.parse()?;
                exprs.push(format_args.format_string);
                exprs.extend(format_args.args);
            }
            Ok(KnownMacro::Exprs(exprs))
        }
        fn parse_assert_cmp(input: parse::Stream) -> Res<Self> {
            let mut exprs = Vec::new();
            let left: Expr = input.parse()?;
            exprs.push(left);
            input.parse::<Token![,]>()?;
            let right: Expr = input.parse()?;
            exprs.push(right);
            if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
                let format_args: FormatArgs = input.parse()?;
                exprs.push(format_args.format_string);
                exprs.extend(format_args.args);
            }
            Ok(KnownMacro::Exprs(exprs))
        }
        fn parse_cfg(input: parse::Stream) -> Res<Self> {
            fn parse_single(input: parse::Stream) -> Res<Cfg> {
                let ident: Ident = input.parse()?;
                if input.peek(tok::Paren) && (ident == "all" || ident == "any") {
                    let content;
                    parenthesized!(content in input);
                    let list = content.call(parse_multiple)?;
                    Ok(Cfg::Call(ident, list))
                } else if input.peek(tok::Paren) && ident == "not" {
                    let content;
                    parenthesized!(content in input);
                    let cfg = content.call(parse_single)?;
                    content.parse::<Option<Token![,]>>()?;
                    Ok(Cfg::Call(ident, vec![cfg]))
                } else if input.peek(Token![=]) {
                    input.parse::<Token![=]>()?;
                    let string: Lit = input.parse()?;
                    Ok(Cfg::Eq(ident, Some(string)))
                } else {
                    Ok(Cfg::Eq(ident, None))
                }
            }
            fn parse_multiple(input: parse::Stream) -> Res<Vec<Cfg>> {
                let mut vec = Vec::new();
                while !input.is_empty() {
                    let cfg = input.call(parse_single)?;
                    vec.push(cfg);
                    if input.is_empty() {
                        break;
                    }
                    input.parse::<Token![,]>()?;
                }
                Ok(vec)
            }
            let cfg = input.call(parse_single)?;
            input.parse::<Option<Token![,]>>()?;
            Ok(KnownMacro::Cfg(cfg))
        }
        fn parse_env(input: parse::Stream) -> Res<Self> {
            let mut exprs = Vec::new();
            let name: Expr = input.parse()?;
            exprs.push(name);
            if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
                let error_msg: Expr = input.parse()?;
                exprs.push(error_msg);
                input.parse::<Option<Token![,]>>()?;
            }
            Ok(KnownMacro::Exprs(exprs))
        }
        fn parse_format_args(input: parse::Stream) -> Res<Self> {
            let format_args: FormatArgs = input.parse()?;
            let mut exprs = format_args.args;
            exprs.insert(0, format_args.format_string);
            Ok(KnownMacro::Exprs(exprs))
        }
        fn parse_matches(input: parse::Stream) -> Res<Self> {
            let expression: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pattern = input.call(pat::Pat::parse_multi_with_leading_vert)?;
            let guard = if input.parse::<Option<Token![if]>>()?.is_some() {
                Some(input.parse()?)
            } else {
                None
            };
            input.parse::<Option<Token![,]>>()?;
            Ok(KnownMacro::Matches(Matches {
                expression,
                pattern,
                guard,
            }))
        }
        fn parse_thread_local(input: parse::Stream) -> Res<Self> {
            let mut items = Vec::new();
            while !input.is_empty() {
                let attrs = input.call(attr::Attr::parse_outer)?;
                let vis: data::Visibility = input.parse()?;
                input.parse::<Token![static]>()?;
                let name: Ident = input.parse()?;
                input.parse::<Token![:]>()?;
                let ty: Type = input.parse()?;
                input.parse::<Token![=]>()?;
                let init: Expr = input.parse()?;
                if input.is_empty() {
                    break;
                }
                input.parse::<Token![;]>()?;
                items.push(ThreadLocal {
                    attrs,
                    vis,
                    name,
                    ty,
                    init,
                });
            }
            Ok(KnownMacro::ThreadLocal(items))
        }
        fn parse_vec(input: parse::Stream) -> Res<Self> {
            if input.is_empty() {
                return Ok(KnownMacro::VecArray(Vec::new()));
            }
            let first: Expr = input.parse()?;
            if input.parse::<Option<Token![;]>>()?.is_some() {
                let len: Expr = input.parse()?;
                Ok(KnownMacro::VecRepeat { elem: first, n: len })
            } else {
                let mut vec = vec![first];
                while !input.is_empty() {
                    input.parse::<Token![,]>()?;
                    if input.is_empty() {
                        break;
                    }
                    let next: Expr = input.parse()?;
                    vec.push(next);
                }
                Ok(KnownMacro::VecArray(vec))
            }
        }
        fn parse_write(input: parse::Stream) -> Res<Self> {
            let mut exprs = Vec::new();
            let dst: Expr = input.parse()?;
            exprs.push(dst);
            input.parse::<Token![,]>()?;
            let format_args: FormatArgs = input.parse()?;
            exprs.push(format_args.format_string);
            exprs.extend(format_args.args);
            Ok(KnownMacro::Exprs(exprs))
        }
        fn parse_writeln(input: parse::Stream) -> Res<Self> {
            let mut exprs = Vec::new();
            let dst: Expr = input.parse()?;
            exprs.push(dst);
            if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
                let format_args: FormatArgs = input.parse()?;
                exprs.push(format_args.format_string);
                exprs.extend(format_args.args);
            }
            Ok(KnownMacro::Exprs(exprs))
        }
    }
    impl Print {
        pub fn standard_library_macro(&mut self, mac: &Macro, mut semicolon: bool) -> bool {
            let name = mac.path.segments.last().unwrap().ident.to_string();
            let parser = match name.as_str() {
                "addr_of" | "addr_of_mut" => KnownMacro::parse_expr,
                "assert" | "debug_assert" => KnownMacro::parse_assert,
                "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne" => KnownMacro::parse_assert_cmp,
                "cfg" => KnownMacro::parse_cfg,
                "compile_error" | "include" | "include_bytes" | "include_str" | "option_env" => {
                    KnownMacro::parse_expr_comma
                },
                "concat" | "concat_bytes" | "dbg" => KnownMacro::parse_exprs,
                "const_format_args" | "eprint" | "eprintln" | "format" | "format_args" | "format_args_nl" | "panic"
                | "print" | "println" | "todo" | "unimplemented" | "unreachable" => KnownMacro::parse_format_args,
                "env" => KnownMacro::parse_env,
                "matches" => KnownMacro::parse_matches,
                "thread_local" => KnownMacro::parse_thread_local,
                "vec" => KnownMacro::parse_vec,
                "write" => KnownMacro::parse_write,
                "writeln" => KnownMacro::parse_writeln,
                _ => return false,
            };
            let known_macro = match parser.parse2(mac.tokens.clone()) {
                Ok(known_macro) => known_macro,
                Err(_) => return false,
            };
            self.path(&mac.path, path::Kind::Simple);
            self.word("!");
            match &known_macro {
                KnownMacro::Expr(expr) => {
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.expr(expr);
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                },
                KnownMacro::Exprs(exprs) => {
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
                KnownMacro::Cfg(cfg) => {
                    self.word("(");
                    self.cfg(cfg);
                    self.word(")");
                },
                KnownMacro::Matches(matches) => {
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
                KnownMacro::ThreadLocal(items) => {
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
                        self.ty(&item.ty);
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
                    semicolon = false;
                },
                KnownMacro::VecArray(vec) => {
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
                KnownMacro::VecRepeat { elem, n } => {
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
            if semicolon {
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
    Punct(char, Spacing),
    Literal(Literal),
}
impl From<Tree> for Token {
    fn from(tt: Tree) -> Self {
        match tt {
            Tree::Group(group) => Token::Group(group.delim(), group.stream()),
            Tree::Ident(ident) => Token::Ident(ident),
            Tree::Punct(punct) => Token::Punct(punct.as_char(), punct.spacing()),
            Tree::Literal(literal) => Token::Literal(literal),
        }
    }
}

pub fn unparse(file: &File) -> String {
    let mut p = Print::new();
    p.file(file);
    p.eof()
}
