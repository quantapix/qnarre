use super::pm2::{Delim, Group, Ident, Literal, Spacing, Stream, Tree};
use super::{
    braced, item::File, AngleBracketedGenericArguments, Arm, AssocConst, AssocType, BinOp, Block, Block,
    BoundLifetimes, ConstParam, Constraint, Expr, Field, FieldPat, FieldValue, Fields, FieldsUnnamed, FnArg,
    ForeignItem, ForeignItemFn, ForeignItemMacro, ForeignItemStatic, ForeignItemType, GenericArgument, GenericParam,
    Generics, Ident, IdentExt, ImplItem, ImplItemConst, ImplItemFn, ImplItemMacro, ImplItemType, Index, Item,
    ItemConst, ItemEnum, ItemExternCrate, ItemFn, ItemForeignMod, ItemImpl, ItemMacro, ItemMod, ItemStatic, ItemStruct,
    ItemTrait, ItemTraitAlias, ItemType, ItemUnion, ItemUse, Label, Lifetime, LifetimeParam, Lit, LitBool, LitByte,
    LitByteStr, LitChar, LitFloat, LitInt, LitStr, Macro, MacroDelim, MacroDelim, Member, Meta, MetaList,
    MetaNameValue, ParenthesizedGenericArguments, Path, PathArguments, PathSegment, PredicateLifetime, PredicateType,
    QSelf, RangeLimits, Receiver, Signature, Signature, StaticMutability, Stmt, Token, TraitBound, TraitBoundModifier,
    TraitItem, TraitItemConst, TraitItemFn, TraitItemMacro, TraitItemType, TypeImplTrait, TypeParam, TypeParamBound,
    TypeParamBound, UnOp, UseGlob, UseGroup, UseName, UsePath, UseRename, UseTree, Variant, VisRestricted, Visibility,
    Visibility, WhereClause, WherePredicate, *,
};

use std::iter::Peekable;
use std::ops::Deref;
use std::ops::{Index, IndexMut};
use std::ptr;
use std::{borrow::Cow, cmp, collections::VecDeque, iter};

#[derive(Clone, Copy, PartialEq)]
pub enum Breaks {
    Consistent,
    Inconsistent,
}
#[derive(Clone, Copy, Default)]
pub struct BreakToken {
    pub offset: isize,
    pub blank_space: usize,
    pub pre_break: Option<char>,
    pub post_break: Option<char>,
    pub no_break: Option<char>,
    pub if_nonempty: bool,
    pub never_break: bool,
}
#[derive(Clone, Copy)]
pub struct BeginToken {
    pub offset: isize,
    pub breaks: Breaks,
}
#[derive(Clone)]
pub enum Token {
    String(Cow<'static, str>),
    Break(BreakToken),
    Begin(BeginToken),
    End,
}
#[derive(Copy, Clone)]
enum PrintFrame {
    Fits(Breaks),
    Broken(usize, Breaks),
}
pub const SIZE_INFINITY: isize = 0xffff;
pub struct Printer {
    out: String,
    space: isize,
    buf: RingBuffer<BufEntry>,
    left_total: isize,
    right_total: isize,
    scan_stack: VecDeque<usize>,
    print_stack: Vec<PrintFrame>,
    indent: usize,
    pending_indentation: usize,
}
impl Printer {
    pub fn new() -> Self {
        Printer {
            out: String::new(),
            space: MARGIN,
            buf: RingBuffer::new(),
            left_total: 0,
            right_total: 0,
            scan_stack: VecDeque::new(),
            print_stack: Vec::new(),
            indent: 0,
            pending_indentation: 0,
        }
    }
    pub fn eof(mut self) -> String {
        if !self.scan_stack.is_empty() {
            self.check_stack(0);
            self.advance_left();
        }
        self.out
    }
    pub fn scan_begin(&mut self, token: BeginToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.buf.clear();
        }
        let right = self.buf.push(BufEntry {
            token: Token::Begin(token),
            size: -self.right_total,
        });
        self.scan_stack.push_back(right);
    }
    pub fn scan_end(&mut self) {
        if self.scan_stack.is_empty() {
            self.print_end();
        } else {
            if !self.buf.is_empty() {
                if let Token::Break(break_token) = self.buf.last().token {
                    if self.buf.len() >= 2 {
                        if let Token::Begin(_) = self.buf.second_last().token {
                            self.buf.pop_last();
                            self.buf.pop_last();
                            self.scan_stack.pop_back();
                            self.scan_stack.pop_back();
                            self.right_total -= break_token.blank_space as isize;
                            return;
                        }
                    }
                    if break_token.if_nonempty {
                        self.buf.pop_last();
                        self.scan_stack.pop_back();
                        self.right_total -= break_token.blank_space as isize;
                    }
                }
            }
            let right = self.buf.push(BufEntry {
                token: Token::End,
                size: -1,
            });
            self.scan_stack.push_back(right);
        }
    }
    pub fn scan_break(&mut self, token: BreakToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.buf.clear();
        } else {
            self.check_stack(0);
        }
        let right = self.buf.push(BufEntry {
            token: Token::Break(token),
            size: -self.right_total,
        });
        self.scan_stack.push_back(right);
        self.right_total += token.blank_space as isize;
    }
    pub fn scan_string(&mut self, string: Cow<'static, str>) {
        if self.scan_stack.is_empty() {
            self.print_string(string);
        } else {
            let len = string.len() as isize;
            self.buf.push(BufEntry {
                token: Token::String(string),
                size: len,
            });
            self.right_total += len;
            self.check_stream();
        }
    }
    pub fn offset(&mut self, offset: isize) {
        match &mut self.buf.last_mut().token {
            Token::Break(token) => token.offset += offset,
            Token::Begin(_) => {},
            Token::String(_) | Token::End => unreachable!(),
        }
    }
    pub fn end_with_max_width(&mut self, max: isize) {
        let mut depth = 1;
        for &index in self.scan_stack.iter().rev() {
            let entry = &self.buf[index];
            match entry.token {
                Token::Begin(_) => {
                    depth -= 1;
                    if depth == 0 {
                        if entry.size < 0 {
                            let actual_width = entry.size + self.right_total;
                            if actual_width > max {
                                self.buf.push(BufEntry {
                                    token: Token::String(Cow::Borrowed("")),
                                    size: SIZE_INFINITY,
                                });
                                self.right_total += SIZE_INFINITY;
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
        while self.right_total - self.left_total > self.space {
            if *self.scan_stack.front().unwrap() == self.buf.index_of_first() {
                self.scan_stack.pop_front().unwrap();
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
            match left.token {
                Token::String(string) => {
                    self.left_total += left.size;
                    self.print_string(string);
                },
                Token::Break(token) => {
                    self.left_total += token.blank_space as isize;
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
        while let Some(&index) = self.scan_stack.back() {
            let entry = &mut self.buf[index];
            match entry.token {
                Token::Begin(_) => {
                    if depth == 0 {
                        break;
                    }
                    self.scan_stack.pop_back().unwrap();
                    entry.size += self.right_total;
                    depth -= 1;
                },
                Token::End => {
                    self.scan_stack.pop_back().unwrap();
                    entry.size = 1;
                    depth += 1;
                },
                Token::Break(_) => {
                    self.scan_stack.pop_back().unwrap();
                    entry.size += self.right_total;
                    if depth == 0 {
                        break;
                    }
                },
                Token::String(_) => unreachable!(),
            }
        }
    }
    fn get_top(&self) -> PrintFrame {
        const OUTER: PrintFrame = PrintFrame::Broken(0, Breaks::Inconsistent);
        self.print_stack.last().map_or(OUTER, PrintFrame::clone)
    }
    fn print_begin(&mut self, token: BeginToken, size: isize) {
        if cfg!(prettyplease_debug) {
            self.out.push(match token.breaks {
                Breaks::Consistent => '«',
                Breaks::Inconsistent => '‹',
            });
            if cfg!(prettyplease_debug_indent) {
                self.out.extend(token.offset.to_string().chars().map(|ch| match ch {
                    '0'..='9' => ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'][(ch as u8 - b'0') as usize],
                    '-' => '₋',
                    _ => unreachable!(),
                }));
            }
        }
        if size > self.space {
            self.print_stack.push(PrintFrame::Broken(self.indent, token.breaks));
            self.indent = usize::try_from(self.indent as isize + token.offset).unwrap();
        } else {
            self.print_stack.push(PrintFrame::Fits(token.breaks));
        }
    }
    fn print_end(&mut self) {
        let breaks = match self.print_stack.pop().unwrap() {
            PrintFrame::Broken(indent, breaks) => {
                self.indent = indent;
                breaks
            },
            PrintFrame::Fits(breaks) => breaks,
        };
        if cfg!(prettyplease_debug) {
            self.out.push(match breaks {
                Breaks::Consistent => '»',
                Breaks::Inconsistent => '›',
            });
        }
    }
    fn print_break(&mut self, token: BreakToken, size: isize) {
        let fits = token.never_break
            || match self.get_top() {
                PrintFrame::Fits(..) => true,
                PrintFrame::Broken(.., Breaks::Consistent) => false,
                PrintFrame::Broken(.., Breaks::Inconsistent) => size <= self.space,
            };
        if fits {
            self.pending_indentation += token.blank_space;
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
            let indent = self.indent as isize + token.offset;
            self.pending_indentation = usize::try_from(indent).unwrap();
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
        self.out.reserve(self.pending_indentation);
        self.out.extend(iter::repeat(' ').take(self.pending_indentation));
        self.pending_indentation = 0;
    }
}

#[derive(Clone)]
struct BufEntry {
    token: Token,
    size: isize,
}

impl Printer {
    //attr
    pub fn outer_attrs(&mut self, attrs: &[attr::Attr]) {
        for attr in attrs {
            if let attr::Style::Outer = attr.style {
                self.attr(attr);
            }
        }
    }
    pub fn inner_attrs(&mut self, attrs: &[attr::Attr]) {
        for attr in attrs {
            if let attr::Style::Inner(_) = attr.style {
                self.attr(attr);
            }
        }
    }
    fn attr(&mut self, attr: &attr::Attr) {
        if let Some(mut doc) = value_of_attr("doc", attr) {
            if !doc.contains('\n')
                && match attr.style {
                    attr::Style::Outer => !doc.starts_with('/'),
                    attr::Style::Inner(_) => true,
                }
            {
                trim_trailing_spaces(&mut doc);
                self.word(match attr.style {
                    attr::Style::Outer => "///",
                    attr::Style::Inner(_) => "//!",
                });
                self.word(doc);
                self.hardbreak();
                return;
            } else if can_be_block_comment(&doc)
                && match attr.style {
                    attr::Style::Outer => !doc.starts_with(&['*', '/'][..]),
                    attr::Style::Inner(_) => true,
                }
            {
                trim_interior_trailing_spaces(&mut doc);
                self.word(match attr.style {
                    attr::Style::Outer => "/**",
                    attr::Style::Inner(_) => "/*!",
                });
                self.word(doc);
                self.word("*/");
                self.hardbreak();
                return;
            }
        } else if let Some(mut comment) = value_of_attr("comment", attr) {
            if !comment.contains('\n') {
                trim_trailing_spaces(&mut comment);
                self.word("//");
                self.word(comment);
                self.hardbreak();
                return;
            } else if can_be_block_comment(&comment) && !comment.starts_with(&['*', '!'][..]) {
                trim_interior_trailing_spaces(&mut comment);
                self.word("/*");
                self.word(comment);
                self.word("*/");
                self.hardbreak();
                return;
            }
        }
        self.word(match attr.style {
            attr::Style::Outer => "#",
            attr::Style::Inner(_) => "#!",
        });
        self.word("[");
        self.meta(&attr.meta);
        self.word("]");
        self.space();
    }
    fn meta(&mut self, meta: &Meta) {
        match meta {
            Meta::Path(path) => self.path(path, PathKind::Simple),
            Meta::List(meta) => self.meta_list(meta),
            Meta::NameValue(meta) => self.meta_name_value(meta),
        }
    }
    fn meta_list(&mut self, meta: &MetaList) {
        self.path(&meta.path, PathKind::Simple);
        let delim = match meta.delim {
            MacroDelim::Paren(_) => Delim::Parenthesis,
            MacroDelim::Brace(_) => Delim::Brace,
            MacroDelim::Bracket(_) => Delim::Bracket,
        };
        let group = Group::new(delim, meta.tokens.clone());
        self.attr_tokens(Stream::from(Tree::Group(group)));
    }
    fn meta_name_value(&mut self, meta: &MetaNameValue) {
        self.path(&meta.path, PathKind::Simple);
        self.word(" = ");
        self.expr(&meta.value);
    }
    fn attr_tokens(&mut self, tokens: Stream) {
        let mut stack = Vec::new();
        stack.push((tokens.into_iter().peekable(), Delim::None));
        let mut space = Self::nbsp as fn(&mut Self);
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
                Some(Tree::Ident(ident)) => {
                    if let Word = state {
                        space(self);
                    }
                    self.ident(&ident);
                    state = Word;
                },
                Some(Tree::Punct(punct)) => {
                    let ch = punct.as_char();
                    if let (Word, '=') = (state, ch) {
                        self.nbsp();
                    }
                    if ch == ',' && tokens.peek().is_none() {
                        self.trailing_comma(true);
                        state = TrailingComma;
                    } else {
                        self.token_punct(ch);
                        if ch == '=' {
                            self.nbsp();
                        } else if ch == ',' {
                            space(self);
                        }
                        state = Punct;
                    }
                },
                Some(Tree::Literal(literal)) => {
                    if let Word = state {
                        space(self);
                    }
                    self.token_literal(&literal);
                    state = Word;
                },
                Some(Tree::Group(group)) => {
                    let delim = group.delim();
                    let stream = group.stream();
                    match delim {
                        Delim::Parenthesis => {
                            self.word("(");
                            self.cbox(INDENT);
                            self.zerobreak();
                            state = Punct;
                        },
                        Delim::Brace => {
                            self.word("{");
                            state = Punct;
                        },
                        Delim::Bracket => {
                            self.word("[");
                            state = Punct;
                        },
                        Delim::None => {},
                    }
                    stack.push((stream.into_iter().peekable(), delim));
                    space = Self::space;
                },
                None => {
                    match delim {
                        Delim::Parenthesis => {
                            if state != TrailingComma {
                                self.zerobreak();
                            }
                            self.offset(-INDENT);
                            self.end();
                            self.word(")");
                            state = Punct;
                        },
                        Delim::Brace => {
                            self.word("}");
                            state = Punct;
                        },
                        Delim::Bracket => {
                            self.word("]");
                            state = Punct;
                        },
                        Delim::None => {},
                    }
                    stack.pop();
                    if stack.is_empty() {
                        space = Self::nbsp;
                    }
                },
            }
        }
    }
}
fn value_of_attr(requested: &str, attr: &attr::Attr) -> Option<String> {
    let value = match &attr.meta {
        Meta::NameValue(meta) if meta.path.is_ident(requested) => &meta.value,
        _ => return None,
    };
    let lit = match value {
        Expr::Lit(expr) if expr.attrs.is_empty() => &expr.lit,
        _ => return None,
    };
    match lit {
        Lit::Str(string) => Some(string.value()),
        _ => None,
    }
}
fn has_outer(attrs: &[attr::Attr]) -> bool {
    for attr in attrs {
        if let attr::Style::Outer = attr.style {
            return true;
        }
    }
    false
}
fn has_inner(attrs: &[attr::Attr]) -> bool {
    for attr in attrs {
        if let attr::Style::Inner(_) = attr.style {
            return true;
        }
    }
    false
}
fn trim_trailing_spaces(doc: &mut String) {
    doc.truncate(doc.trim_end_matches(' ').len());
}
fn trim_interior_trailing_spaces(doc: &mut String) {
    if !doc.contains(" \n") {
        return;
    }
    let mut trimmed = String::with_capacity(doc.len());
    let mut lines = doc.split('\n').peekable();
    while let Some(line) = lines.next() {
        if lines.peek().is_some() {
            trimmed.push_str(line.trim_end_matches(' '));
            trimmed.push('\n');
        } else {
            trimmed.push_str(line);
        }
    }
    *doc = trimmed;
}
fn can_be_block_comment(value: &str) -> bool {
    let mut depth = 0usize;
    let bytes = value.as_bytes();
    let mut i = 0usize;
    let upper = bytes.len() - 1;
    while i < upper {
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
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

impl Printer {
    //convenience
    pub fn ibox(&mut self, indent: isize) {
        self.scan_begin(BeginToken {
            offset: indent,
            breaks: Breaks::Inconsistent,
        });
    }
    pub fn cbox(&mut self, indent: isize) {
        self.scan_begin(BeginToken {
            offset: indent,
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
        self.scan_break(BreakToken {
            blank_space: n,
            ..BreakToken::default()
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
        self.scan_break(BreakToken {
            blank_space: 1,
            if_nonempty: true,
            ..BreakToken::default()
        });
    }
    pub fn hardbreak_if_nonempty(&mut self) {
        self.scan_break(BreakToken {
            blank_space: SIZE_INFINITY as usize,
            if_nonempty: true,
            ..BreakToken::default()
        });
    }
    pub fn trailing_comma(&mut self, is_last: bool) {
        if is_last {
            self.scan_break(BreakToken {
                pre_break: Some(','),
                ..BreakToken::default()
            });
        } else {
            self.word(",");
            self.space();
        }
    }
    pub fn trailing_comma_or_space(&mut self, is_last: bool) {
        if is_last {
            self.scan_break(BreakToken {
                blank_space: 1,
                pre_break: Some(','),
                ..BreakToken::default()
            });
        } else {
            self.word(",");
            self.space();
        }
    }
    pub fn neverbreak(&mut self) {
        self.scan_break(BreakToken {
            never_break: true,
            ..BreakToken::default()
        });
    }
}

impl Printer {
    //data
    pub fn variant(&mut self, variant: &Variant) {
        self.outer_attrs(&variant.attrs);
        self.ident(&variant.ident);
        match &variant.fields {
            Fields::Named(fields) => {
                self.nbsp();
                self.word("{");
                self.cbox(INDENT);
                self.space();
                for field in fields.named.iter().delimited() {
                    self.field(&field);
                    self.trailing_comma_or_space(field.is_last);
                }
                self.offset(-INDENT);
                self.end();
                self.word("}");
            },
            Fields::Unnamed(fields) => {
                self.cbox(INDENT);
                self.fields_unnamed(fields);
                self.end();
            },
            Fields::Unit => {},
        }
        if let Some((_eq_token, discriminant)) = &variant.discriminant {
            self.word(" = ");
            self.expr(discriminant);
        }
    }
    pub fn fields_unnamed(&mut self, fields: &FieldsUnnamed) {
        self.word("(");
        self.zerobreak();
        for field in fields.unnamed.iter().delimited() {
            self.field(&field);
            self.trailing_comma(field.is_last);
        }
        self.offset(-INDENT);
        self.word(")");
    }
    pub fn field(&mut self, field: &Field) {
        self.outer_attrs(&field.attrs);
        self.visibility(&field.vis);
        if let Some(ident) = &field.ident {
            self.ident(ident);
            self.word(": ");
        }
        self.ty(&field.ty);
    }
    pub fn visibility(&mut self, vis: &Visibility) {
        match vis {
            Visibility::Public(_) => self.word("pub "),
            Visibility::Restricted(vis) => self.vis_restricted(vis),
            Visibility::Inherited => {},
        }
    }
    fn vis_restricted(&mut self, vis: &VisRestricted) {
        self.word("pub(");
        let omit_in = vis.path.get_ident().map_or(false, |ident| {
            matches!(ident.to_string().as_str(), "self" | "super" | "crate")
        });
        if !omit_in {
            self.word("in ");
        }
        self.path(&vis.path, PathKind::Simple);
        self.word(") ");
    }
}

impl Printer {
    //expr
    pub fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Array(expr) => self.expr_array(expr),
            Expr::Assign(expr) => self.expr_assign(expr),
            Expr::Async(expr) => self.expr_async(expr),
            Expr::Await(expr) => self.expr_await(expr, false),
            Expr::Binary(expr) => self.expr_binary(expr),
            Expr::Block(expr) => self.expr_block(expr),
            Expr::Break(expr) => self.expr_break(expr),
            Expr::Call(expr) => self.expr_call(expr, false),
            Expr::Cast(expr) => self.expr_cast(expr),
            Expr::Closure(expr) => self.expr_closure(expr),
            Expr::Const(expr) => self.expr_const(expr),
            Expr::Continue(expr) => self.expr_continue(expr),
            Expr::Field(expr) => self.expr_field(expr, false),
            Expr::ForLoop(expr) => self.expr_for_loop(expr),
            Expr::Group(expr) => self.expr_group(expr),
            Expr::If(expr) => self.expr_if(expr),
            Expr::Index(expr) => self.expr_index(expr, false),
            Expr::Infer(expr) => self.expr_infer(expr),
            Expr::Let(expr) => self.expr_let(expr),
            Expr::Lit(expr) => self.expr_lit(expr),
            Expr::Loop(expr) => self.expr_loop(expr),
            Expr::Macro(expr) => self.expr_macro(expr),
            Expr::Match(expr) => self.expr_match(expr),
            Expr::MethodCall(expr) => self.expr_method_call(expr, false),
            Expr::Paren(expr) => self.expr_paren(expr),
            Expr::Path(expr) => self.expr_path(expr),
            Expr::Range(expr) => self.expr_range(expr),
            Expr::Reference(expr) => self.expr_reference(expr),
            Expr::Repeat(expr) => self.expr_repeat(expr),
            Expr::Return(expr) => self.expr_return(expr),
            Expr::Struct(expr) => self.expr_struct(expr),
            Expr::Try(expr) => self.expr_try(expr, false),
            Expr::TryBlock(expr) => self.expr_try_block(expr),
            Expr::Tuple(expr) => self.expr_tuple(expr),
            Expr::Unary(expr) => self.expr_unary(expr),
            Expr::Unsafe(expr) => self.expr_unsafe(expr),
            Expr::Verbatim(expr) => self.expr_verbatim(expr),
            Expr::While(expr) => self.expr_while(expr),
            Expr::Yield(expr) => self.expr_yield(expr),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown Expr"),
        }
    }
    pub fn expr_beginning_of_line(&mut self, expr: &Expr, beginning_of_line: bool) {
        match expr {
            Expr::Await(expr) => self.expr_await(expr, beginning_of_line),
            Expr::Field(expr) => self.expr_field(expr, beginning_of_line),
            Expr::Index(expr) => self.expr_index(expr, beginning_of_line),
            Expr::MethodCall(expr) => self.expr_method_call(expr, beginning_of_line),
            Expr::Try(expr) => self.expr_try(expr, beginning_of_line),
            _ => self.expr(expr),
        }
    }
    fn subexpr(&mut self, expr: &Expr, beginning_of_line: bool) {
        match expr {
            Expr::Await(expr) => self.subexpr_await(expr, beginning_of_line),
            Expr::Call(expr) => self.subexpr_call(expr),
            Expr::Field(expr) => self.subexpr_field(expr, beginning_of_line),
            Expr::Index(expr) => self.subexpr_index(expr, beginning_of_line),
            Expr::MethodCall(expr) => self.subexpr_method_call(expr, beginning_of_line, false),
            Expr::Try(expr) => self.subexpr_try(expr, beginning_of_line),
            _ => {
                self.cbox(-INDENT);
                self.expr(expr);
                self.end();
            },
        }
    }
    fn wrap_exterior_struct(&mut self, expr: &Expr) {
        let needs_paren = contains_exterior_struct_lit(expr);
        if needs_paren {
            self.word("(");
        }
        self.cbox(0);
        self.expr(expr);
        if needs_paren {
            self.word(")");
        }
        if needs_newline_if_wrap(expr) {
            self.space();
        } else {
            self.nbsp();
        }
        self.end();
    }
    fn expr_array(&mut self, expr: &expr::Array) {
        self.outer_attrs(&expr.attrs);
        self.word("[");
        self.cbox(INDENT);
        self.zerobreak();
        for element in expr.elems.iter().delimited() {
            self.expr(&element);
            self.trailing_comma(element.is_last);
        }
        self.offset(-INDENT);
        self.end();
        self.word("]");
    }
    fn expr_assign(&mut self, expr: &expr::Assign) {
        self.outer_attrs(&expr.attrs);
        self.ibox(0);
        self.expr(&expr.left);
        self.word(" = ");
        self.expr(&expr.right);
        self.end();
    }
    fn expr_async(&mut self, expr: &expr::Async) {
        self.outer_attrs(&expr.attrs);
        self.word("async ");
        if expr.capture.is_some() {
            self.word("move ");
        }
        self.cbox(INDENT);
        self.small_block(&expr.block, &expr.attrs);
        self.end();
    }
    fn expr_await(&mut self, expr: &expr::Await, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.cbox(INDENT);
        self.subexpr_await(expr, beginning_of_line);
        self.end();
    }
    fn subexpr_await(&mut self, expr: &expr::Await, beginning_of_line: bool) {
        self.subexpr(&expr.base, beginning_of_line);
        self.zerobreak_unless_short_ident(beginning_of_line, &expr.base);
        self.word(".await");
    }
    fn expr_binary(&mut self, expr: &expr::Binary) {
        self.outer_attrs(&expr.attrs);
        self.ibox(INDENT);
        self.ibox(-INDENT);
        self.expr(&expr.left);
        self.end();
        self.space();
        self.binary_operator(&expr.op);
        self.nbsp();
        self.expr(&expr.right);
        self.end();
    }
    pub fn expr_block(&mut self, expr: &expr::Block) {
        self.outer_attrs(&expr.attrs);
        if let Some(label) = &expr.label {
            self.label(label);
        }
        self.cbox(INDENT);
        self.small_block(&expr.block, &expr.attrs);
        self.end();
    }
    fn expr_break(&mut self, expr: &expr::Break) {
        self.outer_attrs(&expr.attrs);
        self.word("break");
        if let Some(lifetime) = &expr.label {
            self.nbsp();
            self.lifetime(lifetime);
        }
        if let Some(value) = &expr.expr {
            self.nbsp();
            self.expr(value);
        }
    }
    fn expr_call(&mut self, expr: &expr::Call, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.expr_beginning_of_line(&expr.func, beginning_of_line);
        self.word("(");
        self.call_args(&expr.args);
        self.word(")");
    }
    fn subexpr_call(&mut self, expr: &expr::Call) {
        self.subexpr(&expr.func, false);
        self.word("(");
        self.call_args(&expr.args);
        self.word(")");
    }
    fn expr_cast(&mut self, expr: &expr::Cast) {
        self.outer_attrs(&expr.attrs);
        self.ibox(INDENT);
        self.ibox(-INDENT);
        self.expr(&expr.expr);
        self.end();
        self.space();
        self.word("as ");
        self.ty(&expr.ty);
        self.end();
    }
    fn expr_closure(&mut self, expr: &expr::Closure) {
        self.outer_attrs(&expr.attrs);
        self.ibox(0);
        if let Some(bound_lifetimes) = &expr.lifetimes {
            self.bound_lifetimes(bound_lifetimes);
        }
        if expr.constness.is_some() {
            self.word("const ");
        }
        if expr.movability.is_some() {
            self.word("static ");
        }
        if expr.asyncness.is_some() {
            self.word("async ");
        }
        if expr.capture.is_some() {
            self.word("move ");
        }
        self.cbox(INDENT);
        self.word("|");
        for pat in expr.inputs.iter().delimited() {
            if pat.is_first {
                self.zerobreak();
            }
            self.pat(&pat);
            if !pat.is_last {
                self.word(",");
                self.space();
            }
        }
        match &expr.output {
            typ::Ret::Default => {
                self.word("|");
                self.space();
                self.offset(-INDENT);
                self.end();
                self.neverbreak();
                let wrap_in_brace = match &*expr.body {
                    Expr::Match(expr::Match { attrs, .. }) | Expr::Call(expr::Call { attrs, .. }) => has_outer(attrs),
                    body => !is_blocklike(body),
                };
                if wrap_in_brace {
                    self.cbox(INDENT);
                    let okay_to_brace = parseable_as_stmt(&expr.body);
                    self.scan_break(BreakToken {
                        pre_break: Some(if okay_to_brace { '{' } else { '(' }),
                        ..BreakToken::default()
                    });
                    self.expr(&expr.body);
                    self.scan_break(BreakToken {
                        offset: -INDENT,
                        pre_break: (okay_to_brace && add_semi(&expr.body)).then(|| ';'),
                        post_break: Some(if okay_to_brace { '}' } else { ')' }),
                        ..BreakToken::default()
                    });
                    self.end();
                } else {
                    self.expr(&expr.body);
                }
            },
            typ::Ret::Type(_arrow, ty) => {
                if !expr.inputs.is_empty() {
                    self.trailing_comma(true);
                    self.offset(-INDENT);
                }
                self.word("|");
                self.end();
                self.word(" -> ");
                self.ty(ty);
                self.nbsp();
                self.neverbreak();
                self.expr(&expr.body);
            },
        }
        self.end();
    }
    pub fn expr_const(&mut self, expr: &expr::Const) {
        self.outer_attrs(&expr.attrs);
        self.word("const ");
        self.cbox(INDENT);
        self.small_block(&expr.block, &expr.attrs);
        self.end();
    }
    fn expr_continue(&mut self, expr: &expr::Continue) {
        self.outer_attrs(&expr.attrs);
        self.word("continue");
        if let Some(lifetime) = &expr.label {
            self.nbsp();
            self.lifetime(lifetime);
        }
    }
    fn expr_field(&mut self, expr: &expr::Field, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.cbox(INDENT);
        self.subexpr_field(expr, beginning_of_line);
        self.end();
    }
    fn subexpr_field(&mut self, expr: &expr::Field, beginning_of_line: bool) {
        self.subexpr(&expr.base, beginning_of_line);
        self.zerobreak_unless_short_ident(beginning_of_line, &expr.base);
        self.word(".");
        self.member(&expr.member);
    }
    fn expr_for_loop(&mut self, expr: &expr::ForLoop) {
        self.outer_attrs(&expr.attrs);
        self.ibox(0);
        if let Some(label) = &expr.label {
            self.label(label);
        }
        self.word("for ");
        self.pat(&expr.pat);
        self.word(" in ");
        self.neverbreak();
        self.wrap_exterior_struct(&expr.expr);
        self.word("{");
        self.neverbreak();
        self.cbox(INDENT);
        self.hardbreak_if_nonempty();
        self.inner_attrs(&expr.attrs);
        for stmt in &expr.body.stmts {
            self.stmt(stmt);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.end();
    }
    fn expr_group(&mut self, expr: &expr::Group) {
        self.outer_attrs(&expr.attrs);
        self.expr(&expr.expr);
    }
    fn expr_if(&mut self, expr: &expr::If) {
        self.outer_attrs(&expr.attrs);
        self.cbox(INDENT);
        self.word("if ");
        self.cbox(-INDENT);
        self.wrap_exterior_struct(&expr.cond);
        self.end();
        if let Some((_else_token, else_branch)) = &expr.else_branch {
            let mut else_branch = &**else_branch;
            self.small_block(&expr.then_branch, &[]);
            loop {
                self.word(" else ");
                match else_branch {
                    Expr::If(expr) => {
                        self.word("if ");
                        self.cbox(-INDENT);
                        self.wrap_exterior_struct(&expr.cond);
                        self.end();
                        self.small_block(&expr.then_branch, &[]);
                        if let Some((_else_token, next)) = &expr.else_branch {
                            else_branch = next;
                            continue;
                        }
                    },
                    Expr::Block(expr) => {
                        self.small_block(&expr.block, &[]);
                    },
                    other => {
                        self.word("{");
                        self.space();
                        self.ibox(INDENT);
                        self.expr(other);
                        self.end();
                        self.space();
                        self.offset(-INDENT);
                        self.word("}");
                    },
                }
                break;
            }
        } else if expr.then_branch.stmts.is_empty() {
            self.word("{}");
        } else {
            self.word("{");
            self.hardbreak();
            for stmt in &expr.then_branch.stmts {
                self.stmt(stmt);
            }
            self.offset(-INDENT);
            self.word("}");
        }
        self.end();
    }
    fn expr_index(&mut self, expr: &expr::Index, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.expr_beginning_of_line(&expr.expr, beginning_of_line);
        self.word("[");
        self.expr(&expr.index);
        self.word("]");
    }
    fn subexpr_index(&mut self, expr: &expr::Index, beginning_of_line: bool) {
        self.subexpr(&expr.expr, beginning_of_line);
        self.word("[");
        self.expr(&expr.index);
        self.word("]");
    }
    fn expr_infer(&mut self, expr: &expr::Infer) {
        self.outer_attrs(&expr.attrs);
        self.word("_");
    }
    fn expr_let(&mut self, expr: &expr::Let) {
        self.outer_attrs(&expr.attrs);
        self.ibox(INDENT);
        self.word("let ");
        self.ibox(-INDENT);
        self.pat(&expr.pat);
        self.end();
        self.space();
        self.word("= ");
        let needs_paren = contains_exterior_struct_lit(&expr.expr);
        if needs_paren {
            self.word("(");
        }
        self.expr(&expr.expr);
        if needs_paren {
            self.word(")");
        }
        self.end();
    }
    pub fn expr_lit(&mut self, expr: &expr::Lit) {
        self.outer_attrs(&expr.attrs);
        self.lit(&expr.lit);
    }
    fn expr_loop(&mut self, expr: &expr::Loop) {
        self.outer_attrs(&expr.attrs);
        if let Some(label) = &expr.label {
            self.label(label);
        }
        self.word("loop {");
        self.cbox(INDENT);
        self.hardbreak_if_nonempty();
        self.inner_attrs(&expr.attrs);
        for stmt in &expr.body.stmts {
            self.stmt(stmt);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
    }
    pub fn expr_macro(&mut self, expr: &expr::Mac) {
        self.outer_attrs(&expr.attrs);
        let semicolon = false;
        self.mac(&expr.mac, None, semicolon);
    }
    fn expr_match(&mut self, expr: &expr::Match) {
        self.outer_attrs(&expr.attrs);
        self.ibox(0);
        self.word("match ");
        self.wrap_exterior_struct(&expr.expr);
        self.word("{");
        self.neverbreak();
        self.cbox(INDENT);
        self.hardbreak_if_nonempty();
        self.inner_attrs(&expr.attrs);
        for arm in &expr.arms {
            self.arm(arm);
            self.hardbreak();
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.end();
    }
    fn expr_method_call(&mut self, expr: &expr::MethodCall, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.cbox(INDENT);
        let unindent_call_args = beginning_of_line && is_short_ident(&expr.receiver);
        self.subexpr_method_call(expr, beginning_of_line, unindent_call_args);
        self.end();
    }
    fn subexpr_method_call(&mut self, expr: &expr::MethodCall, beginning_of_line: bool, unindent_call_args: bool) {
        self.subexpr(&expr.receiver, beginning_of_line);
        self.zerobreak_unless_short_ident(beginning_of_line, &expr.receiver);
        self.word(".");
        self.ident(&expr.method);
        if let Some(turbofish) = &expr.turbofish {
            self.angle_bracketed_generic_arguments(turbofish, PathKind::Expr);
        }
        self.cbox(if unindent_call_args { -INDENT } else { 0 });
        self.word("(");
        self.call_args(&expr.args);
        self.word(")");
        self.end();
    }
    fn expr_paren(&mut self, expr: &expr::Paren) {
        self.outer_attrs(&expr.attrs);
        self.word("(");
        self.expr(&expr.expr);
        self.word(")");
    }
    pub fn expr_path(&mut self, expr: &expr::Path) {
        self.outer_attrs(&expr.attrs);
        self.qpath(&expr.qself, &expr.path, PathKind::Expr);
    }
    pub fn expr_range(&mut self, expr: &expr::Range) {
        self.outer_attrs(&expr.attrs);
        if let Some(start) = &expr.start {
            self.expr(start);
        }
        self.word(match expr.limits {
            RangeLimits::HalfOpen(_) => "..",
            RangeLimits::Closed(_) => "..=",
        });
        if let Some(end) = &expr.end {
            self.expr(end);
        }
    }
    fn expr_reference(&mut self, expr: &expr::Ref) {
        self.outer_attrs(&expr.attrs);
        self.word("&");
        if expr.mutability.is_some() {
            self.word("mut ");
        }
        self.expr(&expr.expr);
    }
    fn expr_repeat(&mut self, expr: &expr::Repeat) {
        self.outer_attrs(&expr.attrs);
        self.word("[");
        self.expr(&expr.expr);
        self.word("; ");
        self.expr(&expr.len);
        self.word("]");
    }
    fn expr_return(&mut self, expr: &expr::Ret) {
        self.outer_attrs(&expr.attrs);
        self.word("return");
        if let Some(value) = &expr.expr {
            self.nbsp();
            self.expr(value);
        }
    }
    fn expr_struct(&mut self, expr: &expr::Struct) {
        self.outer_attrs(&expr.attrs);
        self.cbox(INDENT);
        self.ibox(-INDENT);
        self.qpath(&expr.qself, &expr.path, PathKind::Expr);
        self.end();
        self.word(" {");
        self.space_if_nonempty();
        for field_value in expr.fields.iter().delimited() {
            self.field_value(&field_value);
            self.trailing_comma_or_space(field_value.is_last && expr.rest.is_none());
        }
        if let Some(rest) = &expr.rest {
            self.word("..");
            self.expr(rest);
            self.space();
        }
        self.offset(-INDENT);
        self.end_with_max_width(34);
        self.word("}");
    }
    fn expr_try(&mut self, expr: &expr::Try, beginning_of_line: bool) {
        self.outer_attrs(&expr.attrs);
        self.expr_beginning_of_line(&expr.expr, beginning_of_line);
        self.word("?");
    }
    fn subexpr_try(&mut self, expr: &expr::Try, beginning_of_line: bool) {
        self.subexpr(&expr.expr, beginning_of_line);
        self.word("?");
    }
    fn expr_try_block(&mut self, expr: &expr::TryBlock) {
        self.outer_attrs(&expr.attrs);
        self.word("try ");
        self.cbox(INDENT);
        self.small_block(&expr.block, &expr.attrs);
        self.end();
    }
    fn expr_tuple(&mut self, expr: &ExprTuple) {
        self.outer_attrs(&expr.attrs);
        self.word("(");
        self.cbox(INDENT);
        self.zerobreak();
        for elem in expr.elems.iter().delimited() {
            self.expr(&elem);
            if expr.elems.len() == 1 {
                self.word(",");
                self.zerobreak();
            } else {
                self.trailing_comma(elem.is_last);
            }
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
    }
    fn expr_unary(&mut self, expr: &expr::Unary) {
        self.outer_attrs(&expr.attrs);
        self.unary_operator(&expr.op);
        self.expr(&expr.expr);
    }
    fn expr_unsafe(&mut self, expr: &expr::Unsafe) {
        self.outer_attrs(&expr.attrs);
        self.word("unsafe ");
        self.cbox(INDENT);
        self.small_block(&expr.block, &expr.attrs);
        self.end();
    }
    #[cfg(not(feature = "verbatim"))]
    fn expr_verbatim(&mut self, expr: &Stream) {
        if !expr.is_empty() {
            unimplemented!("Expr::Verbatim `{}`", expr);
        }
    }
    #[cfg(feature = "verbatim")]
    fn expr_verbatim(&mut self, tokens: &Stream) {
        enum ExprVerbatim {
            Empty,
            Ellipsis,
            Builtin(Builtin),
            RawReference(RawReference),
        }
        struct Builtin {
            attrs: Vec<attr::Attr>,
            name: Ident,
            args: Stream,
        }
        struct RawReference {
            attrs: Vec<attr::Attr>,
            mutable: bool,
            expr: Expr,
        }
        mod kw {
            syn::custom_keyword!(builtin);
            syn::custom_keyword!(raw);
        }
        impl parse::Parse for ExprVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                let ahead = input.fork();
                let attrs = ahead.call(attr::Attr::parse_outer)?;
                let lookahead = ahead.lookahead1();
                if input.is_empty() {
                    Ok(ExprVerbatim::Empty)
                } else if lookahead.peek(kw::builtin) {
                    input.advance_to(&ahead);
                    input.parse::<kw::builtin>()?;
                    input.parse::<Token![#]>()?;
                    let name: Ident = input.parse()?;
                    let args;
                    parenthesized!(args in input);
                    let args: Stream = args.parse()?;
                    Ok(ExprVerbatim::Builtin(Builtin { attrs, name, args }))
                } else if lookahead.peek(Token![&]) {
                    input.advance_to(&ahead);
                    input.parse::<Token![&]>()?;
                    input.parse::<kw::raw>()?;
                    let mutable = input.parse::<Option<Token![mut]>>()?.is_some();
                    if !mutable {
                        input.parse::<Token![const]>()?;
                    }
                    let expr: Expr = input.parse()?;
                    Ok(ExprVerbatim::RawReference(RawReference { attrs, mutable, expr }))
                } else if lookahead.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    Ok(ExprVerbatim::Ellipsis)
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let expr: ExprVerbatim = match syn::parse2(tokens.clone()) {
            Ok(expr) => expr,
            Err(_) => unimplemented!("Expr::Verbatim `{}`", tokens),
        };
        match expr {
            ExprVerbatim::Empty => {},
            ExprVerbatim::Ellipsis => {
                self.word("...");
            },
            ExprVerbatim::Builtin(expr) => {
                self.outer_attrs(&expr.attrs);
                self.word("builtin # ");
                self.ident(&expr.name);
                self.word("(");
                if !expr.args.is_empty() {
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.ibox(0);
                    self.macro_rules_tokens(expr.args, false);
                    self.end();
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                }
                self.word(")");
            },
            ExprVerbatim::RawReference(expr) => {
                self.outer_attrs(&expr.attrs);
                self.word("&raw ");
                self.word(if expr.mutable { "mut " } else { "const " });
                self.expr(&expr.expr);
            },
        }
    }
    fn expr_while(&mut self, expr: &expr::While) {
        self.outer_attrs(&expr.attrs);
        if let Some(label) = &expr.label {
            self.label(label);
        }
        self.word("while ");
        self.wrap_exterior_struct(&expr.cond);
        self.word("{");
        self.neverbreak();
        self.cbox(INDENT);
        self.hardbreak_if_nonempty();
        self.inner_attrs(&expr.attrs);
        for stmt in &expr.body.stmts {
            self.stmt(stmt);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
    }
    fn expr_yield(&mut self, expr: &expr::Yield) {
        self.outer_attrs(&expr.attrs);
        self.word("yield");
        if let Some(value) = &expr.expr {
            self.nbsp();
            self.expr(value);
        }
    }
    fn label(&mut self, label: &Label) {
        self.lifetime(&label.name);
        self.word(": ");
    }
    fn field_value(&mut self, field_value: &FieldValue) {
        self.outer_attrs(&field_value.attrs);
        self.member(&field_value.member);
        if field_value.colon_token.is_some() {
            self.word(": ");
            self.ibox(0);
            self.expr(&field_value.expr);
            self.end();
        }
    }
    fn arm(&mut self, arm: &Arm) {
        self.outer_attrs(&arm.attrs);
        self.ibox(0);
        self.pat(&arm.pat);
        if let Some((_if_token, guard)) = &arm.guard {
            self.word(" if ");
            self.expr(guard);
        }
        self.word(" =>");
        let empty_block;
        let mut body = &*arm.body;
        while let Expr::Block(expr) = body {
            if expr.attrs.is_empty() && expr.label.is_none() {
                let mut stmts = expr.block.stmts.iter();
                if let (Some(Stmt::Expr(inner, None)), None) = (stmts.next(), stmts.next()) {
                    body = inner;
                    continue;
                }
            }
            break;
        }
        if let Expr::Tuple(expr) = body {
            if expr.elems.is_empty() && expr.attrs.is_empty() {
                empty_block = Expr::Block(expr::Block {
                    attrs: Vec::new(),
                    label: None,
                    block: Block {
                        brace_token: tok::Brace::default(),
                        stmts: Vec::new(),
                    },
                });
                body = &empty_block;
            }
        }
        if let Expr::Block(body) = body {
            self.nbsp();
            if let Some(label) = &body.label {
                self.label(label);
            }
            self.word("{");
            self.neverbreak();
            self.cbox(INDENT);
            self.hardbreak_if_nonempty();
            self.inner_attrs(&body.attrs);
            for stmt in &body.block.stmts {
                self.stmt(stmt);
            }
            self.offset(-INDENT);
            self.end();
            self.word("}");
            self.end();
        } else {
            self.nbsp();
            self.neverbreak();
            self.cbox(INDENT);
            self.scan_break(BreakToken {
                pre_break: Some('{'),
                ..BreakToken::default()
            });
            self.expr_beginning_of_line(body, true);
            self.scan_break(BreakToken {
                offset: -INDENT,
                pre_break: add_semi(body).then(|| ';'),
                post_break: Some('}'),
                no_break: requires_terminator(body).then(|| ','),
                ..BreakToken::default()
            });
            self.end();
            self.end();
        }
    }
    fn call_args(&mut self, args: &punct::Puncted<Expr, Token![,]>) {
        let mut iter = args.iter();
        match (iter.next(), iter.next()) {
            (Some(expr), None) if is_blocklike(expr) => {
                self.expr(expr);
            },
            _ => {
                self.cbox(INDENT);
                self.zerobreak();
                for arg in args.iter().delimited() {
                    self.expr(&arg);
                    self.trailing_comma(arg.is_last);
                }
                self.offset(-INDENT);
                self.end();
            },
        }
    }
    pub fn small_block(&mut self, block: &Block, attrs: &[attr::Attr]) {
        self.word("{");
        if has_inner(attrs) || !block.stmts.is_empty() {
            self.space();
            self.inner_attrs(attrs);
            match (block.stmts.get(0), block.stmts.get(1)) {
                (Some(Stmt::Expr(expr, None)), None) if break_after(expr) => {
                    self.ibox(0);
                    self.expr_beginning_of_line(expr, true);
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
    pub fn member(&mut self, member: &Member) {
        match member {
            Member::Named(ident) => self.ident(ident),
            Member::Unnamed(index) => self.index(index),
        }
    }
    fn index(&mut self, member: &Index) {
        self.word(member.index.to_string());
    }
    fn binary_operator(&mut self, op: &BinOp) {
        self.word(match op {
            BinOp::Add(_) => "+",
            BinOp::Sub(_) => "-",
            BinOp::Mul(_) => "*",
            BinOp::Div(_) => "/",
            BinOp::Rem(_) => "%",
            BinOp::And(_) => "&&",
            BinOp::Or(_) => "||",
            BinOp::BitXor(_) => "^",
            BinOp::BitAnd(_) => "&",
            BinOp::BitOr(_) => "|",
            BinOp::Shl(_) => "<<",
            BinOp::Shr(_) => ">>",
            BinOp::Eq(_) => "==",
            BinOp::Lt(_) => "<",
            BinOp::Le(_) => "<=",
            BinOp::Ne(_) => "!=",
            BinOp::Ge(_) => ">=",
            BinOp::Gt(_) => ">",
            BinOp::AddAssign(_) => "+=",
            BinOp::SubAssign(_) => "-=",
            BinOp::MulAssign(_) => "*=",
            BinOp::DivAssign(_) => "/=",
            BinOp::RemAssign(_) => "%=",
            BinOp::BitXorAssign(_) => "^=",
            BinOp::BitAndAssign(_) => "&=",
            BinOp::BitOrAssign(_) => "|=",
            BinOp::ShlAssign(_) => "<<=",
            BinOp::ShrAssign(_) => ">>=",
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown BinOp"),
        });
    }
    fn unary_operator(&mut self, op: &UnOp) {
        self.word(match op {
            UnOp::Deref(_) => "*",
            UnOp::Not(_) => "!",
            UnOp::Neg(_) => "-",
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown UnOp"),
        });
    }
    fn zerobreak_unless_short_ident(&mut self, beginning_of_line: bool, expr: &Expr) {
        if beginning_of_line && is_short_ident(expr) {
            return;
        }
        self.zerobreak();
    }
}
fn requires_terminator(expr: &Expr) -> bool {
    match expr {
        Expr::If(_)
        | Expr::Match(_)
        | Expr::Block(_)
        | Expr::Unsafe(_)
        | Expr::While(_)
        | Expr::Loop(_)
        | Expr::ForLoop(_)
        | Expr::TryBlock(_)
        | Expr::Const(_) => false,
        Expr::Array(_)
        | Expr::Assign(_)
        | Expr::Async(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Closure(_)
        | Expr::Continue(_)
        | Expr::Field(_)
        | Expr::Group(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Macro(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Struct(_)
        | Expr::Try(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Verbatim(_)
        | Expr::Yield(_) => true,
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => true,
    }
}
fn contains_exterior_struct_lit(expr: &Expr) -> bool {
    match expr {
        Expr::Struct(_) => true,
        Expr::Assign(expr::Assign { left, right, .. }) | Expr::Binary(expr::Binary { left, right, .. }) => {
            contains_exterior_struct_lit(left) || contains_exterior_struct_lit(right)
        },
        Expr::Await(expr::Await { base: e, .. })
        | Expr::Cast(expr::Cast { expr: e, .. })
        | Expr::Field(expr::Field { base: e, .. })
        | Expr::Index(expr::Index { expr: e, .. })
        | Expr::MethodCall(expr::MethodCall { receiver: e, .. })
        | Expr::Reference(expr::Ref { expr: e, .. })
        | Expr::Unary(expr::Unary { expr: e, .. }) => contains_exterior_struct_lit(e),
        Expr::Array(_)
        | Expr::Async(_)
        | Expr::Block(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Closure(_)
        | Expr::Const(_)
        | Expr::Continue(_)
        | Expr::ForLoop(_)
        | Expr::Group(_)
        | Expr::If(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Try(_)
        | Expr::TryBlock(_)
        | Expr::Tuple(_)
        | Expr::Unsafe(_)
        | Expr::Verbatim(_)
        | Expr::While(_)
        | Expr::Yield(_) => false,
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}
fn needs_newline_if_wrap(expr: &Expr) -> bool {
    match expr {
        Expr::Array(_)
        | Expr::Async(_)
        | Expr::Block(_)
        | Expr::Break(expr::Break { expr: None, .. })
        | Expr::Closure(_)
        | Expr::Const(_)
        | Expr::Continue(_)
        | Expr::ForLoop(_)
        | Expr::If(_)
        | Expr::Infer(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::Path(_)
        | Expr::Range(expr::Range { end: None, .. })
        | Expr::Repeat(_)
        | Expr::Return(expr::Ret { expr: None, .. })
        | Expr::Struct(_)
        | Expr::TryBlock(_)
        | Expr::Tuple(_)
        | Expr::Unsafe(_)
        | Expr::Verbatim(_)
        | Expr::While(_)
        | Expr::Yield(expr::Yield { expr: None, .. }) => false,
        Expr::Assign(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Cast(_)
        | Expr::Field(_)
        | Expr::Index(_)
        | Expr::MethodCall(_) => true,
        Expr::Break(expr::Break { expr: Some(e), .. })
        | Expr::Call(expr::Call { func: e, .. })
        | Expr::Group(expr::Group { expr: e, .. })
        | Expr::Let(expr::Let { expr: e, .. })
        | Expr::Paren(expr::Paren { expr: e, .. })
        | Expr::Range(expr::Range { end: Some(e), .. })
        | Expr::Reference(expr::Ref { expr: e, .. })
        | Expr::Return(expr::Ret { expr: Some(e), .. })
        | Expr::Try(expr::Try { expr: e, .. })
        | Expr::Unary(expr::Unary { expr: e, .. })
        | Expr::Yield(expr::Yield { expr: Some(e), .. }) => needs_newline_if_wrap(e),
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}
fn is_short_ident(expr: &Expr) -> bool {
    if let Expr::Path(expr) = expr {
        return expr.attrs.is_empty()
            && expr.qself.is_none()
            && expr
                .path
                .get_ident()
                .map_or(false, |ident| ident.to_string().len() as isize <= INDENT);
    }
    false
}
fn is_blocklike(expr: &Expr) -> bool {
    match expr {
        Expr::Array(expr::Array { attrs, .. })
        | Expr::Async(expr::Async { attrs, .. })
        | Expr::Block(expr::Block { attrs, .. })
        | Expr::Closure(expr::Closure { attrs, .. })
        | Expr::Const(expr::Const { attrs, .. })
        | Expr::Struct(expr::Struct { attrs, .. })
        | Expr::TryBlock(expr::TryBlock { attrs, .. })
        | Expr::Tuple(ExprTuple { attrs, .. })
        | Expr::Unsafe(expr::Unsafe { attrs, .. }) => !has_outer(attrs),
        Expr::Assign(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Continue(_)
        | Expr::Field(_)
        | Expr::ForLoop(_)
        | Expr::Group(_)
        | Expr::If(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Try(_)
        | Expr::Unary(_)
        | Expr::Verbatim(_)
        | Expr::While(_)
        | Expr::Yield(_) => false,
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}
fn parseable_as_stmt(expr: &Expr) -> bool {
    match expr {
        Expr::Array(_)
        | Expr::Async(_)
        | Expr::Block(_)
        | Expr::Break(_)
        | Expr::Closure(_)
        | Expr::Const(_)
        | Expr::Continue(_)
        | Expr::ForLoop(_)
        | Expr::If(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Struct(_)
        | Expr::TryBlock(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Unsafe(_)
        | Expr::Verbatim(_)
        | Expr::While(_)
        | Expr::Yield(_) => true,
        Expr::Assign(expr) => parseable_as_stmt(&expr.left),
        Expr::Await(expr) => parseable_as_stmt(&expr.base),
        Expr::Binary(expr) => requires_terminator(&expr.left) && parseable_as_stmt(&expr.left),
        Expr::Call(expr) => requires_terminator(&expr.func) && parseable_as_stmt(&expr.func),
        Expr::Cast(expr) => requires_terminator(&expr.expr) && parseable_as_stmt(&expr.expr),
        Expr::Field(expr) => parseable_as_stmt(&expr.base),
        Expr::Group(expr) => parseable_as_stmt(&expr.expr),
        Expr::Index(expr) => requires_terminator(&expr.expr) && parseable_as_stmt(&expr.expr),
        Expr::MethodCall(expr) => parseable_as_stmt(&expr.receiver),
        Expr::Range(expr) => match &expr.start {
            None => true,
            Some(start) => requires_terminator(start) && parseable_as_stmt(start),
        },
        Expr::Try(expr) => parseable_as_stmt(&expr.expr),
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}

impl Printer {
    //file
    pub fn file(&mut self, file: &File) {
        self.cbox(0);
        if let Some(shebang) = &file.shebang {
            self.word(shebang.clone());
            self.hardbreak();
        }
        self.inner_attrs(&file.attrs);
        for item in &file.items {
            self.item(item);
        }
        self.end();
    }
}

impl Printer {
    //generics
    pub fn generics(&mut self, generics: &Generics) {
        if generics.params.is_empty() {
            return;
        }
        self.word("<");
        self.cbox(0);
        self.zerobreak();
        #[derive(Ord, PartialOrd, Eq, PartialEq)]
        enum Group {
            First,
            Second,
        }
        fn group(param: &GenericParam) -> Group {
            match param {
                GenericParam::Lifetime(_) => Group::First,
                GenericParam::Type(_) | GenericParam::Const(_) => Group::Second,
            }
        }
        let last = generics.params.iter().max_by_key(|param| group(param));
        for current_group in [Group::First, Group::Second] {
            for param in &generics.params {
                if group(param) == current_group {
                    self.generic_param(param);
                    self.trailing_comma(ptr::eq(param, last.unwrap()));
                }
            }
        }
        self.offset(-INDENT);
        self.end();
        self.word(">");
    }
    fn generic_param(&mut self, generic_param: &GenericParam) {
        match generic_param {
            GenericParam::Type(type_param) => self.type_param(type_param),
            GenericParam::Lifetime(lifetime_param) => self.lifetime_param(lifetime_param),
            GenericParam::Const(const_param) => self.const_param(const_param),
        }
    }
    pub fn bound_lifetimes(&mut self, bound_lifetimes: &BoundLifetimes) {
        self.word("for<");
        for param in bound_lifetimes.lifetimes.iter().delimited() {
            self.generic_param(&param);
            if !param.is_last {
                self.word(", ");
            }
        }
        self.word("> ");
    }
    fn lifetime_param(&mut self, lifetime_param: &LifetimeParam) {
        self.outer_attrs(&lifetime_param.attrs);
        self.lifetime(&lifetime_param.lifetime);
        for lifetime in lifetime_param.bounds.iter().delimited() {
            if lifetime.is_first {
                self.word(": ");
            } else {
                self.word(" + ");
            }
            self.lifetime(&lifetime);
        }
    }
    fn type_param(&mut self, type_param: &TypeParam) {
        self.outer_attrs(&type_param.attrs);
        self.ident(&type_param.ident);
        self.ibox(INDENT);
        for type_param_bound in type_param.bounds.iter().delimited() {
            if type_param_bound.is_first {
                self.word(": ");
            } else {
                self.space();
                self.word("+ ");
            }
            self.type_param_bound(&type_param_bound);
        }
        if let Some(default) = &type_param.default {
            self.space();
            self.word("= ");
            self.ty(default);
        }
        self.end();
    }
    pub fn type_param_bound(&mut self, type_param_bound: &TypeParamBound) {
        match type_param_bound {
            TypeParamBound::Trait(trait_bound) => {
                let tilde_const = false;
                self.trait_bound(trait_bound, tilde_const);
            },
            TypeParamBound::Lifetime(lifetime) => self.lifetime(lifetime),
            TypeParamBound::Verbatim(bound) => self.type_param_bound_verbatim(bound),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown TypeParamBound"),
        }
    }
    fn trait_bound(&mut self, trait_bound: &TraitBound, tilde_const: bool) {
        if trait_bound.paren_token.is_some() {
            self.word("(");
        }
        if tilde_const {
            self.word("~const ");
        }
        self.trait_bound_modifier(&trait_bound.modifier);
        if let Some(bound_lifetimes) = &trait_bound.lifetimes {
            self.bound_lifetimes(bound_lifetimes);
        }
        for segment in trait_bound.path.segments.iter().delimited() {
            if !segment.is_first || trait_bound.path.leading_colon.is_some() {
                self.word("::");
            }
            self.path_segment(&segment, PathKind::Type);
        }
        if trait_bound.paren_token.is_some() {
            self.word(")");
        }
    }
    fn trait_bound_modifier(&mut self, trait_bound_modifier: &TraitBoundModifier) {
        match trait_bound_modifier {
            TraitBoundModifier::None => {},
            TraitBoundModifier::Maybe(_question_mark) => self.word("?"),
        }
    }
    #[cfg(not(feature = "verbatim"))]
    fn type_param_bound_verbatim(&mut self, bound: &Stream) {
        unimplemented!("TypeParamBound::Verbatim `{}`", bound);
    }
    #[cfg(feature = "verbatim")]
    fn type_param_bound_verbatim(&mut self, tokens: &Stream) {
        enum TypeParamBoundVerbatim {
            Ellipsis,
            TildeConst(TraitBound),
        }
        impl parse::Parse for TypeParamBoundVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                let content;
                let (paren_token, content) = if input.peek(tok::Paren) {
                    (Some(parenthesized!(content in input)), &content)
                } else {
                    (None, input)
                };
                let lookahead = content.lookahead1();
                if lookahead.peek(Token![~]) {
                    content.parse::<Token![~]>()?;
                    content.parse::<Token![const]>()?;
                    let mut bound: TraitBound = content.parse()?;
                    bound.paren_token = paren_token;
                    Ok(TypeParamBoundVerbatim::TildeConst(bound))
                } else if lookahead.peek(Token![...]) {
                    content.parse::<Token![...]>()?;
                    Ok(TypeParamBoundVerbatim::Ellipsis)
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let bound: TypeParamBoundVerbatim = match syn::parse2(tokens.clone()) {
            Ok(bound) => bound,
            Err(_) => unimplemented!("TypeParamBound::Verbatim `{}`", tokens),
        };
        match bound {
            TypeParamBoundVerbatim::Ellipsis => {
                self.word("...");
            },
            TypeParamBoundVerbatim::TildeConst(trait_bound) => {
                let tilde_const = true;
                self.trait_bound(&trait_bound, tilde_const);
            },
        }
    }
    fn const_param(&mut self, const_param: &ConstParam) {
        self.outer_attrs(&const_param.attrs);
        self.word("const ");
        self.ident(&const_param.ident);
        self.word(": ");
        self.ty(&const_param.ty);
        if let Some(default) = &const_param.default {
            self.word(" = ");
            self.expr(default);
        }
    }
    pub fn where_clause_for_body(&mut self, where_clause: &Option<WhereClause>) {
        let hardbreaks = true;
        let semi = false;
        self.where_clause_impl(where_clause, hardbreaks, semi);
    }
    pub fn where_clause_semi(&mut self, where_clause: &Option<WhereClause>) {
        let hardbreaks = true;
        let semi = true;
        self.where_clause_impl(where_clause, hardbreaks, semi);
    }
    pub fn where_clause_oneline(&mut self, where_clause: &Option<WhereClause>) {
        let hardbreaks = false;
        let semi = false;
        self.where_clause_impl(where_clause, hardbreaks, semi);
    }
    pub fn where_clause_oneline_semi(&mut self, where_clause: &Option<WhereClause>) {
        let hardbreaks = false;
        let semi = true;
        self.where_clause_impl(where_clause, hardbreaks, semi);
    }
    fn where_clause_impl(&mut self, where_clause: &Option<WhereClause>, hardbreaks: bool, semi: bool) {
        let where_clause = match where_clause {
            Some(where_clause) if !where_clause.predicates.is_empty() => where_clause,
            _ => {
                if semi {
                    self.word(";");
                } else {
                    self.nbsp();
                }
                return;
            },
        };
        if hardbreaks {
            self.hardbreak();
            self.offset(-INDENT);
            self.word("where");
            self.hardbreak();
            for predicate in where_clause.predicates.iter().delimited() {
                self.where_predicate(&predicate);
                if predicate.is_last && semi {
                    self.word(";");
                } else {
                    self.word(",");
                    self.hardbreak();
                }
            }
            if !semi {
                self.offset(-INDENT);
            }
        } else {
            self.space();
            self.offset(-INDENT);
            self.word("where");
            self.space();
            for predicate in where_clause.predicates.iter().delimited() {
                self.where_predicate(&predicate);
                if predicate.is_last && semi {
                    self.word(";");
                } else {
                    self.trailing_comma_or_space(predicate.is_last);
                }
            }
            if !semi {
                self.offset(-INDENT);
            }
        }
    }
    fn where_predicate(&mut self, predicate: &WherePredicate) {
        match predicate {
            WherePredicate::Type(predicate) => self.predicate_type(predicate),
            WherePredicate::Lifetime(predicate) => self.predicate_lifetime(predicate),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown WherePredicate"),
        }
    }
    fn predicate_type(&mut self, predicate: &PredicateType) {
        if let Some(bound_lifetimes) = &predicate.lifetimes {
            self.bound_lifetimes(bound_lifetimes);
        }
        self.ty(&predicate.bounded_ty);
        self.word(":");
        if predicate.bounds.len() == 1 {
            self.ibox(0);
        } else {
            self.ibox(INDENT);
        }
        for type_param_bound in predicate.bounds.iter().delimited() {
            if type_param_bound.is_first {
                self.nbsp();
            } else {
                self.space();
                self.word("+ ");
            }
            self.type_param_bound(&type_param_bound);
        }
        self.end();
    }
    fn predicate_lifetime(&mut self, predicate: &PredicateLifetime) {
        self.lifetime(&predicate.lifetime);
        self.word(":");
        self.ibox(INDENT);
        for lifetime in predicate.bounds.iter().delimited() {
            if lifetime.is_first {
                self.nbsp();
            } else {
                self.space();
                self.word("+ ");
            }
            self.lifetime(&lifetime);
        }
        self.end();
    }
}

impl Printer {
    //item
    pub fn item(&mut self, item: &Item) {
        match item {
            Item::Const(item) => self.item_const(item),
            Item::Enum(item) => self.item_enum(item),
            Item::ExternCrate(item) => self.item_extern_crate(item),
            Item::Fn(item) => self.item_fn(item),
            Item::ForeignMod(item) => self.item_foreign_mod(item),
            Item::Impl(item) => self.item_impl(item),
            Item::Macro(item) => self.item_macro(item),
            Item::Mod(item) => self.item_mod(item),
            Item::Static(item) => self.item_static(item),
            Item::Struct(item) => self.item_struct(item),
            Item::Trait(item) => self.item_trait(item),
            Item::TraitAlias(item) => self.item_trait_alias(item),
            Item::Type(item) => self.item_type(item),
            Item::Union(item) => self.item_union(item),
            Item::Use(item) => self.item_use(item),
            Item::Verbatim(item) => self.item_verbatim(item),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown Item"),
        }
    }
    fn item_const(&mut self, item: &ItemConst) {
        self.outer_attrs(&item.attrs);
        self.cbox(0);
        self.visibility(&item.vis);
        self.word("const ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        self.word(": ");
        self.ty(&item.ty);
        self.word(" = ");
        self.neverbreak();
        self.expr(&item.expr);
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn item_enum(&mut self, item: &ItemEnum) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.word("enum ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        self.where_clause_for_body(&item.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        for variant in &item.variants {
            self.variant(variant);
            self.word(",");
            self.hardbreak();
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_extern_crate(&mut self, item: &ItemExternCrate) {
        self.outer_attrs(&item.attrs);
        self.visibility(&item.vis);
        self.word("extern crate ");
        self.ident(&item.ident);
        if let Some((_as_token, rename)) = &item.rename {
            self.word(" as ");
            self.ident(rename);
        }
        self.word(";");
        self.hardbreak();
    }
    fn item_fn(&mut self, item: &ItemFn) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.signature(&item.sig);
        self.where_clause_for_body(&item.sig.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        self.inner_attrs(&item.attrs);
        for stmt in &item.block.stmts {
            self.stmt(stmt);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_foreign_mod(&mut self, item: &ItemForeignMod) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        if item.unsafety.is_some() {
            self.word("unsafe ");
        }
        self.abi(&item.abi);
        self.word("{");
        self.hardbreak_if_nonempty();
        self.inner_attrs(&item.attrs);
        for foreign_item in &item.items {
            self.foreign_item(foreign_item);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_impl(&mut self, item: &ItemImpl) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.ibox(-INDENT);
        self.cbox(INDENT);
        if item.defaultness.is_some() {
            self.word("default ");
        }
        if item.unsafety.is_some() {
            self.word("unsafe ");
        }
        self.word("impl");
        self.generics(&item.generics);
        self.end();
        self.nbsp();
        if let Some((negative_polarity, path, _for_token)) = &item.trait_ {
            if negative_polarity.is_some() {
                self.word("!");
            }
            self.path(path, PathKind::Type);
            self.space();
            self.word("for ");
        }
        self.ty(&item.self_ty);
        self.end();
        self.where_clause_for_body(&item.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        self.inner_attrs(&item.attrs);
        for impl_item in &item.items {
            self.impl_item(impl_item);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_macro(&mut self, item: &ItemMacro) {
        self.outer_attrs(&item.attrs);
        let semicolon = true;
        self.mac(&item.mac, item.ident.as_ref(), semicolon);
        self.hardbreak();
    }
    fn item_mod(&mut self, item: &ItemMod) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        if item.unsafety.is_some() {
            self.word("unsafe ");
        }
        self.word("mod ");
        self.ident(&item.ident);
        if let Some((_brace, items)) = &item.content {
            self.word(" {");
            self.hardbreak_if_nonempty();
            self.inner_attrs(&item.attrs);
            for item in items {
                self.item(item);
            }
            self.offset(-INDENT);
            self.end();
            self.word("}");
        } else {
            self.word(";");
            self.end();
        }
        self.hardbreak();
    }
    fn item_static(&mut self, item: &ItemStatic) {
        self.outer_attrs(&item.attrs);
        self.cbox(0);
        self.visibility(&item.vis);
        self.word("static ");
        self.static_mutability(&item.mutability);
        self.ident(&item.ident);
        self.word(": ");
        self.ty(&item.ty);
        self.word(" = ");
        self.neverbreak();
        self.expr(&item.expr);
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn item_struct(&mut self, item: &ItemStruct) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.word("struct ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        match &item.fields {
            Fields::Named(fields) => {
                self.where_clause_for_body(&item.generics.where_clause);
                self.word("{");
                self.hardbreak_if_nonempty();
                for field in &fields.named {
                    self.field(field);
                    self.word(",");
                    self.hardbreak();
                }
                self.offset(-INDENT);
                self.end();
                self.word("}");
            },
            Fields::Unnamed(fields) => {
                self.fields_unnamed(fields);
                self.where_clause_semi(&item.generics.where_clause);
                self.end();
            },
            Fields::Unit => {
                self.where_clause_semi(&item.generics.where_clause);
                self.end();
            },
        }
        self.hardbreak();
    }
    fn item_trait(&mut self, item: &ItemTrait) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        if item.unsafety.is_some() {
            self.word("unsafe ");
        }
        if item.auto_token.is_some() {
            self.word("auto ");
        }
        self.word("trait ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        for supertrait in item.supertraits.iter().delimited() {
            if supertrait.is_first {
                self.word(": ");
            } else {
                self.word(" + ");
            }
            self.type_param_bound(&supertrait);
        }
        self.where_clause_for_body(&item.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        self.inner_attrs(&item.attrs);
        for trait_item in &item.items {
            self.trait_item(trait_item);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_trait_alias(&mut self, item: &ItemTraitAlias) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.word("trait ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        self.word(" = ");
        self.neverbreak();
        for bound in item.bounds.iter().delimited() {
            if !bound.is_first {
                self.space();
                self.word("+ ");
            }
            self.type_param_bound(&bound);
        }
        self.where_clause_semi(&item.generics.where_clause);
        self.end();
        self.hardbreak();
    }
    fn item_type(&mut self, item: &ItemType) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.word("type ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        self.where_clause_oneline(&item.generics.where_clause);
        self.word("= ");
        self.neverbreak();
        self.ibox(-INDENT);
        self.ty(&item.ty);
        self.end();
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn item_union(&mut self, item: &ItemUnion) {
        self.outer_attrs(&item.attrs);
        self.cbox(INDENT);
        self.visibility(&item.vis);
        self.word("union ");
        self.ident(&item.ident);
        self.generics(&item.generics);
        self.where_clause_for_body(&item.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        for field in &item.fields.named {
            self.field(field);
            self.word(",");
            self.hardbreak();
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn item_use(&mut self, item: &ItemUse) {
        self.outer_attrs(&item.attrs);
        self.visibility(&item.vis);
        self.word("use ");
        if item.leading_colon.is_some() {
            self.word("::");
        }
        self.use_tree(&item.tree);
        self.word(";");
        self.hardbreak();
    }
    #[cfg(not(feature = "verbatim"))]
    fn item_verbatim(&mut self, item: &Stream) {
        if !item.is_empty() {
            unimplemented!("Item::Verbatim `{}`", item);
        }
        self.hardbreak();
    }
    #[cfg(feature = "verbatim")]
    fn item_verbatim(&mut self, tokens: &Stream) {
        use verbatim::{FlexibleItemConst, FlexibleItemFn, FlexibleItemStatic, FlexibleItemType, WhereClauseLocation};
        enum ItemVerbatim {
            Empty,
            Ellipsis,
            ConstFlexible(FlexibleItemConst),
            FnFlexible(FlexibleItemFn),
            ImplFlexible(ImplFlexible),
            Macro2(Macro2),
            StaticFlexible(FlexibleItemStatic),
            TypeFlexible(FlexibleItemType),
            UseBrace(UseBrace),
        }
        struct ImplFlexible {
            attrs: Vec<attr::Attr>,
            vis: Visibility,
            defaultness: bool,
            unsafety: bool,
            generics: Generics,
            constness: ImplConstness,
            negative_impl: bool,
            trait_: Option<Type>,
            self_ty: Type,
            items: Vec<ImplItem>,
        }
        enum ImplConstness {
            None,
            MaybeConst,
            Const,
        }
        struct Macro2 {
            attrs: Vec<attr::Attr>,
            vis: Visibility,
            ident: Ident,
            args: Option<Stream>,
            body: Stream,
        }
        struct UseBrace {
            attrs: Vec<attr::Attr>,
            vis: Visibility,
            trees: punct::Puncted<RootUseTree, Token![,]>,
        }
        struct RootUseTree {
            leading_colon: Option<Token![::]>,
            inner: UseTree,
        }
        impl parse::Parse for ImplConstness {
            fn parse(input: parse::Stream) -> Res<Self> {
                if input.parse::<Option<Token![?]>>()?.is_some() {
                    input.parse::<Token![const]>()?;
                    Ok(ImplConstness::MaybeConst)
                } else if input.parse::<Option<Token![const]>>()?.is_some() {
                    Ok(ImplConstness::Const)
                } else {
                    Ok(ImplConstness::None)
                }
            }
        }
        impl parse::Parse for RootUseTree {
            fn parse(input: parse::Stream) -> Res<Self> {
                Ok(RootUseTree {
                    leading_colon: input.parse()?,
                    inner: input.parse()?,
                })
            }
        }
        impl parse::Parse for ItemVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                if input.is_empty() {
                    return Ok(ItemVerbatim::Empty);
                } else if input.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    return Ok(ItemVerbatim::Ellipsis);
                }
                let mut attrs = input.call(attr::Attr::parse_outer)?;
                let vis: Visibility = input.parse()?;
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![const]) && (input.peek2(Ident) || input.peek2(Token![_])) {
                    let defaultness = false;
                    let flexible_item = FlexibleItemConst::parse(attrs, vis, defaultness, input)?;
                    Ok(ItemVerbatim::ConstFlexible(flexible_item))
                } else if input.peek(Token![const])
                    || lookahead.peek(Token![async])
                    || lookahead.peek(Token![unsafe]) && !input.peek2(Token![impl])
                    || lookahead.peek(Token![extern])
                    || lookahead.peek(Token![fn])
                {
                    let defaultness = false;
                    let flexible_item = FlexibleItemFn::parse(attrs, vis, defaultness, input)?;
                    Ok(ItemVerbatim::FnFlexible(flexible_item))
                } else if lookahead.peek(Token![default]) || input.peek(Token![unsafe]) || lookahead.peek(Token![impl])
                {
                    let defaultness = input.parse::<Option<Token![default]>>()?.is_some();
                    let unsafety = input.parse::<Option<Token![unsafe]>>()?.is_some();
                    input.parse::<Token![impl]>()?;
                    let has_generics = input.peek(Token![<])
                        && (input.peek2(Token![>])
                            || input.peek2(Token![#])
                            || (input.peek2(Ident) || input.peek2(Lifetime))
                                && (input.peek3(Token![:])
                                    || input.peek3(Token![,])
                                    || input.peek3(Token![>])
                                    || input.peek3(Token![=]))
                            || input.peek2(Token![const]));
                    let mut generics: Generics = if has_generics {
                        input.parse()?
                    } else {
                        Generics::default()
                    };
                    let constness: ImplConstness = input.parse()?;
                    let negative_impl = !input.peek2(tok::Brace) && input.parse::<Option<Token![!]>>()?.is_some();
                    let first_ty: Type = input.parse()?;
                    let (trait_, self_ty) = if input.parse::<Option<Token![for]>>()?.is_some() {
                        (Some(first_ty), input.parse()?)
                    } else {
                        (None, first_ty)
                    };
                    generics.where_clause = input.parse()?;
                    let content;
                    braced!(content in input);
                    let inner_attrs = content.call(attr::Attr::parse_inner)?;
                    attrs.extend(inner_attrs);
                    let mut items = Vec::new();
                    while !content.is_empty() {
                        items.push(content.parse()?);
                    }
                    Ok(ItemVerbatim::ImplFlexible(ImplFlexible {
                        attrs,
                        vis,
                        defaultness,
                        unsafety,
                        generics,
                        constness,
                        negative_impl,
                        trait_,
                        self_ty,
                        items,
                    }))
                } else if lookahead.peek(Token![macro]) {
                    input.parse::<Token![macro]>()?;
                    let ident: Ident = input.parse()?;
                    let args = if input.peek(tok::Paren) {
                        let paren_content;
                        parenthesized!(paren_content in input);
                        Some(paren_content.parse::<Stream>()?)
                    } else {
                        None
                    };
                    let brace_content;
                    braced!(brace_content in input);
                    let body: Stream = brace_content.parse()?;
                    Ok(ItemVerbatim::Macro2(Macro2 {
                        attrs,
                        vis,
                        ident,
                        args,
                        body,
                    }))
                } else if lookahead.peek(Token![static]) {
                    let flexible_item = FlexibleItemStatic::parse(attrs, vis, input)?;
                    Ok(ItemVerbatim::StaticFlexible(flexible_item))
                } else if lookahead.peek(Token![type]) {
                    let defaultness = false;
                    let flexible_item =
                        FlexibleItemType::parse(attrs, vis, defaultness, input, WhereClauseLocation::BeforeEq)?;
                    Ok(ItemVerbatim::TypeFlexible(flexible_item))
                } else if lookahead.peek(Token![use]) {
                    input.parse::<Token![use]>()?;
                    let content;
                    braced!(content in input);
                    let trees = content.parse_terminated(RootUseTree::parse, Token![,])?;
                    input.parse::<Token![;]>()?;
                    Ok(ItemVerbatim::UseBrace(UseBrace { attrs, vis, trees }))
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let item: ItemVerbatim = match syn::parse2(tokens.clone()) {
            Ok(item) => item,
            Err(_) => unimplemented!("Item::Verbatim `{}`", tokens),
        };
        match item {
            ItemVerbatim::Empty => {
                self.hardbreak();
            },
            ItemVerbatim::Ellipsis => {
                self.word("...");
                self.hardbreak();
            },
            ItemVerbatim::ConstFlexible(item) => {
                self.flexible_item_const(&item);
            },
            ItemVerbatim::FnFlexible(item) => {
                self.flexible_item_fn(&item);
            },
            ItemVerbatim::ImplFlexible(item) => {
                self.outer_attrs(&item.attrs);
                self.cbox(INDENT);
                self.ibox(-INDENT);
                self.cbox(INDENT);
                self.visibility(&item.vis);
                if item.defaultness {
                    self.word("default ");
                }
                if item.unsafety {
                    self.word("unsafe ");
                }
                self.word("impl");
                self.generics(&item.generics);
                self.end();
                self.nbsp();
                match item.constness {
                    ImplConstness::None => {},
                    ImplConstness::MaybeConst => self.word("?const "),
                    ImplConstness::Const => self.word("const "),
                }
                if item.negative_impl {
                    self.word("!");
                }
                if let Some(trait_) = &item.trait_ {
                    self.ty(trait_);
                    self.space();
                    self.word("for ");
                }
                self.ty(&item.self_ty);
                self.end();
                self.where_clause_for_body(&item.generics.where_clause);
                self.word("{");
                self.hardbreak_if_nonempty();
                self.inner_attrs(&item.attrs);
                for impl_item in &item.items {
                    self.impl_item(impl_item);
                }
                self.offset(-INDENT);
                self.end();
                self.word("}");
                self.hardbreak();
            },
            ItemVerbatim::Macro2(item) => {
                self.outer_attrs(&item.attrs);
                self.visibility(&item.vis);
                self.word("macro ");
                self.ident(&item.ident);
                if let Some(args) = &item.args {
                    self.word("(");
                    self.cbox(INDENT);
                    self.zerobreak();
                    self.ibox(0);
                    self.macro_rules_tokens(args.clone(), true);
                    self.end();
                    self.zerobreak();
                    self.offset(-INDENT);
                    self.end();
                    self.word(")");
                }
                self.word(" {");
                if !item.body.is_empty() {
                    self.neverbreak();
                    self.cbox(INDENT);
                    self.hardbreak();
                    self.ibox(0);
                    self.macro_rules_tokens(item.body.clone(), false);
                    self.end();
                    self.hardbreak();
                    self.offset(-INDENT);
                    self.end();
                }
                self.word("}");
                self.hardbreak();
            },
            ItemVerbatim::StaticFlexible(item) => {
                self.flexible_item_static(&item);
            },
            ItemVerbatim::TypeFlexible(item) => {
                self.flexible_item_type(&item);
            },
            ItemVerbatim::UseBrace(item) => {
                self.outer_attrs(&item.attrs);
                self.visibility(&item.vis);
                self.word("use ");
                if item.trees.len() == 1 {
                    self.word("::");
                    self.use_tree(&item.trees[0].inner);
                } else {
                    self.cbox(INDENT);
                    self.word("{");
                    self.zerobreak();
                    self.ibox(0);
                    for use_tree in item.trees.iter().delimited() {
                        if use_tree.leading_colon.is_some() {
                            self.word("::");
                        }
                        self.use_tree(&use_tree.inner);
                        if !use_tree.is_last {
                            self.word(",");
                            let mut use_tree = &use_tree.inner;
                            while let UseTree::Path(use_path) = use_tree {
                                use_tree = &use_path.tree;
                            }
                            if let UseTree::Group(_) = use_tree {
                                self.hardbreak();
                            } else {
                                self.space();
                            }
                        }
                    }
                    self.end();
                    self.trailing_comma(true);
                    self.offset(-INDENT);
                    self.word("}");
                    self.end();
                }
                self.word(";");
                self.hardbreak();
            },
        }
    }
    fn use_tree(&mut self, use_tree: &UseTree) {
        match use_tree {
            UseTree::Path(use_path) => self.use_path(use_path),
            UseTree::Name(use_name) => self.use_name(use_name),
            UseTree::Rename(use_rename) => self.use_rename(use_rename),
            UseTree::Glob(use_glob) => self.use_glob(use_glob),
            UseTree::Group(use_group) => self.use_group(use_group),
        }
    }
    fn use_path(&mut self, use_path: &UsePath) {
        self.ident(&use_path.ident);
        self.word("::");
        self.use_tree(&use_path.tree);
    }
    fn use_name(&mut self, use_name: &UseName) {
        self.ident(&use_name.ident);
    }
    fn use_rename(&mut self, use_rename: &UseRename) {
        self.ident(&use_rename.ident);
        self.word(" as ");
        self.ident(&use_rename.rename);
    }
    fn use_glob(&mut self, use_glob: &UseGlob) {
        let _ = use_glob;
        self.word("*");
    }
    fn use_group(&mut self, use_group: &UseGroup) {
        if use_group.items.is_empty() {
            self.word("{}");
        } else if use_group.items.len() == 1 {
            self.use_tree(&use_group.items[0]);
        } else {
            self.cbox(INDENT);
            self.word("{");
            self.zerobreak();
            self.ibox(0);
            for use_tree in use_group.items.iter().delimited() {
                self.use_tree(&use_tree);
                if !use_tree.is_last {
                    self.word(",");
                    let mut use_tree = *use_tree;
                    while let UseTree::Path(use_path) = use_tree {
                        use_tree = &use_path.tree;
                    }
                    if let UseTree::Group(_) = use_tree {
                        self.hardbreak();
                    } else {
                        self.space();
                    }
                }
            }
            self.end();
            self.trailing_comma(true);
            self.offset(-INDENT);
            self.word("}");
            self.end();
        }
    }
    fn foreign_item(&mut self, foreign_item: &ForeignItem) {
        match foreign_item {
            ForeignItem::Fn(item) => self.foreign_item_fn(item),
            ForeignItem::Static(item) => self.foreign_item_static(item),
            ForeignItem::Type(item) => self.foreign_item_type(item),
            ForeignItem::Macro(item) => self.foreign_item_macro(item),
            ForeignItem::Verbatim(item) => self.foreign_item_verbatim(item),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown ForeignItem"),
        }
    }
    fn foreign_item_fn(&mut self, foreign_item: &ForeignItemFn) {
        self.outer_attrs(&foreign_item.attrs);
        self.cbox(INDENT);
        self.visibility(&foreign_item.vis);
        self.signature(&foreign_item.sig);
        self.where_clause_semi(&foreign_item.sig.generics.where_clause);
        self.end();
        self.hardbreak();
    }
    fn foreign_item_static(&mut self, foreign_item: &ForeignItemStatic) {
        self.outer_attrs(&foreign_item.attrs);
        self.cbox(0);
        self.visibility(&foreign_item.vis);
        self.word("static ");
        self.static_mutability(&foreign_item.mutability);
        self.ident(&foreign_item.ident);
        self.word(": ");
        self.ty(&foreign_item.ty);
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn foreign_item_type(&mut self, foreign_item: &ForeignItemType) {
        self.outer_attrs(&foreign_item.attrs);
        self.cbox(0);
        self.visibility(&foreign_item.vis);
        self.word("type ");
        self.ident(&foreign_item.ident);
        self.generics(&foreign_item.generics);
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn foreign_item_macro(&mut self, foreign_item: &ForeignItemMacro) {
        self.outer_attrs(&foreign_item.attrs);
        let semicolon = true;
        self.mac(&foreign_item.mac, None, semicolon);
        self.hardbreak();
    }
    #[cfg(not(feature = "verbatim"))]
    fn foreign_item_verbatim(&mut self, foreign_item: &Stream) {
        if !foreign_item.is_empty() {
            unimplemented!("ForeignItem::Verbatim `{}`", foreign_item);
        }
        self.hardbreak();
    }
    #[cfg(feature = "verbatim")]
    fn foreign_item_verbatim(&mut self, tokens: &Stream) {
        use verbatim::{FlexibleItemFn, FlexibleItemStatic, FlexibleItemType, WhereClauseLocation};
        enum ForeignItemVerbatim {
            Empty,
            Ellipsis,
            FnFlexible(FlexibleItemFn),
            StaticFlexible(FlexibleItemStatic),
            TypeFlexible(FlexibleItemType),
        }
        impl parse::Parse for ForeignItemVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                if input.is_empty() {
                    return Ok(ForeignItemVerbatim::Empty);
                } else if input.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    return Ok(ForeignItemVerbatim::Ellipsis);
                }
                let attrs = input.call(attr::Attr::parse_outer)?;
                let vis: Visibility = input.parse()?;
                let defaultness = false;
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![const])
                    || lookahead.peek(Token![async])
                    || lookahead.peek(Token![unsafe])
                    || lookahead.peek(Token![extern])
                    || lookahead.peek(Token![fn])
                {
                    let flexible_item = FlexibleItemFn::parse(attrs, vis, defaultness, input)?;
                    Ok(ForeignItemVerbatim::FnFlexible(flexible_item))
                } else if lookahead.peek(Token![static]) {
                    let flexible_item = FlexibleItemStatic::parse(attrs, vis, input)?;
                    Ok(ForeignItemVerbatim::StaticFlexible(flexible_item))
                } else if lookahead.peek(Token![type]) {
                    let flexible_item =
                        FlexibleItemType::parse(attrs, vis, defaultness, input, WhereClauseLocation::Both)?;
                    Ok(ForeignItemVerbatim::TypeFlexible(flexible_item))
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let foreign_item: ForeignItemVerbatim = match syn::parse2(tokens.clone()) {
            Ok(foreign_item) => foreign_item,
            Err(_) => unimplemented!("ForeignItem::Verbatim `{}`", tokens),
        };
        match foreign_item {
            ForeignItemVerbatim::Empty => {
                self.hardbreak();
            },
            ForeignItemVerbatim::Ellipsis => {
                self.word("...");
                self.hardbreak();
            },
            ForeignItemVerbatim::FnFlexible(foreign_item) => {
                self.flexible_item_fn(&foreign_item);
            },
            ForeignItemVerbatim::StaticFlexible(foreign_item) => {
                self.flexible_item_static(&foreign_item);
            },
            ForeignItemVerbatim::TypeFlexible(foreign_item) => {
                self.flexible_item_type(&foreign_item);
            },
        }
    }
    fn trait_item(&mut self, trait_item: &TraitItem) {
        match trait_item {
            TraitItem::Const(item) => self.trait_item_const(item),
            TraitItem::Fn(item) => self.trait_item_fn(item),
            TraitItem::Type(item) => self.trait_item_type(item),
            TraitItem::Macro(item) => self.trait_item_macro(item),
            TraitItem::Verbatim(item) => self.trait_item_verbatim(item),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown TraitItem"),
        }
    }
    fn trait_item_const(&mut self, trait_item: &TraitItemConst) {
        self.outer_attrs(&trait_item.attrs);
        self.cbox(0);
        self.word("const ");
        self.ident(&trait_item.ident);
        self.generics(&trait_item.generics);
        self.word(": ");
        self.ty(&trait_item.ty);
        if let Some((_eq_token, default)) = &trait_item.default {
            self.word(" = ");
            self.neverbreak();
            self.expr(default);
        }
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn trait_item_fn(&mut self, trait_item: &TraitItemFn) {
        self.outer_attrs(&trait_item.attrs);
        self.cbox(INDENT);
        self.signature(&trait_item.sig);
        if let Some(block) = &trait_item.default {
            self.where_clause_for_body(&trait_item.sig.generics.where_clause);
            self.word("{");
            self.hardbreak_if_nonempty();
            self.inner_attrs(&trait_item.attrs);
            for stmt in &block.stmts {
                self.stmt(stmt);
            }
            self.offset(-INDENT);
            self.end();
            self.word("}");
        } else {
            self.where_clause_semi(&trait_item.sig.generics.where_clause);
            self.end();
        }
        self.hardbreak();
    }
    fn trait_item_type(&mut self, trait_item: &TraitItemType) {
        self.outer_attrs(&trait_item.attrs);
        self.cbox(INDENT);
        self.word("type ");
        self.ident(&trait_item.ident);
        self.generics(&trait_item.generics);
        for bound in trait_item.bounds.iter().delimited() {
            if bound.is_first {
                self.word(": ");
            } else {
                self.space();
                self.word("+ ");
            }
            self.type_param_bound(&bound);
        }
        if let Some((_eq_token, default)) = &trait_item.default {
            self.word(" = ");
            self.neverbreak();
            self.ibox(-INDENT);
            self.ty(default);
            self.end();
        }
        self.where_clause_oneline_semi(&trait_item.generics.where_clause);
        self.end();
        self.hardbreak();
    }
    fn trait_item_macro(&mut self, trait_item: &TraitItemMacro) {
        self.outer_attrs(&trait_item.attrs);
        let semicolon = true;
        self.mac(&trait_item.mac, None, semicolon);
        self.hardbreak();
    }
    #[cfg(not(feature = "verbatim"))]
    fn trait_item_verbatim(&mut self, trait_item: &Stream) {
        if !trait_item.is_empty() {
            unimplemented!("TraitItem::Verbatim `{}`", trait_item);
        }
        self.hardbreak();
    }
    #[cfg(feature = "verbatim")]
    fn trait_item_verbatim(&mut self, tokens: &Stream) {
        use verbatim::{FlexibleItemType, WhereClauseLocation};
        enum TraitItemVerbatim {
            Empty,
            Ellipsis,
            TypeFlexible(FlexibleItemType),
            PubOrDefault(PubOrDefaultTraitItem),
        }
        struct PubOrDefaultTraitItem {
            attrs: Vec<attr::Attr>,
            vis: Visibility,
            defaultness: bool,
            trait_item: TraitItem,
        }
        impl parse::Parse for TraitItemVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                if input.is_empty() {
                    return Ok(TraitItemVerbatim::Empty);
                } else if input.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    return Ok(TraitItemVerbatim::Ellipsis);
                }
                let attrs = input.call(attr::Attr::parse_outer)?;
                let vis: Visibility = input.parse()?;
                let defaultness = input.parse::<Option<Token![default]>>()?.is_some();
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![type]) {
                    let flexible_item =
                        FlexibleItemType::parse(attrs, vis, defaultness, input, WhereClauseLocation::AfterEq)?;
                    Ok(TraitItemVerbatim::TypeFlexible(flexible_item))
                } else if (lookahead.peek(Token![const])
                    || lookahead.peek(Token![async])
                    || lookahead.peek(Token![unsafe])
                    || lookahead.peek(Token![extern])
                    || lookahead.peek(Token![fn]))
                    && (!matches!(vis, Visibility::Inherited) || defaultness)
                {
                    Ok(TraitItemVerbatim::PubOrDefault(PubOrDefaultTraitItem {
                        attrs,
                        vis,
                        defaultness,
                        trait_item: input.parse()?,
                    }))
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let impl_item: TraitItemVerbatim = match syn::parse2(tokens.clone()) {
            Ok(impl_item) => impl_item,
            Err(_) => unimplemented!("TraitItem::Verbatim `{}`", tokens),
        };
        match impl_item {
            TraitItemVerbatim::Empty => {
                self.hardbreak();
            },
            TraitItemVerbatim::Ellipsis => {
                self.word("...");
                self.hardbreak();
            },
            TraitItemVerbatim::TypeFlexible(trait_item) => {
                self.flexible_item_type(&trait_item);
            },
            TraitItemVerbatim::PubOrDefault(trait_item) => {
                self.outer_attrs(&trait_item.attrs);
                self.visibility(&trait_item.vis);
                if trait_item.defaultness {
                    self.word("default ");
                }
                self.trait_item(&trait_item.trait_item);
            },
        }
    }
    fn impl_item(&mut self, impl_item: &ImplItem) {
        match impl_item {
            ImplItem::Const(item) => self.impl_item_const(item),
            ImplItem::Fn(item) => self.impl_item_fn(item),
            ImplItem::Type(item) => self.impl_item_type(item),
            ImplItem::Macro(item) => self.impl_item_macro(item),
            ImplItem::Verbatim(item) => self.impl_item_verbatim(item),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown ImplItem"),
        }
    }
    fn impl_item_const(&mut self, impl_item: &ImplItemConst) {
        self.outer_attrs(&impl_item.attrs);
        self.cbox(0);
        self.visibility(&impl_item.vis);
        if impl_item.defaultness.is_some() {
            self.word("default ");
        }
        self.word("const ");
        self.ident(&impl_item.ident);
        self.generics(&impl_item.generics);
        self.word(": ");
        self.ty(&impl_item.ty);
        self.word(" = ");
        self.neverbreak();
        self.expr(&impl_item.expr);
        self.word(";");
        self.end();
        self.hardbreak();
    }
    fn impl_item_fn(&mut self, impl_item: &ImplItemFn) {
        self.outer_attrs(&impl_item.attrs);
        self.cbox(INDENT);
        self.visibility(&impl_item.vis);
        if impl_item.defaultness.is_some() {
            self.word("default ");
        }
        self.signature(&impl_item.sig);
        self.where_clause_for_body(&impl_item.sig.generics.where_clause);
        self.word("{");
        self.hardbreak_if_nonempty();
        self.inner_attrs(&impl_item.attrs);
        for stmt in &impl_item.block.stmts {
            self.stmt(stmt);
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
        self.hardbreak();
    }
    fn impl_item_type(&mut self, impl_item: &ImplItemType) {
        self.outer_attrs(&impl_item.attrs);
        self.cbox(INDENT);
        self.visibility(&impl_item.vis);
        if impl_item.defaultness.is_some() {
            self.word("default ");
        }
        self.word("type ");
        self.ident(&impl_item.ident);
        self.generics(&impl_item.generics);
        self.word(" = ");
        self.neverbreak();
        self.ibox(-INDENT);
        self.ty(&impl_item.ty);
        self.end();
        self.where_clause_oneline_semi(&impl_item.generics.where_clause);
        self.end();
        self.hardbreak();
    }
    fn impl_item_macro(&mut self, impl_item: &ImplItemMacro) {
        self.outer_attrs(&impl_item.attrs);
        let semicolon = true;
        self.mac(&impl_item.mac, None, semicolon);
        self.hardbreak();
    }
    #[cfg(not(feature = "verbatim"))]
    fn impl_item_verbatim(&mut self, impl_item: &Stream) {
        if !impl_item.is_empty() {
            unimplemented!("ImplItem::Verbatim `{}`", impl_item);
        }
        self.hardbreak();
    }
    #[cfg(feature = "verbatim")]
    fn impl_item_verbatim(&mut self, tokens: &Stream) {
        use verbatim::{FlexibleItemConst, FlexibleItemFn, FlexibleItemType, WhereClauseLocation};
        enum ImplItemVerbatim {
            Empty,
            Ellipsis,
            ConstFlexible(FlexibleItemConst),
            FnFlexible(FlexibleItemFn),
            TypeFlexible(FlexibleItemType),
        }
        impl parse::Parse for ImplItemVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                if input.is_empty() {
                    return Ok(ImplItemVerbatim::Empty);
                } else if input.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    return Ok(ImplItemVerbatim::Ellipsis);
                }
                let attrs = input.call(attr::Attr::parse_outer)?;
                let vis: Visibility = input.parse()?;
                let defaultness = input.parse::<Option<Token![default]>>()?.is_some();
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![const]) && (input.peek2(Ident) || input.peek2(Token![_])) {
                    let flexible_item = FlexibleItemConst::parse(attrs, vis, defaultness, input)?;
                    Ok(ImplItemVerbatim::ConstFlexible(flexible_item))
                } else if input.peek(Token![const])
                    || lookahead.peek(Token![async])
                    || lookahead.peek(Token![unsafe])
                    || lookahead.peek(Token![extern])
                    || lookahead.peek(Token![fn])
                {
                    let flexible_item = FlexibleItemFn::parse(attrs, vis, defaultness, input)?;
                    Ok(ImplItemVerbatim::FnFlexible(flexible_item))
                } else if lookahead.peek(Token![type]) {
                    let flexible_item =
                        FlexibleItemType::parse(attrs, vis, defaultness, input, WhereClauseLocation::AfterEq)?;
                    Ok(ImplItemVerbatim::TypeFlexible(flexible_item))
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let impl_item: ImplItemVerbatim = match syn::parse2(tokens.clone()) {
            Ok(impl_item) => impl_item,
            Err(_) => unimplemented!("ImplItem::Verbatim `{}`", tokens),
        };
        match impl_item {
            ImplItemVerbatim::Empty => {
                self.hardbreak();
            },
            ImplItemVerbatim::Ellipsis => {
                self.word("...");
                self.hardbreak();
            },
            ImplItemVerbatim::ConstFlexible(impl_item) => {
                self.flexible_item_const(&impl_item);
            },
            ImplItemVerbatim::FnFlexible(impl_item) => {
                self.flexible_item_fn(&impl_item);
            },
            ImplItemVerbatim::TypeFlexible(impl_item) => {
                self.flexible_item_type(&impl_item);
            },
        }
    }
    fn signature(&mut self, signature: &Signature) {
        if signature.constness.is_some() {
            self.word("const ");
        }
        if signature.asyncness.is_some() {
            self.word("async ");
        }
        if signature.unsafety.is_some() {
            self.word("unsafe ");
        }
        if let Some(abi) = &signature.abi {
            self.abi(abi);
        }
        self.word("fn ");
        self.ident(&signature.ident);
        self.generics(&signature.generics);
        self.word("(");
        self.neverbreak();
        self.cbox(0);
        self.zerobreak();
        for input in signature.inputs.iter().delimited() {
            self.fn_arg(&input);
            let is_last = input.is_last && signature.variadic.is_none();
            self.trailing_comma(is_last);
        }
        if let Some(variadic) = &signature.variadic {
            self.variadic(variadic);
            self.zerobreak();
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
        self.cbox(-INDENT);
        self.return_type(&signature.output);
        self.end();
    }
    fn fn_arg(&mut self, fn_arg: &FnArg) {
        match fn_arg {
            FnArg::Receiver(receiver) => self.receiver(receiver),
            FnArg::Typed(pat_type) => self.pat_type(pat_type),
        }
    }
    fn receiver(&mut self, receiver: &Receiver) {
        self.outer_attrs(&receiver.attrs);
        if let Some((_ampersand, lifetime)) = &receiver.reference {
            self.word("&");
            if let Some(lifetime) = lifetime {
                self.lifetime(lifetime);
                self.nbsp();
            }
        }
        if receiver.mutability.is_some() {
            self.word("mut ");
        }
        self.word("self");
        if receiver.colon_token.is_some() {
            self.word(": ");
            self.ty(&receiver.ty);
        } else {
            let consistent = match (&receiver.reference, &receiver.mutability, &*receiver.ty) {
                (Some(_), mutability, item::Type::Reference(ty)) => {
                    mutability.is_some() == ty.mutability.is_some()
                        && match &*ty.elem {
                            item::Type::Path(ty) => ty.qself.is_none() && ty.path.is_ident("Self"),
                            _ => false,
                        }
                },
                (None, _, item::Type::Path(ty)) => ty.qself.is_none() && ty.path.is_ident("Self"),
                _ => false,
            };
            if !consistent {
                self.word(": ");
                self.ty(&receiver.ty);
            }
        }
    }
    fn variadic(&mut self, variadic: &item::Variadic) {
        self.outer_attrs(&variadic.attrs);
        if let Some((pat, _colon)) = &variadic.pat {
            self.pat(pat);
            self.word(": ");
        }
        self.word("...");
    }
    fn static_mutability(&mut self, mutability: &StaticMutability) {
        match mutability {
            StaticMutability::Mut(_) => self.word("mut "),
            StaticMutability::None => {},
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown StaticMutability"),
        }
    }
}
#[cfg(feature = "verbatim")]
mod verbatim {
    pub struct FlexibleItemConst {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub defaultness: bool,
        pub ident: Ident,
        pub ty: Type,
    }
    pub struct FlexibleItemFn {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub defaultness: bool,
        pub sig: Signature,
        pub body: Option<Vec<Stmt>>,
    }
    pub struct FlexibleItemStatic {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub mutability: StaticMutability,
        pub ident: Ident,
        pub ty: Option<Type>,
        pub expr: Option<Expr>,
    }
    pub struct FlexibleItemType {
        pub attrs: Vec<attr::Attr>,
        pub vis: Visibility,
        pub defaultness: bool,
        pub ident: Ident,
        pub generics: Generics,
        pub bounds: Vec<TypeParamBound>,
        pub definition: Option<Type>,
        pub where_clause_after_eq: Option<WhereClause>,
    }
    pub enum WhereClauseLocation {
        BeforeEq,
        AfterEq,
        Both,
    }
    impl FlexibleItemConst {
        pub fn parse(attrs: Vec<attr::Attr>, vis: Visibility, defaultness: bool, input: parse::Stream) -> Res<Self> {
            input.parse::<Token![const]>()?;
            let ident = input.call(Ident::parse_any)?;
            input.parse::<Token![:]>()?;
            let ty: Type = input.parse()?;
            input.parse::<Token![;]>()?;
            Ok(FlexibleItemConst {
                attrs,
                vis,
                defaultness,
                ident,
                ty,
            })
        }
    }
    impl FlexibleItemFn {
        pub fn parse(
            mut attrs: Vec<attr::Attr>,
            vis: Visibility,
            defaultness: bool,
            input: parse::Stream,
        ) -> Res<Self> {
            let sig: Signature = input.parse()?;
            let lookahead = input.lookahead1();
            let body = if lookahead.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                None
            } else if lookahead.peek(tok::Brace) {
                let content;
                braced!(content in input);
                attrs.extend(content.call(attr::Attr::parse_inner)?);
                Some(content.call(Block::parse_within)?)
            } else {
                return Err(lookahead.error());
            };
            Ok(FlexibleItemFn {
                attrs,
                vis,
                defaultness,
                sig,
                body,
            })
        }
    }
    impl FlexibleItemStatic {
        pub fn parse(attrs: Vec<attr::Attr>, vis: Visibility, input: parse::Stream) -> Res<Self> {
            input.parse::<Token![static]>()?;
            let mutability: StaticMutability = input.parse()?;
            let ident = input.parse()?;
            let lookahead = input.lookahead1();
            let has_type = lookahead.peek(Token![:]);
            let has_expr = lookahead.peek(Token![=]);
            if !has_type && !has_expr {
                return Err(lookahead.error());
            }
            let ty: Option<Type> = if has_type {
                input.parse::<Token![:]>()?;
                input.parse().map(Some)?
            } else {
                None
            };
            let expr: Option<Expr> = if input.parse::<Option<Token![=]>>()?.is_some() {
                input.parse().map(Some)?
            } else {
                None
            };
            input.parse::<Token![;]>()?;
            Ok(FlexibleItemStatic {
                attrs,
                vis,
                mutability,
                ident,
                ty,
                expr,
            })
        }
    }
    impl FlexibleItemType {
        pub fn parse(
            attrs: Vec<attr::Attr>,
            vis: Visibility,
            defaultness: bool,
            input: parse::Stream,
            where_clause_location: WhereClauseLocation,
        ) -> Res<Self> {
            input.parse::<Token![type]>()?;
            let ident: Ident = input.parse()?;
            let mut generics: Generics = input.parse()?;
            let mut bounds = Vec::new();
            if input.parse::<Option<Token![:]>>()?.is_some() {
                loop {
                    if input.peek(Token![where]) || input.peek(Token![=]) || input.peek(Token![;]) {
                        break;
                    }
                    bounds.push(input.parse::<TypeParamBound>()?);
                    if input.peek(Token![where]) || input.peek(Token![=]) || input.peek(Token![;]) {
                        break;
                    }
                    input.parse::<Token![+]>()?;
                }
            }
            match where_clause_location {
                WhereClauseLocation::BeforeEq | WhereClauseLocation::Both => {
                    generics.where_clause = input.parse()?;
                },
                WhereClauseLocation::AfterEq => {},
            }
            let definition = if input.parse::<Option<Token![=]>>()?.is_some() {
                Some(input.parse()?)
            } else {
                None
            };
            let where_clause_after_eq = match where_clause_location {
                WhereClauseLocation::AfterEq | WhereClauseLocation::Both if generics.where_clause.is_none() => {
                    input.parse()?
                },
                _ => None,
            };
            input.parse::<Token![;]>()?;
            Ok(FlexibleItemType {
                attrs,
                vis,
                defaultness,
                ident,
                generics,
                bounds,
                definition,
                where_clause_after_eq,
            })
        }
    }
    impl Printer {
        pub fn flexible_item_const(&mut self, item: &FlexibleItemConst) {
            self.outer_attrs(&item.attrs);
            self.cbox(0);
            self.visibility(&item.vis);
            if item.defaultness {
                self.word("default ");
            }
            self.word("const ");
            self.ident(&item.ident);
            self.word(": ");
            self.ty(&item.ty);
            self.word(";");
            self.end();
            self.hardbreak();
        }
        pub fn flexible_item_fn(&mut self, item: &FlexibleItemFn) {
            self.outer_attrs(&item.attrs);
            self.cbox(INDENT);
            self.visibility(&item.vis);
            if item.defaultness {
                self.word("default ");
            }
            self.signature(&item.sig);
            if let Some(body) = &item.body {
                self.where_clause_for_body(&item.sig.generics.where_clause);
                self.word("{");
                self.hardbreak_if_nonempty();
                self.inner_attrs(&item.attrs);
                for stmt in body {
                    self.stmt(stmt);
                }
                self.offset(-INDENT);
                self.end();
                self.word("}");
            } else {
                self.where_clause_semi(&item.sig.generics.where_clause);
                self.end();
            }
            self.hardbreak();
        }
        pub fn flexible_item_static(&mut self, item: &FlexibleItemStatic) {
            self.outer_attrs(&item.attrs);
            self.cbox(0);
            self.visibility(&item.vis);
            self.word("static ");
            self.static_mutability(&item.mutability);
            self.ident(&item.ident);
            if let Some(ty) = &item.ty {
                self.word(": ");
                self.ty(ty);
            }
            if let Some(expr) = &item.expr {
                self.word(" = ");
                self.neverbreak();
                self.expr(expr);
            }
            self.word(";");
            self.end();
            self.hardbreak();
        }
        pub fn flexible_item_type(&mut self, item: &FlexibleItemType) {
            self.outer_attrs(&item.attrs);
            self.cbox(INDENT);
            self.visibility(&item.vis);
            if item.defaultness {
                self.word("default ");
            }
            self.word("type ");
            self.ident(&item.ident);
            self.generics(&item.generics);
            for bound in item.bounds.iter().delimited() {
                if bound.is_first {
                    self.word(": ");
                } else {
                    self.space();
                    self.word("+ ");
                }
                self.type_param_bound(&bound);
            }
            if let Some(definition) = &item.definition {
                self.where_clause_oneline(&item.generics.where_clause);
                self.word("= ");
                self.neverbreak();
                self.ibox(-INDENT);
                self.ty(definition);
                self.end();
                self.where_clause_oneline_semi(&item.where_clause_after_eq);
            } else {
                self.where_clause_oneline_semi(&item.generics.where_clause);
            }
            self.end();
            self.hardbreak();
        }
    }
}

pub struct Delimited<I: Iterator> {
    is_first: bool,
    iter: Peekable<I>,
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
impl<T> Deref for IteratorItem<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl Printer {
    //lifetime
    pub fn lifetime(&mut self, lifetime: &Lifetime) {
        self.word("'");
        self.ident(&lifetime.ident);
    }
}

impl Printer {
    //lit
    pub fn lit(&mut self, lit: &Lit) {
        match lit {
            Lit::Str(lit) => self.lit_str(lit),
            Lit::ByteStr(lit) => self.lit_byte_str(lit),
            Lit::Byte(lit) => self.lit_byte(lit),
            Lit::Char(lit) => self.lit_char(lit),
            Lit::Int(lit) => self.lit_int(lit),
            Lit::Float(lit) => self.lit_float(lit),
            Lit::Bool(lit) => self.lit_bool(lit),
            Lit::Verbatim(lit) => self.lit_verbatim(lit),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown Lit"),
        }
    }
    pub fn lit_str(&mut self, lit: &LitStr) {
        self.word(lit.token().to_string());
    }
    fn lit_byte_str(&mut self, lit: &LitByteStr) {
        self.word(lit.token().to_string());
    }
    fn lit_byte(&mut self, lit: &LitByte) {
        self.word(lit.token().to_string());
    }
    fn lit_char(&mut self, lit: &LitChar) {
        self.word(lit.token().to_string());
    }
    fn lit_int(&mut self, lit: &LitInt) {
        self.word(lit.token().to_string());
    }
    fn lit_float(&mut self, lit: &LitFloat) {
        self.word(lit.token().to_string());
    }
    fn lit_bool(&mut self, lit: &LitBool) {
        self.word(if lit.value { "true" } else { "false" });
    }
    fn lit_verbatim(&mut self, token: &Literal) {
        self.word(token.to_string());
    }
}

impl Printer {
    //mac
    pub fn mac(&mut self, mac: &Macro, ident: Option<&Ident>, semicolon: bool) {
        if mac.path.is_ident("macro_rules") {
            if let Some(ident) = ident {
                self.macro_rules(ident, &mac.tokens);
                return;
            }
        }
        #[cfg(feature = "verbatim")]
        if ident.is_none() && self.standard_library_macro(mac, semicolon) {
            return;
        }
        self.path(&mac.path, PathKind::Simple);
        self.word("!");
        if let Some(ident) = ident {
            self.nbsp();
            self.ident(ident);
        }
        let (open, close, delim_break) = match mac.delim {
            MacroDelim::Paren(_) => ("(", ")", Self::zerobreak as fn(&mut Self)),
            MacroDelim::Brace(_) => (" {", "}", Self::hardbreak as fn(&mut Self)),
            MacroDelim::Bracket(_) => ("[", "]", Self::zerobreak as fn(&mut Self)),
        };
        self.word(open);
        if !mac.tokens.is_empty() {
            self.cbox(INDENT);
            delim_break(self);
            self.ibox(0);
            self.macro_rules_tokens(mac.tokens.clone(), false);
            self.end();
            delim_break(self);
            self.offset(-INDENT);
            self.end();
        }
        self.word(close);
        if semicolon {
            match mac.delim {
                MacroDelim::Paren(_) | MacroDelim::Bracket(_) => self.word(";"),
                MacroDelim::Brace(_) => {},
            }
        }
    }
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
                    self.delim_open(delim);
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
                    self.delim_close(delim);
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
#[cfg(feature = "verbatim")]
mod standard_library {
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
        vis: Visibility,
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
                    let eq_token: Token![=] = input.parse()?;
                    let value: Expr = input.parse()?;
                    Expr::Assign(expr::Assign {
                        attrs: Vec::new(),
                        left: Box::new(Expr::Path(expr::Path {
                            attrs: Vec::new(),
                            qself: None,
                            path: Path::from(key),
                        })),
                        eq_token,
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
                let vis: Visibility = input.parse()?;
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
    impl Printer {
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
            self.path(&mac.path, PathKind::Simple);
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
                        self.visibility(&item.vis);
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

impl Printer {
    //pat
    pub fn pat(&mut self, pat: &pat::Pat) {
        use pat::Pat::*;
        match pat {
            Const(pat) => self.expr_const(pat),
            Ident(pat) => self.pat_ident(pat),
            Lit(pat) => self.expr_lit(pat),
            Mac(pat) => self.expr_macro(pat),
            Or(pat) => self.pat_or(pat),
            Paren(pat) => self.pat_paren(pat),
            Path(pat) => self.expr_path(pat),
            Range(pat) => self.expr_range(pat),
            Ref(pat) => self.pat_reference(pat),
            Rest(pat) => self.pat_rest(pat),
            Slice(pat) => self.pat_slice(pat),
            Struct(pat) => self.pat_struct(pat),
            Tuple(pat) => self.pat_tuple(pat),
            TupleStruct(pat) => self.pat_tuple_struct(pat),
            Type(pat) => self.pat_type(pat),
            Stream(pat) => self.pat_verbatim(pat),
            Wild(pat) => self.pat_wild(pat),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown Pat"),
        }
    }
    fn pat_ident(&mut self, pat: &pat::Ident) {
        self.outer_attrs(&pat.attrs);
        if pat.by_ref.is_some() {
            self.word("ref ");
        }
        if pat.mutability.is_some() {
            self.word("mut ");
        }
        self.ident(&pat.ident);
        if let Some((_at_token, subpat)) = &pat.subpat {
            self.word(" @ ");
            self.pat(subpat);
        }
    }
    fn pat_or(&mut self, pat: &pat::Or) {
        self.outer_attrs(&pat.attrs);
        let mut consistent_break = false;
        for case in &pat.cases {
            match case {
                pat::Pat::Lit(_) | pat::Pat::Wild(_) => {},
                _ => {
                    consistent_break = true;
                    break;
                },
            }
        }
        if consistent_break {
            self.cbox(0);
        } else {
            self.ibox(0);
        }
        for case in pat.cases.iter().delimited() {
            if !case.is_first {
                self.space();
                self.word("| ");
            }
            self.pat(&case);
        }
        self.end();
    }
    fn pat_paren(&mut self, pat: &pat::Paren) {
        self.outer_attrs(&pat.attrs);
        self.word("(");
        self.pat(&pat.pat);
        self.word(")");
    }
    fn pat_reference(&mut self, pat: &pat::Ref) {
        self.outer_attrs(&pat.attrs);
        self.word("&");
        if pat.mutability.is_some() {
            self.word("mut ");
        }
        self.pat(&pat.pat);
    }
    fn pat_rest(&mut self, pat: &pat::Rest) {
        self.outer_attrs(&pat.attrs);
        self.word("..");
    }
    fn pat_slice(&mut self, pat: &pat::Slice) {
        self.outer_attrs(&pat.attrs);
        self.word("[");
        for elem in pat.elems.iter().delimited() {
            self.pat(&elem);
            self.trailing_comma(elem.is_last);
        }
        self.word("]");
    }
    fn pat_struct(&mut self, pat: &pat::Struct) {
        self.outer_attrs(&pat.attrs);
        self.cbox(INDENT);
        self.path(&pat.path, PathKind::Expr);
        self.word(" {");
        self.space_if_nonempty();
        for field in pat.fields.iter().delimited() {
            self.field_pat(&field);
            self.trailing_comma_or_space(field.is_last && pat.rest.is_none());
        }
        if let Some(rest) = &pat.rest {
            self.pat_rest(rest);
            self.space();
        }
        self.offset(-INDENT);
        self.end();
        self.word("}");
    }
    fn pat_tuple(&mut self, pat: &pat::Tuple) {
        self.outer_attrs(&pat.attrs);
        self.word("(");
        self.cbox(INDENT);
        self.zerobreak();
        for elem in pat.elems.iter().delimited() {
            self.pat(&elem);
            if pat.elems.len() == 1 {
                if pat.elems.trailing_punct() {
                    self.word(",");
                }
                self.zerobreak();
            } else {
                self.trailing_comma(elem.is_last);
            }
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
    }
    fn pat_tuple_struct(&mut self, pat: &pat::TupleStruct) {
        self.outer_attrs(&pat.attrs);
        self.path(&pat.path, PathKind::Expr);
        self.word("(");
        self.cbox(INDENT);
        self.zerobreak();
        for elem in pat.elems.iter().delimited() {
            self.pat(&elem);
            self.trailing_comma(elem.is_last);
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
    }
    pub fn pat_type(&mut self, pat: &pat::Type) {
        self.outer_attrs(&pat.attrs);
        self.pat(&pat.pat);
        self.word(": ");
        self.ty(&pat.ty);
    }
    #[cfg(not(feature = "verbatim"))]
    fn pat_verbatim(&mut self, pat: &Stream) {
        unimplemented!("Pat::Stream `{}`", pat);
    }
    #[cfg(feature = "verbatim")]
    fn pat_verbatim(&mut self, tokens: &Stream) {
        enum PatVerbatim {
            Ellipsis,
            Box(Pat),
            Const(PatConst),
        }
        struct PatConst {
            attrs: Vec<attr::Attr>,
            block: Block,
        }
        impl parse::Parse for PatVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![box]) {
                    input.parse::<Token![box]>()?;
                    let inner = pat::Pat::parse_single(input)?;
                    Ok(PatVerbatim::Box(inner))
                } else if lookahead.peek(Token![const]) {
                    input.parse::<Token![const]>()?;
                    let content;
                    let brace_token = braced!(content in input);
                    let attrs = content.call(attr::Attr::parse_inner)?;
                    let stmts = content.call(Block::parse_within)?;
                    Ok(PatVerbatim::Const(PatConst {
                        attrs,
                        block: Block { brace_token, stmts },
                    }))
                } else if lookahead.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    Ok(PatVerbatim::Ellipsis)
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let pat: PatVerbatim = match syn::parse2(tokens.clone()) {
            Ok(pat) => pat,
            Err(_) => unimplemented!("Pat::Stream `{}`", tokens),
        };
        match pat {
            PatVerbatim::Ellipsis => {
                self.word("...");
            },
            PatVerbatim::Box(pat) => {
                self.word("box ");
                self.pat(&pat);
            },
            PatVerbatim::Const(pat) => {
                self.word("const ");
                self.cbox(INDENT);
                self.small_block(&pat.block, &pat.attrs);
                self.end();
            },
        }
    }
    fn pat_wild(&mut self, pat: &pat::Wild) {
        self.outer_attrs(&pat.attrs);
        self.word("_");
    }
    fn field_pat(&mut self, field_pat: &FieldPat) {
        self.outer_attrs(&field_pat.attrs);
        if field_pat.colon_token.is_some() {
            self.member(&field_pat.member);
            self.word(": ");
        }
        self.pat(&field_pat.pat);
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum PathKind {
    Simple,
    Type,
    Expr,
}
impl Printer {
    //path
    pub fn path(&mut self, path: &Path, kind: PathKind) {
        assert!(!path.segments.is_empty());
        for segment in path.segments.iter().delimited() {
            if !segment.is_first || path.leading_colon.is_some() {
                self.word("::");
            }
            self.path_segment(&segment, kind);
        }
    }
    pub fn path_segment(&mut self, segment: &PathSegment, kind: PathKind) {
        self.ident(&segment.ident);
        self.path_arguments(&segment.arguments, kind);
    }
    fn path_arguments(&mut self, arguments: &PathArguments, kind: PathKind) {
        match arguments {
            PathArguments::None => {},
            PathArguments::AngleBracketed(arguments) => {
                self.angle_bracketed_generic_arguments(arguments, kind);
            },
            PathArguments::Parenthesized(arguments) => {
                self.parenthesized_generic_arguments(arguments);
            },
        }
    }
    fn generic_argument(&mut self, arg: &GenericArgument) {
        match arg {
            GenericArgument::Lifetime(lifetime) => self.lifetime(lifetime),
            GenericArgument::Type(ty) => self.ty(ty),
            GenericArgument::Const(expr) => match expr {
                Expr::Lit(expr) => self.expr_lit(expr),
                Expr::Block(expr) => self.expr_block(expr),
                _ => {
                    self.word("{");
                    self.expr(expr);
                    self.word("}");
                },
            },
            GenericArgument::AssocType(assoc) => self.assoc_type(assoc),
            GenericArgument::AssocConst(assoc) => self.assoc_const(assoc),
            GenericArgument::Constraint(constraint) => self.constraint(constraint),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown GenericArgument"),
        }
    }
    pub fn angle_bracketed_generic_arguments(&mut self, generic: &AngleBracketedGenericArguments, path_kind: PathKind) {
        if generic.args.is_empty() || path_kind == PathKind::Simple {
            return;
        }
        if path_kind == PathKind::Expr {
            self.word("::");
        }
        self.word("<");
        self.cbox(INDENT);
        self.zerobreak();
        #[derive(Ord, PartialOrd, Eq, PartialEq)]
        enum Group {
            First,
            Second,
        }
        fn group(arg: &GenericArgument) -> Group {
            match arg {
                GenericArgument::Lifetime(_) => Group::First,
                GenericArgument::Type(_)
                | GenericArgument::Const(_)
                | GenericArgument::AssocType(_)
                | GenericArgument::AssocConst(_)
                | GenericArgument::Constraint(_) => Group::Second,
                #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
                _ => Group::Second,
            }
        }
        let last = generic.args.iter().max_by_key(|param| group(param));
        for current_group in [Group::First, Group::Second] {
            for arg in &generic.args {
                if group(arg) == current_group {
                    self.generic_argument(arg);
                    self.trailing_comma(ptr::eq(arg, last.unwrap()));
                }
            }
        }
        self.offset(-INDENT);
        self.end();
        self.word(">");
    }
    fn assoc_type(&mut self, assoc: &AssocType) {
        self.ident(&assoc.ident);
        if let Some(generics) = &assoc.generics {
            self.angle_bracketed_generic_arguments(generics, PathKind::Type);
        }
        self.word(" = ");
        self.ty(&assoc.ty);
    }
    fn assoc_const(&mut self, assoc: &AssocConst) {
        self.ident(&assoc.ident);
        if let Some(generics) = &assoc.generics {
            self.angle_bracketed_generic_arguments(generics, PathKind::Type);
        }
        self.word(" = ");
        self.expr(&assoc.value);
    }
    fn constraint(&mut self, constraint: &Constraint) {
        self.ident(&constraint.ident);
        if let Some(generics) = &constraint.generics {
            self.angle_bracketed_generic_arguments(generics, PathKind::Type);
        }
        self.ibox(INDENT);
        for bound in constraint.bounds.iter().delimited() {
            if bound.is_first {
                self.word(": ");
            } else {
                self.space();
                self.word("+ ");
            }
            self.type_param_bound(&bound);
        }
        self.end();
    }
    fn parenthesized_generic_arguments(&mut self, arguments: &ParenthesizedGenericArguments) {
        self.cbox(INDENT);
        self.word("(");
        self.zerobreak();
        for ty in arguments.inputs.iter().delimited() {
            self.ty(&ty);
            self.trailing_comma(ty.is_last);
        }
        self.offset(-INDENT);
        self.word(")");
        self.return_type(&arguments.output);
        self.end();
    }
    pub fn qpath(&mut self, qself: &Option<QSelf>, path: &Path, kind: PathKind) {
        let qself = match qself {
            Some(qself) => qself,
            None => {
                self.path(path, kind);
                return;
            },
        };
        assert!(qself.position < path.segments.len());
        self.word("<");
        self.ty(&qself.ty);
        let mut segments = path.segments.iter();
        if qself.position > 0 {
            self.word(" as ");
            for segment in segments.by_ref().take(qself.position).delimited() {
                if !segment.is_first || path.leading_colon.is_some() {
                    self.word("::");
                }
                self.path_segment(&segment, PathKind::Type);
                if segment.is_last {
                    self.word(">");
                }
            }
        } else {
            self.word(">");
        }
        for segment in segments {
            self.word("::");
            self.path_segment(segment, kind);
        }
    }
}

pub struct RingBuffer<T> {
    data: VecDeque<T>,
    offset: usize,
}
impl<T> RingBuffer<T> {
    pub fn new() -> Self {
        RingBuffer {
            data: VecDeque::new(),
            offset: 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn push(&mut self, value: T) -> usize {
        let index = self.offset + self.data.len();
        self.data.push_back(value);
        index
    }
    pub fn clear(&mut self) {
        self.data.clear();
    }
    pub fn index_of_first(&self) -> usize {
        self.offset
    }
    pub fn first(&self) -> &T {
        &self.data[0]
    }
    pub fn first_mut(&mut self) -> &mut T {
        &mut self.data[0]
    }
    pub fn pop_first(&mut self) -> T {
        self.offset += 1;
        self.data.pop_front().unwrap()
    }
    pub fn last(&self) -> &T {
        self.data.back().unwrap()
    }
    pub fn last_mut(&mut self) -> &mut T {
        self.data.back_mut().unwrap()
    }
    pub fn second_last(&self) -> &T {
        &self.data[self.data.len() - 2]
    }
    pub fn pop_last(&mut self) {
        self.data.pop_back().unwrap();
    }
}
impl<T> Index<usize> for RingBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index.checked_sub(self.offset).unwrap()]
    }
}
impl<T> IndexMut<usize> for RingBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index.checked_sub(self.offset).unwrap()]
    }
}

impl Printer {
    //stmt
    pub fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Local(local) => {
                self.outer_attrs(&local.attrs);
                self.ibox(0);
                self.word("let ");
                self.pat(&local.pat);
                if let Some(local_init) = &local.init {
                    self.word(" = ");
                    self.neverbreak();
                    self.expr(&local_init.expr);
                    if let Some((_else, diverge)) = &local_init.diverge {
                        self.word(" else ");
                        if let Expr::Block(expr) = diverge.as_ref() {
                            self.small_block(&expr.block, &[]);
                        } else {
                            self.word("{");
                            self.space();
                            self.ibox(INDENT);
                            self.expr(diverge);
                            self.end();
                            self.space();
                            self.offset(-INDENT);
                            self.word("}");
                        }
                    }
                }
                self.word(";");
                self.end();
                self.hardbreak();
            },
            Stmt::Item(item) => self.item(item),
            Stmt::Expr(expr, None) => {
                if break_after(expr) {
                    self.ibox(0);
                    self.expr_beginning_of_line(expr, true);
                    if add_semi(expr) {
                        self.word(";");
                    }
                    self.end();
                    self.hardbreak();
                } else {
                    self.expr_beginning_of_line(expr, true);
                }
            },
            Stmt::Expr(expr, Some(_semi)) => {
                if let Expr::Verbatim(tokens) = expr {
                    if tokens.is_empty() {
                        return;
                    }
                }
                self.ibox(0);
                self.expr_beginning_of_line(expr, true);
                if !remove_semi(expr) {
                    self.word(";");
                }
                self.end();
                self.hardbreak();
            },
            Stmt::Macro(stmt) => {
                self.outer_attrs(&stmt.attrs);
                let semicolon = true;
                self.mac(&stmt.mac, None, semicolon);
                self.hardbreak();
            },
        }
    }
}
pub fn add_semi(expr: &Expr) -> bool {
    match expr {
        Expr::Assign(_) | Expr::Break(_) | Expr::Continue(_) | Expr::Return(_) | Expr::Yield(_) => true,
        Expr::Binary(expr) => match expr.op {
            BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_) => true,
            BinOp::Add(_)
            | BinOp::Sub(_)
            | BinOp::Mul(_)
            | BinOp::Div(_)
            | BinOp::Rem(_)
            | BinOp::And(_)
            | BinOp::Or(_)
            | BinOp::BitXor(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::Shl(_)
            | BinOp::Shr(_)
            | BinOp::Eq(_)
            | BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Ne(_)
            | BinOp::Ge(_)
            | BinOp::Gt(_) => false,
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown BinOp"),
        },
        Expr::Group(group) => add_semi(&group.expr),
        Expr::Array(_)
        | Expr::Async(_)
        | Expr::Await(_)
        | Expr::Block(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Closure(_)
        | Expr::Const(_)
        | Expr::Field(_)
        | Expr::ForLoop(_)
        | Expr::If(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Struct(_)
        | Expr::Try(_)
        | Expr::TryBlock(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Unsafe(_)
        | Expr::Verbatim(_)
        | Expr::While(_) => false,
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}
pub fn break_after(expr: &Expr) -> bool {
    if let Expr::Group(group) = expr {
        if let Expr::Verbatim(verbatim) = group.expr.as_ref() {
            return !verbatim.is_empty();
        }
    }
    true
}
fn remove_semi(expr: &Expr) -> bool {
    match expr {
        Expr::ForLoop(_) | Expr::While(_) => true,
        Expr::Group(group) => remove_semi(&group.expr),
        Expr::If(expr) => match &expr.else_branch {
            Some((_else_token, else_branch)) => remove_semi(else_branch),
            None => true,
        },
        Expr::Array(_)
        | Expr::Assign(_)
        | Expr::Async(_)
        | Expr::Await(_)
        | Expr::Binary(_)
        | Expr::Block(_)
        | Expr::Break(_)
        | Expr::Call(_)
        | Expr::Cast(_)
        | Expr::Closure(_)
        | Expr::Continue(_)
        | Expr::Const(_)
        | Expr::Field(_)
        | Expr::Index(_)
        | Expr::Infer(_)
        | Expr::Let(_)
        | Expr::Lit(_)
        | Expr::Loop(_)
        | Expr::Macro(_)
        | Expr::Match(_)
        | Expr::MethodCall(_)
        | Expr::Paren(_)
        | Expr::Path(_)
        | Expr::Range(_)
        | Expr::Reference(_)
        | Expr::Repeat(_)
        | Expr::Return(_)
        | Expr::Struct(_)
        | Expr::Try(_)
        | Expr::TryBlock(_)
        | Expr::Tuple(_)
        | Expr::Unary(_)
        | Expr::Unsafe(_)
        | Expr::Verbatim(_)
        | Expr::Yield(_) => false,
        #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
        _ => false,
    }
}

impl Printer {
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
        self.delim_open(delim);
        if !stream.is_empty() {
            if delim == Delim::Brace {
                self.space();
            }
            group_contents(self, stream);
            if delim == Delim::Brace {
                self.space();
            }
        }
        self.delim_close(delim);
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
    pub fn delim_open(&mut self, delim: Delim) {
        self.word(match delim {
            Delim::Parenthesis => "(",
            Delim::Brace => "{",
            Delim::Bracket => "[",
            Delim::None => return,
        });
    }
    pub fn delim_close(&mut self, delim: Delim) {
        self.word(match delim {
            Delim::Parenthesis => ")",
            Delim::Brace => "}",
            Delim::Bracket => "]",
            Delim::None => return,
        });
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

impl Printer {
    //ty
    pub fn ty(&mut self, ty: &typ::Type) {
        use typ::Type::*;
        match ty {
            Array(ty) => self.type_array(ty),
            Fn(ty) => self.type_bare_fn(ty),
            Group(ty) => self.type_group(ty),
            Impl(ty) => self.type_impl_trait(ty),
            Infer(ty) => self.type_infer(ty),
            Mac(ty) => self.type_macro(ty),
            Never(ty) => self.type_never(ty),
            Paren(ty) => self.type_paren(ty),
            Path(ty) => self.type_path(ty),
            Ptr(ty) => self.type_ptr(ty),
            Ref(ty) => self.type_reference(ty),
            Slice(ty) => self.type_slice(ty),
            Trait(ty) => self.type_trait_object(ty),
            Tuple(ty) => self.type_tuple(ty),
            Stream(ty) => self.type_verbatim(ty),
            #[cfg_attr(all(test, exhaustive), deny(non_exhaustive_omitted_patterns))]
            _ => unimplemented!("unknown Type"),
        }
    }
    fn type_array(&mut self, ty: &typ::Array) {
        self.word("[");
        self.ty(&ty.elem);
        self.word("; ");
        self.expr(&ty.len);
        self.word("]");
    }
    fn type_bare_fn(&mut self, ty: &typ::Fn) {
        if let Some(bound_lifetimes) = &ty.lifetimes {
            self.bound_lifetimes(bound_lifetimes);
        }
        if ty.unsafety.is_some() {
            self.word("unsafe ");
        }
        if let Some(abi) = &ty.abi {
            self.abi(abi);
        }
        self.word("fn(");
        self.cbox(INDENT);
        self.zerobreak();
        for bare_fn_arg in ty.inputs.iter().delimited() {
            self.bare_fn_arg(&bare_fn_arg);
            self.trailing_comma(bare_fn_arg.is_last && ty.variadic.is_none());
        }
        if let Some(variadic) = &ty.variadic {
            self.bare_variadic(variadic);
            self.zerobreak();
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
        self.return_type(&ty.output);
    }
    fn type_group(&mut self, ty: &typ::Group) {
        self.ty(&ty.elem);
    }
    fn type_impl_trait(&mut self, ty: &TypeImplTrait) {
        self.word("impl ");
        for type_param_bound in ty.bounds.iter().delimited() {
            if !type_param_bound.is_first {
                self.word(" + ");
            }
            self.type_param_bound(&type_param_bound);
        }
    }
    fn type_infer(&mut self, ty: &typ::Infer) {
        let _ = ty;
        self.word("_");
    }
    fn type_macro(&mut self, ty: &typ::Mac) {
        let semicolon = false;
        self.mac(&ty.mac, None, semicolon);
    }
    fn type_never(&mut self, ty: &typ::Never) {
        let _ = ty;
        self.word("!");
    }
    fn type_paren(&mut self, ty: &typ::Paren) {
        self.word("(");
        self.ty(&ty.elem);
        self.word(")");
    }
    fn type_path(&mut self, ty: &typ::Path) {
        self.qpath(&ty.qself, &ty.path, PathKind::Type);
    }
    fn type_ptr(&mut self, ty: &typ::Ptr) {
        self.word("*");
        if ty.mutability.is_some() {
            self.word("mut ");
        } else {
            self.word("const ");
        }
        self.ty(&ty.elem);
    }
    fn type_reference(&mut self, ty: &typ::Ref) {
        self.word("&");
        if let Some(lifetime) = &ty.lifetime {
            self.lifetime(lifetime);
            self.nbsp();
        }
        if ty.mutability.is_some() {
            self.word("mut ");
        }
        self.ty(&ty.elem);
    }
    fn type_slice(&mut self, ty: &typ::Slice) {
        self.word("[");
        self.ty(&ty.elem);
        self.word("]");
    }
    fn type_trait_object(&mut self, ty: &typ::Trait) {
        self.word("dyn ");
        for type_param_bound in ty.bounds.iter().delimited() {
            if !type_param_bound.is_first {
                self.word(" + ");
            }
            self.type_param_bound(&type_param_bound);
        }
    }
    fn type_tuple(&mut self, ty: &typ::Tuple) {
        self.word("(");
        self.cbox(INDENT);
        self.zerobreak();
        for elem in ty.elems.iter().delimited() {
            self.ty(&elem);
            if ty.elems.len() == 1 {
                self.word(",");
                self.zerobreak();
            } else {
                self.trailing_comma(elem.is_last);
            }
        }
        self.offset(-INDENT);
        self.end();
        self.word(")");
    }
    #[cfg(not(feature = "verbatim"))]
    fn type_verbatim(&mut self, ty: &Stream) {
        unimplemented!("Type::Stream `{}`", ty);
    }
    #[cfg(feature = "verbatim")]
    fn type_verbatim(&mut self, tokens: &Stream) {
        enum TypeVerbatim {
            Ellipsis,
            DynStar(DynStar),
            MutSelf(MutSelf),
            NotType(NotType),
        }
        struct DynStar {
            bounds: punct::Puncted<TypeParamBound, Token![+]>,
        }
        struct MutSelf {
            ty: Option<Type>,
        }
        struct NotType {
            inner: Type,
        }
        impl parse::Parse for TypeVerbatim {
            fn parse(input: parse::Stream) -> Res<Self> {
                let lookahead = input.lookahead1();
                if lookahead.peek(Token![dyn]) {
                    input.parse::<Token![dyn]>()?;
                    input.parse::<Token![*]>()?;
                    let bounds = input.parse_terminated(TypeParamBound::parse, Token![+])?;
                    Ok(TypeVerbatim::DynStar(DynStar { bounds }))
                } else if lookahead.peek(Token![mut]) {
                    input.parse::<Token![mut]>()?;
                    input.parse::<Token![self]>()?;
                    let ty = if input.is_empty() {
                        None
                    } else {
                        input.parse::<Token![:]>()?;
                        let ty: Type = input.parse()?;
                        Some(ty)
                    };
                    Ok(TypeVerbatim::MutSelf(MutSelf { ty }))
                } else if lookahead.peek(Token![!]) {
                    input.parse::<Token![!]>()?;
                    let inner: Type = input.parse()?;
                    Ok(TypeVerbatim::NotType(NotType { inner }))
                } else if lookahead.peek(Token![...]) {
                    input.parse::<Token![...]>()?;
                    Ok(TypeVerbatim::Ellipsis)
                } else {
                    Err(lookahead.error())
                }
            }
        }
        let ty: TypeVerbatim = match syn::parse2(tokens.clone()) {
            Ok(ty) => ty,
            Err(_) => unimplemented!("Type::Stream`{}`", tokens),
        };
        match ty {
            TypeVerbatim::Ellipsis => {
                self.word("...");
            },
            TypeVerbatim::DynStar(ty) => {
                self.word("dyn* ");
                for type_param_bound in ty.bounds.iter().delimited() {
                    if !type_param_bound.is_first {
                        self.word(" + ");
                    }
                    self.type_param_bound(&type_param_bound);
                }
            },
            TypeVerbatim::MutSelf(bare_fn_arg) => {
                self.word("mut self");
                if let Some(ty) = &bare_fn_arg.ty {
                    self.word(": ");
                    self.ty(ty);
                }
            },
            TypeVerbatim::NotType(ty) => {
                self.word("!");
                self.ty(&ty.inner);
            },
        }
    }
    pub fn return_type(&mut self, ty: &typ::Ret) {
        match ty {
            typ::Ret::Default => {},
            typ::Ret::Type(_arrow, ty) => {
                self.word(" -> ");
                self.ty(ty);
            },
        }
    }
    fn bare_fn_arg(&mut self, bare_fn_arg: &typ::FnArg) {
        self.outer_attrs(&bare_fn_arg.attrs);
        if let Some((name, _colon)) = &bare_fn_arg.name {
            self.ident(name);
            self.word(": ");
        }
        self.ty(&bare_fn_arg.ty);
    }
    fn bare_variadic(&mut self, variadic: &typ::Variadic) {
        self.outer_attrs(&variadic.attrs);
        if let Some((name, _colon)) = &variadic.name {
            self.ident(name);
            self.word(": ");
        }
        self.word("...");
    }
    pub fn abi(&mut self, abi: &typ::Abi) {
        self.word("extern ");
        if let Some(name) = &abi.name {
            self.lit_str(name);
            self.nbsp();
        }
    }
}

const MARGIN: isize = 89;
const INDENT: isize = 4;
const MIN_SPACE: isize = 60;
pub fn unparse(file: &File) -> String {
    let mut p = Printer::new();
    p.file(file);
    p.eof()
}
