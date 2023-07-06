use crate::Lifetime;
use proc_macro2::{extra::DelimSpan, Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::{cmp::Ordering, marker::PhantomData};

enum Entry {
    Group(Group, usize),
    Ident(Ident),
    Punct(Punct),
    Literal(Literal),
    End(isize),
}
pub struct TokenBuffer {
    entries: Box<[Entry]>,
}
impl TokenBuffer {
    fn recursive_new(ys: &mut Vec<Entry>, xs: TokenStream) {
        for tt in xs {
            match tt {
                TokenTree::Ident(x) => ys.push(Entry::Ident(x)),
                TokenTree::Punct(x) => ys.push(Entry::Punct(x)),
                TokenTree::Literal(x) => ys.push(Entry::Literal(x)),
                TokenTree::Group(x) => {
                    let beg = ys.len();
                    ys.push(Entry::End(0));
                    Self::recursive_new(ys, x.stream());
                    let end = ys.len();
                    ys.push(Entry::End(-(end as isize)));
                    let off = end - beg;
                    ys[beg] = Entry::Group(x, off);
                },
            }
        }
    }
    pub fn new(x: proc_macro::TokenStream) -> Self {
        Self::new2(x.into())
    }
    pub fn new2(x: TokenStream) -> Self {
        let mut ys = Vec::new();
        Self::recursive_new(&mut ys, x);
        ys.push(Entry::End(-(ys.len() as isize)));
        Self {
            entries: ys.into_boxed_slice(),
        }
    }
    pub fn begin(&self) -> Cursor {
        let ptr = self.entries.as_ptr();
        unsafe { Cursor::create(ptr, ptr.add(self.entries.len() - 1)) }
    }
}
pub struct Cursor<'a> {
    ptr: *const Entry,
    scope: *const Entry,
    marker: PhantomData<&'a Entry>,
}
impl<'a> Cursor<'a> {
    pub fn empty() -> Self {
        struct UnsafeSyncEntry(Entry);
        unsafe impl Sync for UnsafeSyncEntry {}
        static EMPTY_ENTRY: UnsafeSyncEntry = UnsafeSyncEntry(Entry::End(0));
        Cursor {
            ptr: &EMPTY_ENTRY.0,
            scope: &EMPTY_ENTRY.0,
            marker: PhantomData,
        }
    }
    unsafe fn create(mut ptr: *const Entry, scope: *const Entry) -> Self {
        while let Entry::End(_) = *ptr {
            if ptr == scope {
                break;
            }
            ptr = ptr.add(1);
        }
        Cursor {
            ptr,
            scope,
            marker: PhantomData,
        }
    }
    fn entry(self) -> &'a Entry {
        unsafe { &*self.ptr }
    }
    unsafe fn bump_ignore_group(self) -> Cursor<'a> {
        Cursor::create(self.ptr.offset(1), self.scope)
    }
    fn ignore_none(&mut self) {
        while let Entry::Group(x, _) = self.entry() {
            if x.delimiter() == Delimiter::None {
                unsafe { *self = self.bump_ignore_group() };
            } else {
                break;
            }
        }
    }
    pub fn eof(self) -> bool {
        self.ptr == self.scope
    }
    pub fn group(mut self, delim: Delimiter) -> Option<(Cursor<'a>, DelimSpan, Cursor<'a>)> {
        if delim != Delimiter::None {
            self.ignore_none();
        }
        if let Entry::Group(x, end) = self.entry() {
            if x.delimiter() == delim {
                let span = x.delim_span();
                let end_of_group = unsafe { self.ptr.add(*end) };
                let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
                let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                return Some((inside_of_group, span, after_group));
            }
        }
        None
    }
    pub(crate) fn any_group(self) -> Option<(Cursor<'a>, Delimiter, DelimSpan, Cursor<'a>)> {
        if let Entry::Group(x, end) = self.entry() {
            let delimiter = x.delimiter();
            let span = x.delim_span();
            let end_of_group = unsafe { self.ptr.add(*end) };
            let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
            let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
            return Some((inside_of_group, delimiter, span, after_group));
        }
        None
    }
    pub(crate) fn any_group_token(self) -> Option<(Group, Cursor<'a>)> {
        if let Entry::Group(x, end) = self.entry() {
            let end_of_group = unsafe { self.ptr.add(*end) };
            let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
            return Some((x.clone(), after_group));
        }
        None
    }
    pub fn ident(mut self) -> Option<(Ident, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Ident(x) => Some((x.clone(), unsafe { self.bump_ignore_group() })),
            _ => None,
        }
    }
    pub fn punct(mut self) -> Option<(Punct, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Punct(x) if x.as_char() != '\'' => Some((x.clone(), unsafe { self.bump_ignore_group() })),
            _ => None,
        }
    }
    pub fn literal(mut self) -> Option<(Literal, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Literal(x) => Some((x.clone(), unsafe { self.bump_ignore_group() })),
            _ => None,
        }
    }
    pub fn lifetime(mut self) -> Option<(Lifetime, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => {
                let next = unsafe { self.bump_ignore_group() };
                let (ident, rest) = next.ident()?;
                let lifetime = Lifetime {
                    apostrophe: x.span(),
                    ident,
                };
                Some((lifetime, rest))
            },
            _ => None,
        }
    }
    pub fn token_stream(self) -> TokenStream {
        let mut ys = Vec::new();
        let mut cur = self;
        while let Some((x, rest)) = cur.token_tree() {
            ys.push(x);
            cur = rest;
        }
        ys.into_iter().collect()
    }
    pub fn token_tree(self) -> Option<(TokenTree, Cursor<'a>)> {
        let (tree, len) = match self.entry() {
            Entry::Group(x, end) => (x.clone().into(), *end),
            Entry::Literal(x) => (x.clone().into(), 1),
            Entry::Ident(x) => (x.clone().into(), 1),
            Entry::Punct(x) => (x.clone().into(), 1),
            Entry::End(_) => return None,
        };
        let rest = unsafe { Cursor::create(self.ptr.add(len), self.scope) };
        Some((tree, rest))
    }
    pub fn span(self) -> Span {
        match self.entry() {
            Entry::Group(x, _) => x.span(),
            Entry::Literal(x) => x.span(),
            Entry::Ident(x) => x.span(),
            Entry::Punct(x) => x.span(),
            Entry::End(_) => Span::call_site(),
        }
    }
    pub(crate) fn prev_span(mut self) -> Span {
        if start_of_buffer(self) < self.ptr {
            self.ptr = unsafe { self.ptr.offset(-1) };
            if let Entry::End(_) = self.entry() {
                let mut depth = 1;
                loop {
                    self.ptr = unsafe { self.ptr.offset(-1) };
                    match self.entry() {
                        Entry::Group(x, _) => {
                            depth -= 1;
                            if depth == 0 {
                                return x.span();
                            }
                        },
                        Entry::End(_) => depth += 1,
                        Entry::Literal(_) | Entry::Ident(_) | Entry::Punct(_) => {},
                    }
                }
            }
        }
        self.span()
    }
    pub(crate) fn skip(self) -> Option<Cursor<'a>> {
        let y = match self.entry() {
            Entry::End(_) => return None,
            Entry::Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => {
                match unsafe { &*self.ptr.add(1) } {
                    Entry::Ident(_) => 2,
                    _ => 1,
                }
            },
            Entry::Group(_, x) => *x,
            _ => 1,
        };
        Some(unsafe { Cursor::create(self.ptr.add(y), self.scope) })
    }
}
impl<'a> Copy for Cursor<'a> {}
impl<'a> Clone for Cursor<'a> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a> Eq for Cursor<'a> {}
impl<'a> PartialEq for Cursor<'a> {
    fn eq(&self, x: &Self) -> bool {
        self.ptr == x.ptr
    }
}
impl<'a> PartialOrd for Cursor<'a> {
    fn partial_cmp(&self, x: &Self) -> Option<Ordering> {
        if same_buffer(*self, *x) {
            Some(self.ptr.cmp(&x.ptr))
        } else {
            None
        }
    }
}
pub(crate) fn same_scope(a: Cursor, b: Cursor) -> bool {
    a.scope == b.scope
}
pub(crate) fn same_buffer(a: Cursor, b: Cursor) -> bool {
    start_of_buffer(a) == start_of_buffer(b)
}
fn start_of_buffer(c: Cursor) -> *const Entry {
    unsafe {
        match &*c.scope {
            Entry::End(x) => c.scope.offset(*x),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn cmp_assuming_same_buffer(a: Cursor, b: Cursor) -> Ordering {
    a.ptr.cmp(&b.ptr)
}
pub(crate) fn open_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_open(),
        _ => c.span(),
    }
}
pub(crate) fn close_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_close(),
        _ => c.span(),
    }
}
