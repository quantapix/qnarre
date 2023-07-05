use crate::Lifetime;
use proc_macro2::extra::DelimSpan;
use proc_macro2::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use std::cmp::Ordering;
use std::marker::PhantomData;
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
    fn recursive_new(entries: &mut Vec<Entry>, stream: TokenStream) {
        for tt in stream {
            match tt {
                TokenTree::Ident(ident) => entries.push(Entry::Ident(ident)),
                TokenTree::Punct(punct) => entries.push(Entry::Punct(punct)),
                TokenTree::Literal(literal) => entries.push(Entry::Literal(literal)),
                TokenTree::Group(group) => {
                    let group_start_index = entries.len();
                    entries.push(Entry::End(0)); // we replace this below
                    Self::recursive_new(entries, group.stream());
                    let group_end_index = entries.len();
                    entries.push(Entry::End(-(group_end_index as isize)));
                    let group_end_offset = group_end_index - group_start_index;
                    entries[group_start_index] = Entry::Group(group, group_end_offset);
                },
            }
        }
    }
    pub fn new(stream: proc_macro::TokenStream) -> Self {
        Self::new2(stream.into())
    }
    pub fn new2(stream: TokenStream) -> Self {
        let mut entries = Vec::new();
        Self::recursive_new(&mut entries, stream);
        entries.push(Entry::End(-(entries.len() as isize)));
        Self {
            entries: entries.into_boxed_slice(),
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
        while let Entry::Group(group, _) = self.entry() {
            if group.delimiter() == Delimiter::None {
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
        if let Entry::Group(group, end_offset) = self.entry() {
            if group.delimiter() == delim {
                let span = group.delim_span();
                let end_of_group = unsafe { self.ptr.add(*end_offset) };
                let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
                let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                return Some((inside_of_group, span, after_group));
            }
        }
        None
    }
    pub(crate) fn any_group(self) -> Option<(Cursor<'a>, Delimiter, DelimSpan, Cursor<'a>)> {
        if let Entry::Group(group, end_offset) = self.entry() {
            let delimiter = group.delimiter();
            let span = group.delim_span();
            let end_of_group = unsafe { self.ptr.add(*end_offset) };
            let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
            let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
            return Some((inside_of_group, delimiter, span, after_group));
        }
        None
    }
    pub(crate) fn any_group_token(self) -> Option<(Group, Cursor<'a>)> {
        if let Entry::Group(group, end_offset) = self.entry() {
            let end_of_group = unsafe { self.ptr.add(*end_offset) };
            let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
            return Some((group.clone(), after_group));
        }
        None
    }
    pub fn ident(mut self) -> Option<(Ident, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Ident(ident) => Some((ident.clone(), unsafe { self.bump_ignore_group() })),
            _ => None,
        }
    }
    pub fn punct(mut self) -> Option<(Punct, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Punct(punct) if punct.as_char() != '\'' => {
                Some((punct.clone(), unsafe { self.bump_ignore_group() }))
            },
            _ => None,
        }
    }
    pub fn literal(mut self) -> Option<(Literal, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Literal(literal) => Some((literal.clone(), unsafe { self.bump_ignore_group() })),
            _ => None,
        }
    }
    pub fn lifetime(mut self) -> Option<(Lifetime, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Punct(punct) if punct.as_char() == '\'' && punct.spacing() == Spacing::Joint => {
                let next = unsafe { self.bump_ignore_group() };
                let (ident, rest) = next.ident()?;
                let lifetime = Lifetime {
                    apostrophe: punct.span(),
                    ident,
                };
                Some((lifetime, rest))
            },
            _ => None,
        }
    }
    pub fn token_stream(self) -> TokenStream {
        let mut tts = Vec::new();
        let mut cursor = self;
        while let Some((tt, rest)) = cursor.token_tree() {
            tts.push(tt);
            cursor = rest;
        }
        tts.into_iter().collect()
    }
    pub fn token_tree(self) -> Option<(TokenTree, Cursor<'a>)> {
        let (tree, len) = match self.entry() {
            Entry::Group(group, end_offset) => (group.clone().into(), *end_offset),
            Entry::Literal(literal) => (literal.clone().into(), 1),
            Entry::Ident(ident) => (ident.clone().into(), 1),
            Entry::Punct(punct) => (punct.clone().into(), 1),
            Entry::End(_) => return None,
        };
        let rest = unsafe { Cursor::create(self.ptr.add(len), self.scope) };
        Some((tree, rest))
    }
    pub fn span(self) -> Span {
        match self.entry() {
            Entry::Group(group, _) => group.span(),
            Entry::Literal(literal) => literal.span(),
            Entry::Ident(ident) => ident.span(),
            Entry::Punct(punct) => punct.span(),
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
                        Entry::Group(group, _) => {
                            depth -= 1;
                            if depth == 0 {
                                return group.span();
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
        let len = match self.entry() {
            Entry::End(_) => return None,
            Entry::Punct(punct) if punct.as_char() == '\'' && punct.spacing() == Spacing::Joint => {
                match unsafe { &*self.ptr.add(1) } {
                    Entry::Ident(_) => 2,
                    _ => 1,
                }
            },
            Entry::Group(_, end_offset) => *end_offset,
            _ => 1,
        };
        Some(unsafe { Cursor::create(self.ptr.add(len), self.scope) })
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
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}
impl<'a> PartialOrd for Cursor<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if same_buffer(*self, *other) {
            Some(self.ptr.cmp(&other.ptr))
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
fn start_of_buffer(cursor: Cursor) -> *const Entry {
    unsafe {
        match &*cursor.scope {
            Entry::End(offset) => cursor.scope.offset(*offset),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn cmp_assuming_same_buffer(a: Cursor, b: Cursor) -> Ordering {
    a.ptr.cmp(&b.ptr)
}
pub(crate) fn open_span_of_group(cursor: Cursor) -> Span {
    match cursor.entry() {
        Entry::Group(group, _) => group.span_open(),
        _ => cursor.span(),
    }
}
pub(crate) fn close_span_of_group(cursor: Cursor) -> Span {
    match cursor.entry() {
        Entry::Group(group, _) => group.span_close(),
        _ => cursor.span(),
    }
}
