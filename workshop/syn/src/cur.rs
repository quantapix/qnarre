use super::{
    path::Path,
    pm2::{self, Delim, DelimSpan, Spacing, Span, Stream, Tree},
    *,
};

enum Entry {
    Group(Group, usize),
    Ident(Ident),
    Punct(Punct),
    Lit(Lit),
    End(isize),
}

pub use super::{pm2::Lit, Group, Ident, Punct};

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
            if x.delimiter() == Delim::None {
                unsafe { *self = self.bump_ignore_group() };
            } else {
                break;
            }
        }
    }
    pub fn eof(self) -> bool {
        self.ptr == self.scope
    }
    pub fn group(mut self, d: Delim) -> Option<(Cursor<'a>, DelimSpan, Cursor<'a>)> {
        if d != Delim::None {
            self.ignore_none();
        }
        if let Entry::Group(x, i) = self.entry() {
            if x.delimiter() == d {
                let span = x.delim_span();
                let end = unsafe { self.ptr.add(*i) };
                let inside = unsafe { Cursor::create(self.ptr.add(1), end) };
                let after = unsafe { Cursor::create(end, self.scope) };
                return Some((inside, span, after));
            }
        }
        None
    }
    pub fn any_group(self) -> Option<(Cursor<'a>, Delim, DelimSpan, Cursor<'a>)> {
        if let Entry::Group(x, i) = self.entry() {
            let delim = x.delimiter();
            let span = x.delim_span();
            let end = unsafe { self.ptr.add(*i) };
            let inside = unsafe { Cursor::create(self.ptr.add(1), end) };
            let after = unsafe { Cursor::create(end, self.scope) };
            return Some((inside, delim, span, after));
        }
        None
    }
    pub fn any_group_token(self) -> Option<(Group, Cursor<'a>)> {
        if let Entry::Group(x, i) = self.entry() {
            let end = unsafe { self.ptr.add(*i) };
            let after = unsafe { Cursor::create(end, self.scope) };
            return Some((x.clone(), after));
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
    pub fn literal(mut self) -> Option<(Lit, Cursor<'a>)> {
        self.ignore_none();
        match self.entry() {
            Entry::Lit(x) => Some((x.clone(), unsafe { self.bump_ignore_group() })),
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
    pub fn token_stream(self) -> pm2::Stream {
        let mut ys = Vec::new();
        let mut c = self;
        while let Some((x, rest)) = c.token_tree() {
            ys.push(x);
            c = rest;
        }
        ys.into_iter().collect()
    }
    pub fn token_tree(self) -> Option<(Tree, Cursor<'a>)> {
        let (y, i) = match self.entry() {
            Entry::Group(x, i) => (x.clone().into(), *i),
            Entry::Lit(x) => (x.clone().into(), 1),
            Entry::Ident(x) => (x.clone().into(), 1),
            Entry::Punct(x) => (x.clone().into(), 1),
            Entry::End(_) => return None,
        };
        let rest = unsafe { Cursor::create(self.ptr.add(i), self.scope) };
        Some((y, rest))
    }
    pub fn span(self) -> Span {
        match self.entry() {
            Entry::Group(x, _) => x.span(),
            Entry::Lit(x) => x.span(),
            Entry::Ident(x) => x.span(),
            Entry::Punct(x) => x.span(),
            Entry::End(_) => Span::call_site(),
        }
    }
    pub fn prev_span(mut self) -> Span {
        if buff_start(self) < self.ptr {
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
                        Entry::Lit(_) | Entry::Ident(_) | Entry::Punct(_) => {},
                    }
                }
            }
        }
        self.span()
    }
    pub fn skip(self) -> Option<Cursor<'a>> {
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
        if same_buff(*self, *x) {
            Some(self.ptr.cmp(&x.ptr))
        } else {
            None
        }
    }
}

pub struct Buffer {
    entries: Box<[Entry]>,
}
impl Buffer {
    fn recursive_new(ys: &mut Vec<Entry>, xs: pm2::Stream) {
        for x in xs {
            match x {
                Tree::Ident(x) => ys.push(Entry::Ident(x)),
                Tree::Punct(x) => ys.push(Entry::Punct(x)),
                Tree::Literal(x) => ys.push(Entry::Lit(x)),
                Tree::Group(x) => {
                    let beg = ys.len();
                    ys.push(Entry::End(0));
                    Self::recursive_new(ys, x.stream());
                    let end = ys.len();
                    ys.push(Entry::End(-(end as isize)));
                    let len = end - beg;
                    ys[beg] = Entry::Group(x, len);
                },
            }
        }
    }
    pub fn new(x: pm2::Stream) -> Self {
        Self::new2(x.into())
    }
    pub fn new2(x: pm2::Stream) -> Self {
        let mut ys = Vec::new();
        Self::recursive_new(&mut ys, x);
        ys.push(Entry::End(-(ys.len() as isize)));
        Self {
            entries: ys.into_boxed_slice(),
        }
    }
    pub fn begin(&self) -> Cursor {
        let y = self.entries.as_ptr();
        unsafe { Cursor::create(y, y.add(self.entries.len() - 1)) }
    }
}

fn same_scope(a: Cursor, b: Cursor) -> bool {
    a.scope == b.scope
}
fn same_buff(a: Cursor, b: Cursor) -> bool {
    buff_start(a) == buff_start(b)
}
fn buff_start(c: Cursor) -> *const Entry {
    unsafe {
        match &*c.scope {
            Entry::End(x) => c.scope.offset(*x),
            _ => unreachable!(),
        }
    }
}
fn cmp_assuming_same_buffer(a: Cursor, b: Cursor) -> Ordering {
    a.ptr.cmp(&b.ptr)
}
fn open_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_open(),
        _ => c.span(),
    }
}
fn close_span_of_group(c: Cursor) -> Span {
    match c.entry() {
        Entry::Group(x, _) => x.span_close(),
        _ => c.span(),
    }
}
