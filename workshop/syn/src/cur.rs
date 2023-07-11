
enum Entry {
    Group(Group, usize),
    Ident(Ident),
    Punct(Punct),
    Literal(Literal),
    End(isize),
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
    pub fn group(mut self, d: Delimiter) -> Option<(Cursor<'a>, DelimSpan, Cursor<'a>)> {
        if d != Delimiter::None {
            self.ignore_none();
        }
        if let Entry::Group(x, end) = self.entry() {
            if x.delimiter() == d {
                let span = x.delim_span();
                let end_of_group = unsafe { self.ptr.add(*end) };
                let inside_of_group = unsafe { Cursor::create(self.ptr.add(1), end_of_group) };
                let after_group = unsafe { Cursor::create(end_of_group, self.scope) };
                return Some((inside_of_group, span, after_group));
            }
        }
        None
    }
    pub fn any_group(self) -> Option<(Cursor<'a>, Delimiter, DelimSpan, Cursor<'a>)> {
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
    pub fn any_group_token(self) -> Option<(Group, Cursor<'a>)> {
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
        use Entry::*;
        let (tree, len) = match self.entry() {
            Group(x, end) => (x.clone().into(), *end),
            Literal(x) => (x.clone().into(), 1),
            Ident(x) => (x.clone().into(), 1),
            Punct(x) => (x.clone().into(), 1),
            End(_) => return None,
        };
        let rest = unsafe { Cursor::create(self.ptr.add(len), self.scope) };
        Some((tree, rest))
    }
    pub fn span(self) -> Span {
        use Entry::*;
        match self.entry() {
            Group(x, _) => x.span(),
            Literal(x) => x.span(),
            Ident(x) => x.span(),
            Punct(x) => x.span(),
            End(_) => Span::call_site(),
        }
    }
    pub fn prev_span(mut self) -> Span {
        if buff_start(self) < self.ptr {
            self.ptr = unsafe { self.ptr.offset(-1) };
            if let Entry::End(_) = self.entry() {
                let mut depth = 1;
                loop {
                    self.ptr = unsafe { self.ptr.offset(-1) };
                    use Entry::*;
                    match self.entry() {
                        Group(x, _) => {
                            depth -= 1;
                            if depth == 0 {
                                return x.span();
                            }
                        },
                        End(_) => depth += 1,
                        Literal(_) | Ident(_) | Punct(_) => {},
                    }
                }
            }
        }
        self.span()
    }
    pub fn skip(self) -> Option<Cursor<'a>> {
        use Entry::*;
        let y = match self.entry() {
            End(_) => return None,
            Punct(x) if x.as_char() == '\'' && x.spacing() == Spacing::Joint => match unsafe { &*self.ptr.add(1) } {
                Ident(_) => 2,
                _ => 1,
            },
            Group(_, x) => *x,
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
pub struct Buffer {
    entries: Box<[Entry]>,
}
impl Buffer {
    fn recursive_new(ys: &mut Vec<Entry>, xs: TokenStream) {
        for x in xs {
            use Entry::*;
            match x {
                TokenTree::Ident(x) => ys.push(Ident(x)),
                TokenTree::Punct(x) => ys.push(Punct(x)),
                TokenTree::Literal(x) => ys.push(Literal(x)),
                TokenTree::Group(x) => {
                    let beg = ys.len();
                    ys.push(End(0));
                    Self::recursive_new(ys, x.stream());
                    let end = ys.len();
                    ys.push(End(-(end as isize)));
                    let off = end - beg;
                    ys[beg] = Group(x, off);
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
        let y = self.entries.as_ptr();
        unsafe { Cursor::create(y, y.add(self.entries.len() - 1)) }
    }
}
