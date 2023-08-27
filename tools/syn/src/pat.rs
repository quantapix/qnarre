use super::*;
pub use expr::{Const, Lit, Mac, Member, Path, Range};

enum_of_structs! {
    pub enum Pat {
        Const(Const),
        Ident(Ident),
        Lit(Lit),
        Mac(Mac),
        Or(Or),
        Parenth(Parenth),
        Path(Path),
        Range(Range),
        Ref(Ref),
        Rest(Rest),
        Slice(Slice),
        Struct(Struct),
        Tuple(Tuple),
        TupleStruct(TupleStruct),
        Type(Type),
        Verbatim(Verbatim),
        Wild(Wild),
    }
}
impl Pat {
    pub fn parse_one(s: Stream) -> Res<Self> {
        let beg = s.fork();
        let look = s.look1();
        if look.peek(ident::Ident)
            && (s.peek2(Token![::])
                || s.peek2(Token![!])
                || s.peek2(tok::Brace)
                || s.peek2(tok::Parenth)
                || s.peek2(Token![..]))
            || s.peek(Token![self]) && s.peek2(Token![::])
            || look.peek(Token![::])
            || look.peek(Token![<])
            || s.peek(Token![Self])
            || s.peek(Token![super])
            || s.peek(Token![crate])
        {
            parse_path_or_mac_or_struct_or_range(s)
        } else if look.peek(Token![_]) {
            s.call(parse_wild).map(Pat::Wild)
        } else if s.peek(Token![box]) {
            parse_verbatim(beg, s)
        } else if s.peek(Token![-]) || look.peek(lit::Lit) || look.peek(Token![const]) {
            parse_lit_or_range(s)
        } else if look.peek(Token![ref]) || look.peek(Token![mut]) || s.peek(Token![self]) || s.peek(ident::Ident) {
            s.call(parse_ident).map(Pat::Ident)
        } else if look.peek(Token![&]) {
            s.call(parse_ref).map(Pat::Ref)
        } else if look.peek(tok::Parenth) {
            s.call(parse_paren_or_tuple)
        } else if look.peek(tok::Bracket) {
            s.call(parse_slice).map(Pat::Slice)
        } else if look.peek(Token![..]) && !s.peek(Token![...]) {
            parse_range_half_open(s)
        } else if look.peek(Token![const]) {
            s.call(parse_const).map(Pat::Verbatim)
        } else {
            Err(look.error())
        }
    }
    pub fn parse_many(s: Stream) -> Res<Self> {
        parse_many(s, None)
    }
    pub fn parse_with_vert(s: Stream) -> Res<Self> {
        let vert: Option<Token![|]> = s.parse()?;
        parse_many(s, vert)
    }
}
impl Clone for Pat {
    fn clone(&self) -> Self {
        use Pat::*;
        match self {
            Const(x) => Const(x.clone()),
            Ident(x) => Ident(x.clone()),
            Lit(x) => Lit(x.clone()),
            Mac(x) => Mac(x.clone()),
            Or(x) => Or(x.clone()),
            Parenth(x) => Parenth(x.clone()),
            Path(x) => Path(x.clone()),
            Range(x) => Range(x.clone()),
            Ref(x) => Ref(x.clone()),
            Rest(x) => Rest(x.clone()),
            Slice(x) => Slice(x.clone()),
            Struct(x) => Struct(x.clone()),
            Tuple(x) => Tuple(x.clone()),
            TupleStruct(x) => TupleStruct(x.clone()),
            Type(x) => Type(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
            Wild(x) => Wild(x.clone()),
        }
    }
}
impl Debug for Pat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("pat::Pat::")?;
        use Pat::*;
        match self {
            Const(x) => x.debug(f, "Const"),
            Ident(x) => x.debug(f, "Ident"),
            Lit(x) => x.debug(f, "Lit"),
            Mac(x) => x.debug(f, "Macro"),
            Or(x) => x.debug(f, "Or"),
            Parenth(x) => x.debug(f, "Parenth"),
            Path(x) => x.debug(f, "Path"),
            Range(x) => x.debug(f, "Range"),
            Ref(x) => x.debug(f, "Reference"),
            Rest(x) => x.debug(f, "Rest"),
            Slice(x) => x.debug(f, "Slice"),
            Struct(x) => x.debug(f, "Struct"),
            Tuple(x) => x.debug(f, "Tuple"),
            TupleStruct(x) => x.debug(f, "TupleStruct"),
            Type(x) => x.debug(f, "Type"),
            Verbatim(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
            Wild(x) => x.debug(f, "Wild"),
        }
    }
}
impl Eq for Pat {}
impl PartialEq for Pat {
    fn eq(&self, x: &Self) -> bool {
        use Pat::*;
        match (self, x) {
            (Const(x), Const(y)) => x == y,
            (Ident(x), Ident(y)) => x == y,
            (Lit(x), Lit(y)) => x == y,
            (Mac(x), Mac(y)) => x == y,
            (Or(x), Or(y)) => x == y,
            (Parenth(x), Parenth(y)) => x == y,
            (Path(x), Path(y)) => x == y,
            (Range(x), Range(y)) => x == y,
            (Ref(x), Ref(y)) => x == y,
            (Rest(x), Rest(y)) => x == y,
            (Slice(x), Slice(y)) => x == y,
            (Struct(x), Struct(y)) => x == y,
            (Tuple(x), Tuple(y)) => x == y,
            (TupleStruct(x), TupleStruct(y)) => x == y,
            (Type(x), Type(y)) => x == y,
            (Verbatim(x), Verbatim(y)) => StreamHelper(x) == StreamHelper(y),
            (Wild(x), Wild(y)) => x == y,
            _ => false,
        }
    }
}
impl Pretty for Pat {
    fn pretty(&self, p: &mut Print) {
        use Pat::*;
        match self {
            Const(x) => x.pretty(p),
            Ident(x) => x.pretty(p),
            Lit(x) => x.pretty(p),
            Mac(x) => x.pretty(p),
            Or(x) => x.pretty(p),
            Parenth(x) => x.pretty(p),
            Path(x) => x.pretty(p),
            Range(x) => p.expr_range(x),
            Ref(x) => x.pretty(p),
            Rest(x) => x.pretty(p),
            Slice(x) => x.pretty(p),
            Struct(x) => x.pretty(p),
            Tuple(x) => x.pretty(p),
            TupleStruct(x) => x.pretty(p),
            Type(x) => x.pretty(p),
            Verbatim(x) => x.pretty(p),
            Wild(x) => x.pretty(p),
        }
    }
}
impl<F: Folder + ?Sized> Fold for Pat {
    fn fold(&self, f: &mut F) {
        use Pat::*;
        match self {
            Const(x) => Const(x.fold(f)),
            Ident(x) => Ident(x.fold(f)),
            Lit(x) => Lit(x.fold(f)),
            Mac(x) => Mac(x.fold(f)),
            Or(x) => Or(x.fold(f)),
            Parenth(x) => Parenth(x.fold(f)),
            Path(x) => Path(x.fold(f)),
            Range(x) => Range(x.fold(f)),
            Ref(x) => Ref(x.fold(f)),
            Rest(x) => Rest(x.fold(f)),
            Slice(x) => Slice(x.fold(f)),
            Struct(x) => Struct(x.fold(f)),
            Tuple(x) => Tuple(x.fold(f)),
            TupleStruct(x) => TupleStruct(x.fold(f)),
            Type(x) => Type(x.fold(f)),
            Verbatim(x) => Verbatim(x),
            Wild(x) => Wild(x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Pat {
    fn hash(&self, h: &mut H) {
        use Pat::*;
        match self {
            Const(x) => {
                h.write_u8(0u8);
                x.hash(h);
            },
            Ident(x) => {
                h.write_u8(1u8);
                x.hash(h);
            },
            Lit(x) => {
                h.write_u8(2u8);
                x.hash(h);
            },
            Mac(x) => {
                h.write_u8(3u8);
                x.hash(h);
            },
            Or(x) => {
                h.write_u8(4u8);
                x.hash(h);
            },
            Parenth(x) => {
                h.write_u8(5u8);
                x.hash(h);
            },
            Path(x) => {
                h.write_u8(6u8);
                x.hash(h);
            },
            Range(x) => {
                h.write_u8(7u8);
                x.hash(h);
            },
            Ref(x) => {
                h.write_u8(8u8);
                x.hash(h);
            },
            Rest(x) => {
                h.write_u8(9u8);
                x.hash(h);
            },
            Slice(x) => {
                h.write_u8(10u8);
                x.hash(h);
            },
            Struct(x) => {
                h.write_u8(11u8);
                x.hash(h);
            },
            Tuple(x) => {
                h.write_u8(12u8);
                x.hash(h);
            },
            TupleStruct(x) => {
                h.write_u8(13u8);
                x.hash(h);
            },
            Type(x) => {
                h.write_u8(14u8);
                x.hash(h);
            },
            Verbatim(x) => {
                h.write_u8(15u8);
                StreamHelper(x).hash(h);
            },
            Wild(x) => {
                h.write_u8(16u8);
                x.hash(h);
            },
        }
    }
}
impl<V: Visitor + ?Sized> Visit for Pat {
    fn visit(&self, v: &mut V) {
        use Pat::*;
        match self {
            Const(x) => {
                x.visit(v);
            },
            Ident(x) => {
                x.visit(v);
            },
            Lit(x) => {
                x.visit(v);
            },
            Mac(x) => {
                x.visit(v);
            },
            Or(x) => {
                x.visit(v);
            },
            Parenth(x) => {
                x.visit(v);
            },
            Path(x) => {
                x.visit(v);
            },
            Range(x) => {
                x.visit(v);
            },
            Ref(x) => {
                x.visit(v);
            },
            Rest(x) => {
                x.visit(v);
            },
            Slice(x) => {
                x.visit(v);
            },
            Struct(x) => {
                x.visit(v);
            },
            Tuple(x) => {
                x.visit(v);
            },
            TupleStruct(x) => {
                x.visit(v);
            },
            Type(x) => {
                x.visit(v);
            },
            Verbatim(_) => {},
            Wild(x) => {
                x.visit(v);
            },
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        use Pat::*;
        match self {
            Const(x) => {
                x.visit_mut(v);
            },
            Ident(x) => {
                x.visit_mut(v);
            },
            Lit(x) => {
                x.visit_mut(v);
            },
            Mac(x) => {
                x.visit_mut(v);
            },
            Or(x) => {
                x.visit_mut(v);
            },
            Parenth(x) => {
                x.visit_mut(v);
            },
            Path(x) => {
                x.visit_mut(v);
            },
            Range(x) => {
                x.visit_mut(v);
            },
            Ref(x) => {
                x.visit_mut(v);
            },
            Rest(x) => {
                x.visit_mut(v);
            },
            Slice(x) => {
                x.visit_mut(v);
            },
            Struct(x) => {
                x.visit_mut(v);
            },
            Tuple(x) => {
                x.visit_mut(v);
            },
            TupleStruct(x) => {
                x.visit_mut(v);
            },
            Type(x) => {
                x.visit_mut(v);
            },
            Verbatim(_) => {},
            Wild(x) => {
                x.visit_mut(v);
            },
        }
    }
}

pub struct Ident {
    pub attrs: Vec<attr::Attr>,
    pub ref_: Option<Token![ref]>,
    pub mut_: Option<Token![mut]>,
    pub ident: super::Ident,
    pub sub: Option<(Token![@], Box<Pat>)>,
}
impl Lower for Ident {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.ref_.lower(s);
        self.mut_.lower(s);
        self.ident.lower(s);
        if let Some((at_, sub)) = &self.sub {
            at_.lower(s);
            sub.lower(s);
        }
    }
}
impl Clone for Ident {
    fn clone(&self) -> Self {
        Ident {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            sub: self.sub.clone(),
        }
    }
}
impl Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Ident {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("ref_", &self.ref_);
                f.field("mut_", &self.mut_);
                f.field("ident", &self.ident);
                f.field("sub", &self.sub);
                f.finish()
            }
        }
        self.debug(f, "pat::Ident")
    }
}
impl Eq for Ident {}
impl PartialEq for Ident {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.ref_ == x.ref_
            && self.mut_ == x.mut_
            && self.ident == x.ident
            && self.sub == x.sub
    }
}
impl Pretty for Ident {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if self.ref_.is_some() {
            p.word("ref ");
        }
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.ident.pretty(p);
        if let Some((_, x)) = &self.sub {
            p.word(" @ ");
            x.pretty(p);
        }
    }
}
impl<F: Folder + ?Sized> Fold for Ident {
    fn fold(&self, f: &mut F) {
        Ident {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            ref_: self.ref_,
            mut_: self.mut_,
            ident: self.ident.fold(f),
            sub: (self.sub).map(|x| ((x).0, Box::new(*(x).1.fold(f)))),
        }
    }
}
impl<H: Hasher> Hash for Ident {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.ref_.hash(h);
        self.mut_.hash(h);
        self.ident.hash(h);
        self.sub.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Ident {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.ident.visit(v);
        if let Some(x) = &self.sub {
            &*(x).1.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.ident.visit_mut(v);
        if let Some(x) = &mut self.sub {
            &mut *(x).1.visit_mut(v);
        }
    }
}

pub struct Or {
    pub attrs: Vec<attr::Attr>,
    pub vert: Option<Token![|]>,
    pub cases: Puncted<Pat, Token![|]>,
}
impl Lower for Or {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.vert.lower(s);
        self.cases.lower(s);
    }
}
impl Clone for Or {
    fn clone(&self) -> Self {
        Or {
            attrs: self.attrs.clone(),
            vert: self.vert.clone(),
            cases: self.cases.clone(),
        }
    }
}
impl Debug for Or {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Or {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("vert", &self.vert);
                f.field("cases", &self.cases);
                f.finish()
            }
        }
        self.debug(f, "pat::Or")
    }
}
impl Eq for Or {}
impl PartialEq for Or {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.vert == x.vert && self.cases == x.cases
    }
}
impl Pretty for Or {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        let mut break_ = false;
        for x in &self.cases {
            match x {
                Pat::Lit(_) | Pat::Wild(_) => {},
                _ => {
                    break_ = true;
                    break;
                },
            }
        }
        if break_ {
            p.cbox(0);
        } else {
            p.ibox(0);
        }
        for x in self.cases.iter().delimited() {
            if !x.is_first {
                p.space();
                p.word("| ");
            }
            &x.pretty(p);
        }
        p.end();
    }
}
impl<F: Folder + ?Sized> Fold for Or {
    fn fold(&self, f: &mut F) {
        Or {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            vert: self.vert,
            cases: FoldHelper::lift(self.cases, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Or {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.vert.hash(h);
        self.cases.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Or {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        for y in Puncted::pairs(&self.cases) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        for mut y in Puncted::pairs_mut(&mut self.cases) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Parenth {
    pub attrs: Vec<attr::Attr>,
    pub parenth: tok::Parenth,
    pub pat: Box<Pat>,
}
impl Lower for Parenth {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.parenth.surround(s, |s| {
            self.pat.lower(s);
        });
    }
}
impl Clone for Parenth {
    fn clone(&self) -> Self {
        Parenth {
            attrs: self.attrs.clone(),
            parenth: self.parenth.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Debug for Parenth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Parenth {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("parenth", &self.parenth);
                f.field("pat", &self.pat);
                f.finish()
            }
        }
        self.debug(f, "pat::Parenth")
    }
}
impl Eq for Parenth {}
impl PartialEq for Parenth {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat
    }
}
impl Pretty for Parenth {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        &self.pat.pretty(p);
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for Parenth {
    fn fold(&self, f: &mut F) {
        Parenth {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Parenth {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Parenth {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &*self.pat.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut *self.pat.visit_mut(v);
    }
}

pub struct Ref {
    pub attrs: Vec<attr::Attr>,
    pub and: Token![&],
    pub mut_: Option<Token![mut]>,
    pub pat: Box<Pat>,
}
impl Lower for Ref {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.and.lower(s);
        self.mut_.lower(s);
        self.pat.lower(s);
    }
}
impl Clone for Ref {
    fn clone(&self) -> Self {
        Ref {
            attrs: self.attrs.clone(),
            and: self.and.clone(),
            mut_: self.mut_.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Debug for Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Ref {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("and", &self.and);
                f.field("mut_", &self.mut_);
                f.field("pat", &self.pat);
                f.finish()
            }
        }
        self.debug(f, "pat::Ref")
    }
}
impl Eq for Ref {}
impl PartialEq for Ref {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.mut_ == x.mut_ && self.pat == x.pat
    }
}
impl Pretty for Ref {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("&");
        if self.mut_.is_some() {
            p.word("mut ");
        }
        &self.pat.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Ref {
    fn fold(&self, f: &mut F) {
        Ref {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            and: self.and,
            mut_: self.mut_,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Ref {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.mut_.hash(h);
        self.pat.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Ref {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &*self.pat.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut *self.pat.visit_mut(v);
    }
}

pub struct Rest {
    pub attrs: Vec<attr::Attr>,
    pub dot2: Token![..],
}
impl Lower for Rest {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.dot2.lower(s);
    }
}
impl Clone for Rest {
    fn clone(&self) -> Self {
        Rest {
            attrs: self.attrs.clone(),
            dot2: self.dot2.clone(),
        }
    }
}
impl Debug for Rest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Rest {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("dot2", &self.dot2);
                f.finish()
            }
        }
        self.debug(f, "pat::Rest")
    }
}
impl Eq for Rest {}
impl PartialEq for Rest {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
    }
}
impl Pretty for Rest {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("..");
    }
}
impl<F: Folder + ?Sized> Fold for Rest {
    fn fold(&self, f: &mut F) {
        Rest {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            dot2: self.dot2,
        }
    }
}
impl<H: Hasher> Hash for Rest {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Rest {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
    }
}

pub struct Slice {
    pub attrs: Vec<attr::Attr>,
    pub bracket: tok::Bracket,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for Slice {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.bracket.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Clone for Slice {
    fn clone(&self) -> Self {
        Slice {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Slice {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("bracket", &self.bracket);
                f.field("elems", &self.pats);
                f.finish()
            }
        }
        self.debug(f, "pat::Slice")
    }
}
impl Eq for Slice {}
impl PartialEq for Slice {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pats == x.pats
    }
}
impl Pretty for Slice {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("[");
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.word("]");
    }
}
impl<F: Folder + ?Sized> Fold for Slice {
    fn fold(&self, f: &mut F) {
        Slice {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            bracket: self.bracket,
            pats: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Slice {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Slice {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        for y in Puncted::pairs(&self.pats) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        for mut y in Puncted::pairs_mut(&mut self.pats) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Struct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: Path,
    pub brace: tok::Brace,
    pub fields: Puncted<Field, Token![,]>,
    pub rest: Option<Rest>,
}
impl Lower for Struct {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        path::path_lower(s, &self.qself, &self.path);
        self.brace.surround(s, |s| {
            self.fields.lower(s);
            if !self.fields.empty_or_trailing() && self.rest.is_some() {
                <Token![,]>::default().lower(s);
            }
            self.rest.lower(s);
        });
    }
}
impl Clone for Struct {
    fn clone(&self) -> Self {
        Struct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace: self.brace.clone(),
            fields: self.fields.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Debug for Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Struct {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.field("brace", &self.brace);
                f.field("fields", &self.fields);
                f.field("rest", &self.rest);
                f.finish()
            }
        }
        self.debug(f, "pat::Struct")
    }
}
impl Eq for Struct {}
impl PartialEq for Struct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
            && self.qself == x.qself
            && self.path == x.path
            && self.fields == x.fields
            && self.rest == x.rest
    }
}
impl Pretty for Struct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.cbox(INDENT);
        &self.path.pretty_with_args(p, path::Kind::Expr);
        p.word(" {");
        p.space_if_nonempty();
        for x in self.fields.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma_or_space(x.is_last && self.rest.is_none());
        }
        if let Some(x) = &self.rest {
            x.pretty(p);
            p.space();
        }
        p.offset(-INDENT);
        p.end();
        p.word("}");
    }
}
impl<F: Folder + ?Sized> Fold for Struct {
    fn fold(&self, f: &mut F) {
        Struct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
            brace: self.brace,
            fields: FoldHelper::lift(self.fields, |x| x.fold(f)),
            rest: (self.rest).map(|x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Struct {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.fields.hash(h);
        self.rest.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Struct {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.qself {
            x.visit(v);
        }
        &self.path.visit(v);
        for y in Puncted::pairs(&self.fields) {
            let x = y.value();
            x.visit(v);
        }
        if let Some(x) = &self.rest {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.qself {
            x.visit_mut(v);
        }
        &mut self.path.visit_mut(v);
        for mut y in Puncted::pairs_mut(&mut self.fields) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.rest {
            x.visit_mut(v);
        }
    }
}

pub struct Field {
    pub attrs: Vec<attr::Attr>,
    pub memb: Member,
    pub colon: Option<Token![:]>,
    pub pat: Box<Pat>,
}
impl Lower for Field {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        if let Some(x) = &self.colon {
            self.memb.lower(s);
            x.lower(s);
        }
        self.pat.lower(s);
    }
}
impl Clone for Field {
    fn clone(&self) -> Self {
        Field {
            attrs: self.attrs.clone(),
            memb: self.memb.clone(),
            colon: self.colon.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Debug for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("pat::Field");
        f.field("attrs", &self.attrs);
        f.field("member", &self.memb);
        f.field("colon", &self.colon);
        f.field("pat", &self.pat);
        f.finish()
    }
}
impl Eq for Field {}
impl PartialEq for Field {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.memb == x.memb && self.colon == x.colon && self.pat == x.pat
    }
}
impl Pretty for Field {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        if self.colon.is_some() {
            &self.memb.pretty(p);
            p.word(": ");
        }
        &self.pat.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Field {
    fn fold(&self, f: &mut F) {
        Field {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            memb: self.memb.fold(f),
            colon: self.colon,
            pat: Box::new(*self.pat.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Field {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.memb.hash(h);
        self.colon.hash(h);
        self.pat.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Field {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &self.memb.visit(v);
        &*self.pat.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut self.memb.visit_mut(v);
        &mut *self.pat.visit_mut(v);
    }
}

impl Member {
    fn is_unnamed(&self) -> bool {
        match self {
            Member::Named(_) => false,
            Member::Unnamed(_) => true,
        }
    }
}

pub struct Tuple {
    pub attrs: Vec<attr::Attr>,
    pub parenth: tok::Parenth,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for Tuple {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.parenth.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Clone for Tuple {
    fn clone(&self) -> Self {
        Tuple {
            attrs: self.attrs.clone(),
            parenth: self.parenth.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Debug for Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Tuple {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("parenth", &self.parenth);
                f.field("pats", &self.pats);
                f.finish()
            }
        }
        self.debug(f, "pat::Tuple")
    }
}
impl Eq for Tuple {}
impl PartialEq for Tuple {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pats == x.pats
    }
}
impl Pretty for Tuple {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            if self.pats.len() == 1 {
                if self.pats.trailing_punct() {
                    p.word(",");
                }
                p.zerobreak();
            } else {
                p.trailing_comma(x.is_last);
            }
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for Tuple {
    fn fold(&self, f: &mut F) {
        Tuple {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            parenth: self.parenth,
            pats: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Tuple {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pats.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Tuple {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        for y in Puncted::pairs(&self.pats) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        for mut y in Puncted::pairs_mut(&mut self.pats) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct TupleStruct {
    pub attrs: Vec<attr::Attr>,
    pub qself: Option<path::QSelf>,
    pub path: Path,
    pub parenth: tok::Parenth,
    pub pats: Puncted<Pat, Token![,]>,
}
impl Lower for TupleStruct {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        path::path_lower(s, &self.qself, &self.path);
        self.parenth.surround(s, |s| {
            self.pats.lower(s);
        });
    }
}
impl Clone for TupleStruct {
    fn clone(&self) -> Self {
        TupleStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            parenth: self.parenth.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Debug for TupleStruct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl TupleStruct {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.field("parenth", &self.parenth);
                f.field("pats", &self.pats);
                f.finish()
            }
        }
        self.debug(f, "pat::TupleStructuct")
    }
}
impl Eq for TupleStruct {}
impl PartialEq for TupleStruct {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.qself == x.qself && self.path == x.path && self.pats == x.pats
    }
}
impl Pretty for TupleStruct {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.path.pretty_with_args(p, path::Kind::Expr);
        p.word("(");
        p.cbox(INDENT);
        p.zerobreak();
        for x in self.pats.iter().delimited() {
            &x.pretty(p);
            p.trailing_comma(x.is_last);
        }
        p.offset(-INDENT);
        p.end();
        p.word(")");
    }
}
impl<F: Folder + ?Sized> Fold for TupleStruct {
    fn fold(&self, f: &mut F) {
        TupleStruct {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            qself: (self.qself).map(|x| x.fold(f)),
            path: self.path.fold(f),
            parenth: self.parenth,
            pats: FoldHelper::lift(self.pats, |x| x.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for TupleStruct {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.qself.hash(h);
        self.path.hash(h);
        self.pats.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for TupleStruct {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        if let Some(x) = &self.qself {
            x.visit(v);
        }
        &self.path.visit(v);
        for y in Puncted::pairs(&self.pats) {
            let x = y.value();
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        if let Some(x) = &mut self.qself {
            x.visit_mut(v);
        }
        &mut self.path.visit_mut(v);
        for mut y in Puncted::pairs_mut(&mut self.pats) {
            let x = y.value_mut();
            x.visit_mut(v);
        }
    }
}

pub struct Type {
    pub attrs: Vec<attr::Attr>,
    pub pat: Box<Pat>,
    pub colon: Token![:],
    pub typ: Box<typ::Type>,
}
impl Lower for Type {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.pat.lower(s);
        self.colon.lower(s);
        self.typ.lower(s);
    }
}
impl Clone for Type {
    fn clone(&self) -> Self {
        Type {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Type {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("pat", &self.pat);
                f.field("colon", &self.colon);
                f.field("typ", &self.typ);
                f.finish()
            }
        }
        self.debug(f, "pat::Type")
    }
}
impl Eq for Type {}
impl PartialEq for Type {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs && self.pat == x.pat && self.typ == x.typ
    }
}
impl Pretty for Type {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        &self.pat.pretty(p);
        p.word(": ");
        &self.typ.pretty(p);
    }
}
impl<F: Folder + ?Sized> Fold for Type {
    fn fold(&self, f: &mut F) {
        Type {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            pat: Box::new(*self.pat.fold(f)),
            colon: self.colon,
            typ: Box::new(*self.typ.fold(f)),
        }
    }
}
impl<H: Hasher> Hash for Type {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
        self.pat.hash(h);
        self.typ.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Type {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
        &*self.pat.visit(v);
        &*self.typ.visit(v);
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
        &mut *self.pat.visit_mut(v);
        &mut *self.typ.visit_mut(v);
    }
}

pub struct Wild {
    pub attrs: Vec<attr::Attr>,
    pub underscore: Token![_],
}
impl Lower for Wild {
    fn lower(&self, s: &mut Stream) {
        s.append_all(self.attrs.outers());
        self.underscore.lower(s);
    }
}
impl Clone for Wild {
    fn clone(&self) -> Self {
        Wild {
            attrs: self.attrs.clone(),
            underscore: self.underscore.clone(),
        }
    }
}
impl Debug for Wild {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Wild {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("attrs", &self.attrs);
                f.field("underscore", &self.underscore);
                f.finish()
            }
        }
        self.debug(f, "pat::Wild")
    }
}
impl Eq for Wild {}
impl PartialEq for Wild {
    fn eq(&self, x: &Self) -> bool {
        self.attrs == x.attrs
    }
}
impl Pretty for Wild {
    fn pretty(&self, p: &mut Print) {
        p.outer_attrs(&self.attrs);
        p.word("_");
    }
}
impl<F: Folder + ?Sized> Fold for Wild {
    fn fold(&self, f: &mut F) {
        Wild {
            attrs: FoldHelper::lift(self.attrs, |x| x.fold(f)),
            underscore: self.underscore,
        }
    }
}
impl<H: Hasher> Hash for Wild {
    fn hash(&self, h: &mut H) {
        self.attrs.hash(h);
    }
}
impl<V: Visitor + ?Sized> Visit for Wild {
    fn visit(&self, v: &mut V) {
        for x in &self.attrs {
            x.visit(v);
        }
    }
    fn visit_mut(&mut self, v: &mut V) {
        for x in &mut self.attrs {
            x.visit_mut(v);
        }
    }
}

pub struct Verbatim(pub pm2::Stream);
impl Pretty for Verbatim {
    fn pretty(&self, p: &mut Print) {
        enum Type {
            Box(Pat),
            Const(Const),
            Ellipsis,
        }
        use Type::*;
        struct Const {
            attrs: Vec<attr::Attr>,
            block: stmt::Block,
        }
        impl parse::Parse for Type {
            fn parse(s: parse::Stream) -> Res<Self> {
                let look = s.lookahead1();
                if look.peek(Token![box]) {
                    s.parse::<Token![box]>()?;
                    let y = Pat::parse_single(s)?;
                    Ok(Box(y))
                } else if look.peek(Token![const]) {
                    s.parse::<Token![const]>()?;
                    let y;
                    let brace = braced!(y in s);
                    let attrs = y.call(attr::Attr::parse_inner)?;
                    let stmts = y.call(stmt::Block::parse_within)?;
                    Ok(Const(Const {
                        attrs,
                        block: stmt::Block { brace, stmts },
                    }))
                } else if look.peek(Token![...]) {
                    s.parse::<Token![...]>()?;
                    Ok(Ellipsis)
                } else {
                    Err(look.error())
                }
            }
        }
        let y: Type = match parse2(self.clone()) {
            Ok(x) => x,
            Err(_) => unimplemented!("Pat::Stream `{}`", self),
        };
        match y {
            Ellipsis => {
                p.word("...");
            },
            Box(x) => {
                p.word("box ");
                &x.pretty(p);
            },
            Const(x) => {
                p.word("const ");
                p.cbox(INDENT);
                p.small_block(&x.block, &x.attrs);
                p.end();
            },
        }
    }
}
enum RangeBound {
    Const(Const),
    Lit(Lit),
    Path(Path),
}
impl RangeBound {
    fn into_expr(self) -> Box<expr::Expr> {
        use RangeBound::*;
        Box::new(match self {
            Const(x) => expr::Expr::Const(x),
            Lit(x) => expr::Expr::Lit(x),
            Path(x) => expr::Expr::Path(x),
        })
    }
    fn into_pat(self) -> Pat {
        use RangeBound::*;
        match self {
            Const(x) => Pat::Const(x),
            Lit(x) => Pat::Lit(x),
            Path(x) => Pat::Path(x),
        }
    }
}

fn parse_const(s: Stream) -> Res<pm2::Stream> {
    let beg = s.fork();
    s.parse::<Token![const]>()?;
    let y;
    braced!(y in s);
    y.call(attr::Attr::parse_inners)?;
    y.call(stmt::Block::parse_within)?;
    Ok(parse::parse_verbatim(&beg, s))
}
fn parse_field(s: Stream) -> Res<Field> {
    let beg = s.fork();
    let box_: Option<Token![box]> = s.parse()?;
    let ref_: Option<Token![ref]> = s.parse()?;
    let mut_: Option<Token![mut]> = s.parse()?;
    let memb = if box_.is_some() || ref_.is_some() || mut_.is_some() {
        s.parse().map(Member::Named)
    } else {
        s.parse()
    }?;
    if box_.is_none() && ref_.is_none() && mut_.is_none() && s.peek(Token![:]) || memb.is_unnamed() {
        return Ok(Field {
            attrs: Vec::new(),
            memb,
            colon: Some(s.parse()?),
            pat: Box::new(Pat::parse_many(s)?),
        });
    }
    let ident = match memb {
        Member::Named(x) => x,
        Member::Unnamed(_) => unreachable!(),
    };
    let pat = if box_.is_some() {
        Pat::Verbatim(parse::parse_verbatim(&beg, s))
    } else {
        Pat::Ident(Ident {
            attrs: Vec::new(),
            ref_,
            mut_,
            ident: ident.clone(),
            sub: None,
        })
    };
    Ok(Field {
        attrs: Vec::new(),
        memb: Member::Named(ident),
        colon: None,
        pat: Box::new(pat),
    })
}
fn parse_ident(s: Stream) -> Res<Ident> {
    Ok(Ident {
        attrs: Vec::new(),
        ref_: s.parse()?,
        mut_: s.parse()?,
        ident: s.call(Ident::parse_any)?,
        sub: {
            if s.peek(Token![@]) {
                let at_: Token![@] = s.parse()?;
                let y = Pat::parse_one(s)?;
                Some((at_, Box::new(y)))
            } else {
                None
            }
        },
    })
}
fn parse_lit_or_range(s: Stream) -> Res<Pat> {
    let beg = s.call(parse_range_bound)?.unwrap();
    if s.peek(Token![..]) {
        let limits = expr::Limits::parse_obsolete(s)?;
        let end = s.call(parse_range_bound)?;
        if let (expr::Limits::Closed(_), None) = (&limits, &end) {
            return Err(s.error("expected range upper bound"));
        }
        Ok(Pat::Range(expr::Range {
            attrs: Vec::new(),
            beg: Some(beg.into_expr()),
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    } else {
        Ok(beg.into_pat())
    }
}
fn parse_many(s: Stream, vert: Option<Token![|]>) -> Res<Pat> {
    let mut y = Pat::parse_one(s)?;
    if vert.is_some() || s.peek(Token![|]) && !s.peek(Token![||]) && !s.peek(Token![|=]) {
        let mut cases = Puncted::new();
        cases.push_value(y);
        while s.peek(Token![|]) && !s.peek(Token![||]) && !s.peek(Token![|=]) {
            let x = s.parse()?;
            cases.push_punct(x);
            let x = Pat::parse_one(s)?;
            cases.push_value(x);
        }
        y = Pat::Or(Or {
            attrs: Vec::new(),
            vert,
            cases,
        });
    }
    Ok(y)
}
fn parse_path_or_mac_or_struct_or_range(s: Stream) -> Res<Pat> {
    let (qself, path) = path::qpath(s, true)?;
    if qself.is_none() && s.peek(Token![!]) && !s.peek(Token![!=]) && path.is_mod_style() {
        let bang: Token![!] = s.parse()?;
        let (delim, toks) = tok::parse_delim(s)?;
        return Ok(Pat::Mac(expr::Mac {
            attrs: Vec::new(),
            mac: mac::Mac {
                path,
                bang,
                delim,
                toks,
            },
        }));
    }
    if s.peek(tok::Brace) {
        parse_struct(s, qself, path).map(Pat::Struct)
    } else if s.peek(tok::Parenth) {
        parse_tuple_struct(s, qself, path).map(Pat::TupleStruct)
    } else if s.peek(Token![..]) {
        parse_range(s, qself, path)
    } else {
        Ok(Pat::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))
    }
}
fn parse_paren_or_tuple(s: Stream) -> Res<Pat> {
    let y;
    let parenth = parenthed!(y in s);
    let mut pats = Puncted::new();
    while !y.is_empty() {
        let x = Pat::parse_many(&y)?;
        if y.is_empty() {
            if pats.is_empty() && !matches!(x, Pat::Rest(_)) {
                return Ok(Pat::Parenth(Parenth {
                    attrs: Vec::new(),
                    parenth,
                    pat: Box::new(x),
                }));
            }
            pats.push_value(x);
            break;
        }
        pats.push_value(x);
        let x = y.parse()?;
        pats.push_punct(x);
    }
    Ok(Pat::Tuple(Tuple {
        attrs: Vec::new(),
        parenth,
        pats,
    }))
}
fn parse_range(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<Pat> {
    let limits = expr::Limits::parse_obsolete(s)?;
    let end = s.call(parse_range_bound)?;
    if let (expr::Limits::Closed(_), None) = (&limits, &end) {
        return Err(s.error("expected range upper bound"));
    }
    Ok(Pat::Range(expr::Range {
        attrs: Vec::new(),
        beg: Some(Box::new(expr::Expr::Path(expr::Path {
            attrs: Vec::new(),
            qself,
            path,
        }))),
        limits,
        end: end.map(RangeBound::into_expr),
    }))
}
fn parse_range_bound(s: Stream) -> Res<Option<RangeBound>> {
    if s.is_empty()
        || s.peek(Token![|])
        || s.peek(Token![=])
        || s.peek(Token![:]) && !s.peek(Token![::])
        || s.peek(Token![,])
        || s.peek(Token![;])
        || s.peek(Token![if])
    {
        return Ok(None);
    }
    let look = s.look1();
    let y = if look.peek(lit::Lit) {
        RangeBound::Lit(s.parse()?)
    } else if look.peek(ident::Ident)
        || look.peek(Token![::])
        || look.peek(Token![<])
        || look.peek(Token![self])
        || look.peek(Token![Self])
        || look.peek(Token![super])
        || look.peek(Token![crate])
    {
        RangeBound::Path(s.parse()?)
    } else if look.peek(Token![const]) {
        RangeBound::Const(s.parse()?)
    } else {
        return Err(look.error());
    };
    Ok(Some(y))
}
fn parse_range_half_open(s: Stream) -> Res<Pat> {
    let limits: expr::Limits = s.parse()?;
    let end = s.call(parse_range_bound)?;
    if end.is_some() {
        Ok(Pat::Range(expr::Range {
            attrs: Vec::new(),
            beg: None,
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    } else {
        use expr::Limits::*;
        match limits {
            HalfOpen(dot2) => Ok(Pat::Rest(Rest {
                attrs: Vec::new(),
                dot2,
            })),
            Closed(_) => Err(s.error("expected range upper bound")),
        }
    }
}
fn parse_ref(s: Stream) -> Res<Ref> {
    Ok(Ref {
        attrs: Vec::new(),
        and: s.parse()?,
        mut_: s.parse()?,
        pat: Box::new(Pat::parse_one(s)?),
    })
}
fn parse_slice(s: Stream) -> Res<Slice> {
    let y;
    let bracket = bracketed!(y in s);
    let mut pats = Puncted::new();
    while !y.is_empty() {
        let y = Pat::parse_many(&y)?;
        match y {
            Pat::Range(x) if x.beg.is_none() || x.end.is_none() => {
                let (start, end) = match x.limits {
                    expr::Limits::HalfOpen(x) => (x.spans[0], x.spans[1]),
                    expr::Limits::Closed(x) => (x.spans[0], x.spans[2]),
                };
                let m = "range pattern is not allowed unparenthesized inside slice pattern";
                return Err(err::new2(start, end, m));
            },
            _ => {},
        }
        pats.push_value(y);
        if y.is_empty() {
            break;
        }
        let y = y.parse()?;
        pats.push_punct(y);
    }
    Ok(Slice {
        attrs: Vec::new(),
        bracket,
        pats,
    })
}
fn parse_struct(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<Struct> {
    let y;
    let brace = braced!(y in s);
    let mut fields = Puncted::new();
    let mut rest = None;
    while !y.is_empty() {
        let attrs = y.call(attr::Attr::parse_outers)?;
        if y.peek(Token![..]) {
            rest = Some(Rest {
                attrs,
                dot2: y.parse()?,
            });
            break;
        }
        let mut y = y.call(parse_field)?;
        y.attrs = attrs;
        fields.push_value(y);
        if y.is_empty() {
            break;
        }
        let y: Token![,] = y.parse()?;
        fields.push_punct(y);
    }
    Ok(Struct {
        attrs: Vec::new(),
        qself,
        path,
        brace,
        fields,
        rest,
    })
}
fn parse_tuple_struct(s: Stream, qself: Option<path::QSelf>, path: Path) -> Res<TupleStruct> {
    let y;
    let parenth = parenthed!(y in s);
    let mut pats = Puncted::new();
    while !y.is_empty() {
        let x = Pat::parse_many(&y)?;
        pats.push_value(x);
        if y.is_empty() {
            break;
        }
        let x = y.parse()?;
        pats.push_punct(x);
    }
    Ok(TupleStruct {
        attrs: Vec::new(),
        qself,
        path,
        parenth,
        pats,
    })
}
fn parse_verbatim(beg: parse::Buffer, s: Stream) -> Res<Pat> {
    s.parse::<Token![box]>()?;
    Pat::parse_one(s)?;
    Ok(Pat::Verbatim(parse::parse_verbatim(&beg, s)))
}
fn parse_wild(s: Stream) -> Res<Wild> {
    Ok(Wild {
        attrs: Vec::new(),
        underscore: s.parse()?,
    })
}
