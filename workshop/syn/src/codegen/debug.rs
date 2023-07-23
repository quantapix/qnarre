use crate::*;

impl Debug for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Abi");
        f.field("extern_", &self.extern_);
        f.field("name", &self.name);
        f.finish()
    }
}
impl Debug for AngledArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl AngledArgs {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("colon2", &self.colon2);
                f.field("lt", &self.lt);
                f.field("args", &self.args);
                f.field("gt", &self.gt);
                f.finish()
            }
        }
        self.debug(f, "path::path::AngledArgs")
    }
}
impl Debug for Arm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Arm");
        f.field("attrs", &self.attrs);
        f.field("pat", &self.pat);
        f.field("guard", &self.guard);
        f.field("fat_arrow", &self.fat_arrow);
        f.field("body", &self.body);
        f.field("comma", &self.comma);
        f.finish()
    }
}
impl Debug for AssocConst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("AssocConst");
        f.field("ident", &self.ident);
        f.field("gens", &self.gnrs);
        f.field("eq", &self.eq);
        f.field("value", &self.val);
        f.finish()
    }
}
impl Debug for AssocType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("AssocType");
        f.field("ident", &self.ident);
        f.field("gens", &self.gnrs);
        f.field("eq", &self.eq);
        f.field("ty", &self.ty);
        f.finish()
    }
}
impl Debug for attr::Style {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("attr::Style::")?;
        match self {
            attr::Style::Outer => f.write_str("Outer"),
            attr::Style::Inner(x) => {
                let mut f = f.debug_tuple("Inner");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for attr::Attr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("attr::Attr");
        f.field("pound", &self.pound);
        f.field("style", &self.style);
        f.field("bracket", &self.bracket);
        f.field("meta", &self.meta);
        f.finish()
    }
}
impl Debug for typ::FnArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::FnArg");
        f.field("attrs", &self.attrs);
        f.field("name", &self.name);
        f.field("ty", &self.typ);
        f.finish()
    }
}
impl Debug for typ::Variadic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("typ::Variadic");
        f.field("attrs", &self.attrs);
        f.field("name", &self.name);
        f.field("dots", &self.dots);
        f.field("comma", &self.comma);
        f.finish()
    }
}
impl Debug for expr::BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("BinOp::")?;
        use expr::BinOp::*;
        match self {
            Add(x) => {
                let mut f = f.debug_tuple("Add");
                f.field(x);
                f.finish()
            },
            Sub(x) => {
                let mut f = f.debug_tuple("Sub");
                f.field(x);
                f.finish()
            },
            Mul(x) => {
                let mut f = f.debug_tuple("Mul");
                f.field(x);
                f.finish()
            },
            Div(x) => {
                let mut f = f.debug_tuple("Div");
                f.field(x);
                f.finish()
            },
            Rem(x) => {
                let mut f = f.debug_tuple("Rem");
                f.field(x);
                f.finish()
            },
            And(x) => {
                let mut f = f.debug_tuple("And");
                f.field(x);
                f.finish()
            },
            Or(x) => {
                let mut f = f.debug_tuple("Or");
                f.field(x);
                f.finish()
            },
            BitXor(x) => {
                let mut f = f.debug_tuple("BitXor");
                f.field(x);
                f.finish()
            },
            BitAnd(x) => {
                let mut f = f.debug_tuple("BitAnd");
                f.field(x);
                f.finish()
            },
            BitOr(x) => {
                let mut f = f.debug_tuple("BitOr");
                f.field(x);
                f.finish()
            },
            Shl(x) => {
                let mut f = f.debug_tuple("Shl");
                f.field(x);
                f.finish()
            },
            Shr(x) => {
                let mut f = f.debug_tuple("Shr");
                f.field(x);
                f.finish()
            },
            Eq(x) => {
                let mut f = f.debug_tuple("Eq");
                f.field(x);
                f.finish()
            },
            Lt(x) => {
                let mut f = f.debug_tuple("Lt");
                f.field(x);
                f.finish()
            },
            Le(x) => {
                let mut f = f.debug_tuple("Le");
                f.field(x);
                f.finish()
            },
            Ne(x) => {
                let mut f = f.debug_tuple("Ne");
                f.field(x);
                f.finish()
            },
            Ge(x) => {
                let mut f = f.debug_tuple("Ge");
                f.field(x);
                f.finish()
            },
            Gt(x) => {
                let mut f = f.debug_tuple("Gt");
                f.field(x);
                f.finish()
            },
            AddAssign(x) => {
                let mut f = f.debug_tuple("AddAssign");
                f.field(x);
                f.finish()
            },
            SubAssign(x) => {
                let mut f = f.debug_tuple("SubAssign");
                f.field(x);
                f.finish()
            },
            MulAssign(x) => {
                let mut f = f.debug_tuple("MulAssign");
                f.field(x);
                f.finish()
            },
            DivAssign(x) => {
                let mut f = f.debug_tuple("DivAssign");
                f.field(x);
                f.finish()
            },
            RemAssign(x) => {
                let mut f = f.debug_tuple("RemAssign");
                f.field(x);
                f.finish()
            },
            BitXorAssign(x) => {
                let mut f = f.debug_tuple("BitXorAssign");
                f.field(x);
                f.finish()
            },
            BitAndAssign(x) => {
                let mut f = f.debug_tuple("BitAndAssign");
                f.field(x);
                f.finish()
            },
            BitOrAssign(x) => {
                let mut f = f.debug_tuple("BitOrAssign");
                f.field(x);
                f.finish()
            },
            ShlAssign(x) => {
                let mut f = f.debug_tuple("ShlAssign");
                f.field(x);
                f.finish()
            },
            ShrAssign(x) => {
                let mut f = f.debug_tuple("ShrAssign");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for stmt::Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Block");
        f.field("brace", &self.brace);
        f.field("stmts", &self.stmts);
        f.finish()
    }
}
impl Debug for gen::bound::Lifes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Bgen::bound::Lifes");
        f.field("for_", &self.for_);
        f.field("lt", &self.lt);
        f.field("lifetimes", &self.lifes);
        f.field("gt", &self.gt);
        f.finish()
    }
}
impl Debug for gen::param::Const {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Const");
        f.field("attrs", &self.attrs);
        f.field("const_", &self.const_);
        f.field("ident", &self.ident);
        f.field("colon", &self.colon);
        f.field("ty", &self.typ);
        f.field("eq", &self.eq);
        f.field("default", &self.default);
        f.finish()
    }
}
impl Debug for path::Constraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Constraint");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.finish()
    }
}
impl Debug for data::Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Data::")?;
        use data::Data::*;
        match self {
            Struct(x) => x.debug(f, "Struct"),
            Enum(x) => x.debug(f, "Enum"),
            Union(x) => x.debug(f, "Union"),
        }
    }
}
impl Debug for data::Enum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Enum {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("enum_", &self.enum_);
                f.field("brace", &self.brace);
                f.field("variants", &self.variants);
                f.finish()
            }
        }
        self.debug(f, "data::Enum")
    }
}
impl Debug for data::Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Struct {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("struct_", &self.struct_);
                f.field("fields", &self.fields);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "data::Struct")
    }
}
impl Debug for data::Union {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Union {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("union_", &self.union_);
                f.field("fields", &self.fields);
                f.finish()
            }
        }
        self.debug(f, "data::Union")
    }
}
impl Debug for Input {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Input");
        f.field("attrs", &self.attrs);
        f.field("vis", &self.vis);
        f.field("ident", &self.ident);
        f.field("gens", &self.gens);
        f.field("data", &self.data);
        f.finish()
    }
}
impl Debug for expr::Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Expr::")?;
        use expr::Expr::*;
        match self {
            Array(x) => x.debug(f, "Array"),
            Assign(x) => x.debug(f, "Assign"),
            Async(x) => x.debug(f, "Async"),
            Await(x) => x.debug(f, "Await"),
            Binary(x) => x.debug(f, "Binary"),
            Block(x) => x.debug(f, "Block"),
            Break(x) => x.debug(f, "Break"),
            Call(x) => x.debug(f, "Call"),
            Cast(x) => x.debug(f, "Cast"),
            Closure(x) => x.debug(f, "Closure"),
            Const(x) => x.debug(f, "Const"),
            Continue(x) => x.debug(f, "Continue"),
            Field(x) => x.debug(f, "Field"),
            ForLoop(x) => x.debug(f, "ForLoop"),
            Group(x) => x.debug(f, "Group"),
            If(x) => x.debug(f, "If"),
            Index(x) => x.debug(f, "Index"),
            Infer(x) => x.debug(f, "Infer"),
            Let(x) => x.debug(f, "Let"),
            Lit(x) => x.debug(f, "Lit"),
            Loop(x) => x.debug(f, "Loop"),
            Mac(x) => x.debug(f, "Macro"),
            Match(x) => x.debug(f, "Match"),
            MethodCall(x) => x.debug(f, "MethodCall"),
            Paren(x) => x.debug(f, "Paren"),
            Path(x) => x.debug(f, "Path"),
            Range(x) => x.debug(f, "Range"),
            Ref(x) => x.debug(f, "Reference"),
            Repeat(x) => x.debug(f, "Repeat"),
            Return(x) => x.debug(f, "Return"),
            Struct(x) => x.debug(f, "Struct"),
            Try(x) => x.debug(f, "Try"),
            TryBlock(x) => x.debug(f, "TryBlock"),
            Tuple(x) => x.debug(f, "Tuple"),
            Unary(x) => x.debug(f, "Unary"),
            Unsafe(x) => x.debug(f, "Unsafe"),
            Verbatim(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
            While(x) => x.debug(f, "While"),
            Yield(x) => x.debug(f, "Yield"),
        }
    }
}
impl Debug for expr::Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Array {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("bracket", &self.bracket);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "expr::Array")
    }
}
impl Debug for expr::Assign {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Assign {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("left", &self.left);
                f.field("eq", &self.eq);
                f.field("right", &self.right);
                f.finish()
            }
        }
        self.debug(f, "expr::Assign")
    }
}
impl Debug for expr::Async {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Async {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("async_", &self.async_);
                f.field("capture", &self.move_);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "expr::Async")
    }
}
impl Debug for expr::Await {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Await {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("base", &self.expr);
                f.field("dot", &self.dot);
                f.field("await_", &self.await_);
                f.finish()
            }
        }
        self.debug(f, "expr::Await")
    }
}
impl Debug for expr::Binary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Binary {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("left", &self.left);
                f.field("op", &self.op);
                f.field("right", &self.right);
                f.finish()
            }
        }
        self.debug(f, "expr::Binary")
    }
}
impl Debug for expr::Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Block {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("label", &self.label);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "expr::Block")
    }
}
impl Debug for expr::Break {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Break {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("break_", &self.break_);
                f.field("label", &self.life);
                f.field("expr", &self.val);
                f.finish()
            }
        }
        self.debug(f, "expr::Break")
    }
}
impl Debug for expr::Call {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Call {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("func", &self.func);
                f.field("paren", &self.paren);
                f.field("args", &self.args);
                f.finish()
            }
        }
        self.debug(f, "expr::Call")
    }
}
impl Debug for expr::Cast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Cast {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("expr", &self.expr);
                f.field("as_", &self.as_);
                f.field("ty", &self.typ);
                f.finish()
            }
        }
        self.debug(f, "expr::Cast")
    }
}
impl Debug for expr::Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Closure {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("lifetimes", &self.lifes);
                f.field("const_", &self.const_);
                f.field("movability", &self.static_);
                f.field("asyncness", &self.async_);
                f.field("capture", &self.move_);
                f.field("or1", &self.or1);
                f.field("inputs", &self.inputs);
                f.field("or2", &self.or2);
                f.field("output", &self.ret);
                f.field("body", &self.body);
                f.finish()
            }
        }
        self.debug(f, "expr::Closure")
    }
}
impl Debug for expr::Const {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Const {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("const_", &self.const_);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "expr::Const")
    }
}
impl Debug for expr::Continue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Continue {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("continue_", &self.continue_);
                f.field("label", &self.life);
                f.finish()
            }
        }
        self.debug(f, "expr::Continue")
    }
}
impl Debug for expr::Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Field {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("base", &self.expr);
                f.field("dot", &self.dot);
                f.field("member", &self.memb);
                f.finish()
            }
        }
        self.debug(f, "expr::Field")
    }
}
impl Debug for expr::ForLoop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::ForLoop {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("label", &self.label);
                f.field("for_", &self.for_);
                f.field("pat", &self.pat);
                f.field("in_", &self.in_);
                f.field("expr", &self.expr);
                f.field("body", &self.body);
                f.finish()
            }
        }
        self.debug(f, "expr::ForLoop")
    }
}
impl Debug for expr::Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Group {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("group", &self.group);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Group")
    }
}
impl Debug for expr::If {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::If {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("if_", &self.if_);
                f.field("cond", &self.cond);
                f.field("then_branch", &self.then_branch);
                f.field("else_branch", &self.else_branch);
                f.finish()
            }
        }
        self.debug(f, "expr::If")
    }
}
impl Debug for expr::Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Index {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("expr", &self.expr);
                f.field("bracket", &self.bracket);
                f.field("index", &self.index);
                f.finish()
            }
        }
        self.debug(f, "expr::Index")
    }
}
impl Debug for expr::Infer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Infer {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("underscore", &self.underscore);
                f.finish()
            }
        }
        self.debug(f, "expr::Infer")
    }
}
impl Debug for expr::Let {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Let {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("let_", &self.let_);
                f.field("pat", &self.pat);
                f.field("eq", &self.eq);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Let")
    }
}
impl Debug for expr::Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Lit {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("lit", &self.lit);
                f.finish()
            }
        }
        self.debug(f, "expr::Lit")
    }
}
impl Debug for expr::Loop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Loop {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("label", &self.label);
                f.field("loop_", &self.loop_);
                f.field("body", &self.body);
                f.finish()
            }
        }
        self.debug(f, "expr::Loop")
    }
}
impl Debug for expr::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.finish()
            }
        }
        self.debug(f, "expr::Mac")
    }
}
impl Debug for expr::Match {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Match {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("match_", &self.match_);
                f.field("expr", &self.expr);
                f.field("brace", &self.brace);
                f.field("arms", &self.arms);
                f.finish()
            }
        }
        self.debug(f, "expr::Match")
    }
}
impl Debug for expr::MethodCall {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::MethodCall {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("receiver", &self.expr);
                f.field("dot", &self.dot);
                f.field("method", &self.method);
                f.field("turbofish", &self.turbofish);
                f.field("paren", &self.paren);
                f.field("args", &self.args);
                f.finish()
            }
        }
        self.debug(f, "expr::MethodCall")
    }
}
impl Debug for expr::Paren {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Paren {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("paren", &self.paren);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Paren")
    }
}
impl Debug for expr::Pathth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Pathth {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.finish()
            }
        }
        self.debug(f, "expr::Pathth")
    }
}
impl Debug for expr::Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Range {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("start", &self.beg);
                f.field("limits", &self.limits);
                f.field("end", &self.end);
                f.finish()
            }
        }
        self.debug(f, "expr::Range")
    }
}
impl Debug for expr::Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Ref {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("and", &self.and);
                f.field("mut_", &self.mut_);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Ref")
    }
}
impl Debug for expr::Repeat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Repeat {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("bracket", &self.bracket);
                f.field("expr", &self.expr);
                f.field("semi", &self.semi);
                f.field("len", &self.len);
                f.finish()
            }
        }
        self.debug(f, "expr::Repeat")
    }
}
impl Debug for expr::Return {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Return {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("return_", &self.return_);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Return")
    }
}
impl Debug for expr::Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Struct {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.field("brace", &self.brace);
                f.field("fields", &self.fields);
                f.field("dot2", &self.dot2);
                f.field("rest", &self.rest);
                f.finish()
            }
        }
        self.debug(f, "expr::ExprStruct")
    }
}
impl Debug for expr::Try {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Try {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("expr", &self.expr);
                f.field("question", &self.question);
                f.finish()
            }
        }
        self.debug(f, "expr::Try")
    }
}
impl Debug for expr::TryBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::TryBlock {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("try_", &self.try_);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "expr::TryBlock")
    }
}
impl Debug for expr::Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Tuple {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("paren", &self.paren);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "expr::Tuple")
    }
}
impl Debug for expr::Unary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Unary {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("op", &self.op);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Unary")
    }
}
impl Debug for expr::Unsafe {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Unsafe {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("unsafe_", &self.unsafe_);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "expr::Unsafe")
    }
}
impl Debug for expr::While {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::While {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("label", &self.label);
                f.field("while_", &self.while_);
                f.field("cond", &self.cond);
                f.field("body", &self.body);
                f.finish()
            }
        }
        self.debug(f, "expr::While")
    }
}
impl Debug for expr::Yield {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl expr::Yield {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("yield_", &self.yield_);
                f.field("expr", &self.expr);
                f.finish()
            }
        }
        self.debug(f, "expr::Yield")
    }
}
impl Debug for data::Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("data::Field");
        f.field("attrs", &self.attrs);
        f.field("vis", &self.vis);
        f.field("mut_", &self.mut_);
        f.field("ident", &self.ident);
        f.field("colon", &self.colon);
        f.field("ty", &self.typ);
        f.finish()
    }
}
impl Debug for data::Mut {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Mut::")?;
        match self {
            data::Mut::None => f.write_str("None"),
        }
    }
}
impl Debug for pat::Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("pat::Field");
        f.field("attrs", &self.attrs);
        f.field("member", &self.memb);
        f.field("colon", &self.colon);
        f.field("pat", &self.pat);
        f.finish()
    }
}
impl Debug for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("FieldValue");
        f.field("attrs", &self.attrs);
        f.field("member", &self.member);
        f.field("colon", &self.colon);
        f.field("expr", &self.expr);
        f.finish()
    }
}
impl Debug for data::Fields {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("data::Fields::")?;
        match self {
            data::Fields::Named(x) => x.debug(f, "Named"),
            data::Fields::Unnamed(x) => x.debug(f, "Unnamed"),
            data::Fields::Unit => f.write_str("Unit"),
        }
    }
}
impl Debug for data::Named {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Named {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("brace", &self.brace);
                f.field("fields", &self.fields);
                f.finish()
            }
        }
        self.debug(f, "data::Named")
    }
}
impl Debug for data::Unnamed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Unnamed {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("paren", &self.paren);
                f.field("fields", &self.fields);
                f.finish()
            }
        }
        self.debug(f, "data::Unnamed")
    }
}
impl Debug for item::File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::File");
        f.field("shebang", &self.shebang);
        f.field("attrs", &self.attrs);
        f.field("items", &self.items);
        f.finish()
    }
}
impl Debug for item::FnArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("item::FnArg::")?;
        match self {
            item::FnArg::Receiver(x) => {
                let mut f = f.debug_tuple("item::Receiver");
                f.field(x);
                f.finish()
            },
            item::FnArg::Type(x) => {
                let mut f = f.debug_tuple("Typed");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::foreign::Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("item::foreign::Item::")?;
        match self {
            item::foreign::Item::Fn(x) => x.debug(f, "Fn"),
            item::foreign::Item::Static(x) => x.debug(f, "Static"),
            item::foreign::Item::Type(x) => x.debug(f, "Type"),
            item::foreign::Item::Macro(x) => x.debug(f, "Macro"),
            item::foreign::Item::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::foreign::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::foreign::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("sig", &self.sig);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::foreign::Fn")
    }
}
impl Debug for item::foreign::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::foreign::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::foreign::Mac")
    }
}
impl Debug for item::foreign::Static {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::foreign::Static {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("static_", &self.static_);
                f.field("mut_", &self.mut_);
                f.field("ident", &self.ident);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::foreign::Static")
    }
}
impl Debug for item::foreign::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::foreign::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("type", &self.type_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::foreign::Type")
    }
}
impl Debug for Arg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("path::Arg::")?;
        match self {
            Arg::Life(x) => {
                let mut f = f.debug_tuple("Life");
                f.field(x);
                f.finish()
            },
            Arg::Type(x) => {
                let mut f = f.debug_tuple("Type");
                f.field(x);
                f.finish()
            },
            Arg::Const(x) => {
                let mut f = f.debug_tuple("Const");
                f.field(x);
                f.finish()
            },
            Arg::AssocType(x) => {
                let mut f = f.debug_tuple("AssocType");
                f.field(x);
                f.finish()
            },
            Arg::AssocConst(x) => {
                let mut f = f.debug_tuple("AssocConst");
                f.field(x);
                f.finish()
            },
            Arg::Constraint(x) => {
                let mut f = f.debug_tuple("Constraint");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for gen::Param {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("gen::Param::")?;
        match self {
            gen::Param::Life(x) => {
                let mut f = f.debug_tuple("Life");
                f.field(x);
                f.finish()
            },
            gen::Param::Type(x) => {
                let mut f = f.debug_tuple("Type");
                f.field(x);
                f.finish()
            },
            gen::Param::Const(x) => {
                let mut f = f.debug_tuple("Const");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for gen::Gens {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Gens");
        f.field("lt", &self.lt);
        f.field("params", &self.params);
        f.field("gt", &self.gt);
        f.field("where_clause", &self.where_);
        f.finish()
    }
}
impl Debug for item::impl_::Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("item::impl_::Item::")?;
        match self {
            item::impl_::Item::Const(x) => x.debug(f, "Const"),
            item::impl_::Item::Fn(x) => x.debug(f, "Fn"),
            item::impl_::Item::Type(x) => x.debug(f, "Type"),
            item::impl_::Item::Macro(x) => x.debug(f, "Macro"),
            item::impl_::Item::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::impl_::Const {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::impl_::Const {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("defaultness", &self.default_);
                f.field("const_", &self.const_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.field("eq", &self.eq);
                f.field("expr", &self.expr);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::impl_::Const")
    }
}
impl Debug for item::impl_::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::impl_::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("defaultness", &self.default_);
                f.field("sig", &self.sig);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "item::impl_::Fn")
    }
}
impl Debug for item::impl_::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::impl_::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::impl_::Mac")
    }
}
impl Debug for item::impl_::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::impl_::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("defaultness", &self.default_);
                f.field("type", &self.type_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("eq", &self.eq);
                f.field("ty", &self.typ);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::impl_::Type")
    }
}
impl Debug for item::impl_::Restriction {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        match *self {}
    }
}
impl Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Index");
        f.field("index", &self.index);
        f.field("span", &self.span);
        f.finish()
    }
}
impl Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Item::")?;
        match self {
            Item::Const(x) => x.debug(f, "Const"),
            Item::Enum(x) => x.debug(f, "Enum"),
            Item::Extern(x) => x.debug(f, "Extern"),
            Item::Fn(x) => x.debug(f, "Fn"),
            Item::Foreign(x) => x.debug(f, "Foreign"),
            Item::Impl(x) => x.debug(f, "Impl"),
            Item::Macro(x) => x.debug(f, "Macro"),
            Item::Mod(x) => x.debug(f, "Mod"),
            Item::Static(x) => x.debug(f, "Static"),
            Item::Struct(x) => x.debug(f, "Struct"),
            Item::Trait(x) => x.debug(f, "Trait"),
            Item::TraitAlias(x) => x.debug(f, "TraitAlias"),
            Item::Type(x) => x.debug(f, "Type"),
            Item::Union(x) => x.debug(f, "Union"),
            Item::Use(x) => x.debug(f, "Use"),
            Item::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::Const {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Const {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("const_", &self.const_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.field("eq", &self.eq);
                f.field("expr", &self.expr);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Const")
    }
}
impl Debug for item::Enum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Enum {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("enum_", &self.enum_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("brace", &self.brace);
                f.field("variants", &self.variants);
                f.finish()
            }
        }
        self.debug(f, "item::Enum")
    }
}
impl Debug for item::Extern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Extern {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("extern_", &self.extern_);
                f.field("crate_", &self.crate_);
                f.field("ident", &self.ident);
                f.field("rename", &self.rename);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Extern")
    }
}
impl Debug for item::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("sig", &self.sig);
                f.field("block", &self.block);
                f.finish()
            }
        }
        self.debug(f, "item::Fn")
    }
}
impl Debug for item::Foreign {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Foreign {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("unsafe_", &self.unsafe_);
                f.field("abi", &self.abi);
                f.field("brace", &self.brace);
                f.field("items", &self.items);
                f.finish()
            }
        }
        self.debug(f, "item::Foreign")
    }
}
impl Debug for item::Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Impl {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("defaultness", &self.default_);
                f.field("unsafe_", &self.unsafe_);
                f.field("impl_", &self.impl_);
                f.field("gens", &self.gens);
                f.field("trait_", &self.trait_);
                f.field("self_ty", &self.typ);
                f.field("brace", &self.brace);
                f.field("items", &self.items);
                f.finish()
            }
        }
        self.debug(f, "item::Impl")
    }
}
impl Debug for item::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("ident", &self.ident);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Mac")
    }
}
impl Debug for item::Mod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Mod {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("unsafe_", &self.unsafe_);
                f.field("mod_", &self.mod_);
                f.field("ident", &self.ident);
                f.field("content", &self.items);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Mod")
    }
}
impl Debug for item::Static {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Static {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("static_", &self.static_);
                f.field("mut_", &self.mut_);
                f.field("ident", &self.ident);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.field("eq", &self.eq);
                f.field("expr", &self.expr);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Static")
    }
}
impl Debug for item::Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Struct {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("struct_", &self.struct_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("fields", &self.fields);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Struct")
    }
}
impl Debug for item::Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Trait {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("unsafe_", &self.unsafe_);
                f.field("auto_", &self.auto_);
                f.field("restriction", &self.restriction);
                f.field("trait_", &self.trait_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("colon", &self.colon);
                f.field("supertraits", &self.supers);
                f.field("brace", &self.brace);
                f.field("items", &self.items);
                f.finish()
            }
        }
        self.debug(f, "item::Trait")
    }
}
impl Debug for item::TraitAlias {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::TraitAlias {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("trait_", &self.trait_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("eq", &self.eq);
                f.field("bounds", &self.bounds);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::TraitAlias")
    }
}
impl Debug for item::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("type", &self.type_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("eq", &self.eq);
                f.field("ty", &self.typ);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Type")
    }
}
impl Debug for item::Union {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Union {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("union_", &self.union_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("fields", &self.fields);
                f.finish()
            }
        }
        self.debug(f, "item::Union")
    }
}
impl Debug for item::Use {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::Use {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("vis", &self.vis);
                f.field("use_", &self.use_);
                f.field("leading_colon", &self.colon);
                f.field("tree", &self.tree);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::Use")
    }
}
impl Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Label");
        f.field("name", &self.name);
        f.field("colon", &self.colon);
        f.finish()
    }
}
impl Debug for Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Life {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("apos", &self.apos);
                f.field("ident", &self.ident);
                f.finish()
            }
        }
        self.debug(f, "Life")
    }
}
impl Debug for gen::param::Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Life");
        f.field("attrs", &self.attrs);
        f.field("life", &self.life);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.finish()
    }
}
impl Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Lit::")?;
        match self {
            Lit::Str(x) => x.debug(f, "Str"),
            Lit::ByteStr(x) => x.debug(f, "ByteStr"),
            Lit::Byte(x) => x.debug(f, "Byte"),
            Lit::Char(x) => x.debug(f, "Char"),
            Lit::Int(x) => x.debug(f, "Int"),
            Lit::Float(x) => x.debug(f, "Float"),
            Lit::Bool(x) => x.debug(f, "Bool"),
            Lit::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for stmt::Local {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl stmt::Local {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("let_", &self.let_);
                f.field("pat", &self.pat);
                f.field("init", &self.init);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "stmt::Local")
    }
}
impl Debug for stmt::Init {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("stmt::Init");
        f.field("eq", &self.eq);
        f.field("expr", &self.expr);
        f.field("diverge", &self.diverge);
        f.finish()
    }
}
impl Debug for Macro {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Macro");
        f.field("path", &self.path);
        f.field("bang", &self.bang);
        f.field("delimiter", &self.delim);
        f.field("tokens", &self.toks);
        f.finish()
    }
}
impl Debug for tok::Delim {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("MacroDelimiter::")?;
        match self {
            tok::Delim::Paren(x) => {
                let mut f = f.debug_tuple("Paren");
                f.field(x);
                f.finish()
            },
            tok::Delim::Brace(x) => {
                let mut f = f.debug_tuple("Brace");
                f.field(x);
                f.finish()
            },
            tok::Delim::Bracket(x) => {
                let mut f = f.debug_tuple("Bracket");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for Member {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Member::")?;
        match self {
            Member::Named(x) => {
                let mut f = f.debug_tuple("Named");
                f.field(x);
                f.finish()
            },
            Member::Unnamed(x) => {
                let mut f = f.debug_tuple("Unnamed");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for attr::Meta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("attr::Meta::")?;
        match self {
            attr::Meta::Path(x) => x.debug(f, "Path"),
            attr::Meta::List(x) => x.debug(f, "List"),
            attr::Meta::NameValue(x) => x.debug(f, "NameValue"),
        }
    }
}
impl Debug for attr::List {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl attr::List {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("path", &self.path);
                f.field("delimiter", &self.delim);
                f.field("tokens", &self.toks);
                f.finish()
            }
        }
        self.debug(f, "attr::List")
    }
}
impl Debug for attr::NameValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl attr::NameValue {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("path", &self.name);
                f.field("eq", &self.eq);
                f.field("value", &self.val);
                f.finish()
            }
        }
        self.debug(f, "attr::NameValue")
    }
}
impl Debug for ParenthesizedArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl ParenthesizedArgs {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("paren", &self.paren);
                f.field("inputs", &self.ins);
                f.field("output", &self.out);
                f.finish()
            }
        }
        self.debug(f, "path::ParenthesizedArgs")
    }
}
impl Debug for pat::Pat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("pat::Pat::")?;
        match self {
            pat::Pat::Const(x) => x.debug(f, "Const"),
            pat::Pat::Ident(x) => x.debug(f, "Ident"),
            pat::Pat::Lit(x) => x.debug(f, "Lit"),
            pat::Pat::Mac(x) => x.debug(f, "Macro"),
            pat::Pat::Or(x) => x.debug(f, "Or"),
            pat::Pat::Paren(x) => x.debug(f, "Paren"),
            pat::Pat::Path(x) => x.debug(f, "Path"),
            pat::Pat::Range(x) => x.debug(f, "Range"),
            pat::Pat::Ref(x) => x.debug(f, "Reference"),
            pat::Pat::Rest(x) => x.debug(f, "Rest"),
            pat::Pat::Slice(x) => x.debug(f, "Slice"),
            pat::Pat::Struct(x) => x.debug(f, "Struct"),
            pat::Pat::Tuple(x) => x.debug(f, "Tuple"),
            pat::Pat::TupleStruct(x) => x.debug(f, "TupleStruct"),
            pat::Pat::Type(x) => x.debug(f, "Type"),
            pat::Pat::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
            pat::Pat::Wild(x) => x.debug(f, "Wild"),
        }
    }
}
impl Debug for pat::Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Ident {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("by_ref", &self.ref_);
                f.field("mut_", &self.mut_);
                f.field("ident", &self.ident);
                f.field("subpat", &self.sub);
                f.finish()
            }
        }
        self.debug(f, "pat::Ident")
    }
}
impl Debug for pat::Or {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Or {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("leading_vert", &self.vert);
                f.field("cases", &self.cases);
                f.finish()
            }
        }
        self.debug(f, "pat::Or")
    }
}
impl Debug for pat::Paren {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Paren {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("paren", &self.paren);
                f.field("pat", &self.pat);
                f.finish()
            }
        }
        self.debug(f, "pat::Paren")
    }
}
impl Debug for pat::Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Ref {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
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
impl Debug for pat::Rest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Rest {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("dot2", &self.dot2);
                f.finish()
            }
        }
        self.debug(f, "pat::Rest")
    }
}
impl Debug for pat::Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Slice {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("bracket", &self.bracket);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "pat::Slice")
    }
}
impl Debug for pat::Struct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Struct {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
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
impl Debug for pat::Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Tuple {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("paren", &self.paren);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "pat::Tuple")
    }
}
impl Debug for pat::TupleStructuct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::TupleStructuct {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.field("paren", &self.paren);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "pat::TupleStructuct")
    }
}
impl Debug for pat::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("pat", &self.pat);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.finish()
            }
        }
        self.debug(f, "pat::Type")
    }
}
impl Debug for pat::Wild {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl pat::Wild {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("underscore", &self.underscore);
                f.finish()
            }
        }
        self.debug(f, "pat::Wild")
    }
}
impl Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("leading_colon", &self.colon);
                f.field("segments", &self.segs);
                f.finish()
            }
        }
        self.debug(f, "Path")
    }
}
impl Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("path::Args::")?;
        match self {
            Args::None => f.write_str("None"),
            Args::Angled(x) => x.debug(f, "AngleBracketed"),
            Args::Parenthesized(x) => x.debug(f, "Parenthesized"),
        }
    }
}
impl Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::Segment");
        f.field("ident", &self.ident);
        f.field("arguments", &self.args);
        f.finish()
    }
}
impl Debug for gen::Where::Life {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("PredicateLifetime");
        f.field("life", &self.life);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.finish()
    }
}
impl Debug for gen::Where::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("PredicateType");
        f.field("lifetimes", &self.lifes);
        f.field("bounded_ty", &self.bounded);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.finish()
    }
}
impl Debug for QSelf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("QSelf");
        f.field("lt", &self.lt);
        f.field("ty", &self.ty);
        f.field("position", &self.pos);
        f.field("as_", &self.as_);
        f.field("gt", &self.gt_);
        f.finish()
    }
}
impl Debug for expr::Limits {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("expr::Limits::")?;
        match self {
            expr::Limits::HalfOpen(x) => {
                let mut f = f.debug_tuple("HalfOpen");
                f.field(x);
                f.finish()
            },
            expr::Limits::Closed(x) => {
                let mut f = f.debug_tuple("Closed");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::Receiver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Receiver");
        f.field("attrs", &self.attrs);
        f.field("reference", &self.ref_);
        f.field("mut_", &self.mut_);
        f.field("self_", &self.self_);
        f.field("colon", &self.colon);
        f.field("ty", &self.typ);
        f.finish()
    }
}
impl Debug for typ::Ret {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("typ::Ret::")?;
        match self {
            typ::Ret::Default => f.write_str("Default"),
            typ::Ret::Type(x, v1) => {
                let mut f = f.debug_tuple("Type");
                f.field(x);
                f.field(v1);
                f.finish()
            },
        }
    }
}
impl Debug for item::Sig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Sig");
        f.field("const_", &self.const_);
        f.field("asyncness", &self.async_);
        f.field("unsafe_", &self.unsafe_);
        f.field("abi", &self.abi);
        f.field("fn_", &self.fn_);
        f.field("ident", &self.ident);
        f.field("gens", &self.gens);
        f.field("paren", &self.paren);
        f.field("inputs", &self.args);
        f.field("vari", &self.vari);
        f.field("output", &self.ret);
        f.finish()
    }
}
impl Debug for StaticMut {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("StaticMutability::")?;
        match self {
            StaticMut::Mut(x) => {
                let mut f = f.debug_tuple("Mut");
                f.field(x);
                f.finish()
            },
            StaticMut::None => f.write_str("None"),
        }
    }
}
impl Debug for stmt::Stmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("stmt::Stmt::")?;
        match self {
            stmt::Stmt::stmt::Local(x) => x.debug(f, "stmt::Local"),
            stmt::Stmt::Item(x) => {
                let mut f = f.debug_tuple("Item");
                f.field(x);
                f.finish()
            },
            stmt::Stmt::Expr(x, v1) => {
                let mut f = f.debug_tuple("Expr");
                f.field(x);
                f.field(v1);
                f.finish()
            },
            stmt::Stmt::Mac(x) => x.debug(f, "Macro"),
        }
    }
}
impl Debug for stmt::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl stmt::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "stmt::Mac")
    }
}
impl Debug for gen::bound::Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::bound::Trait");
        f.field("paren", &self.paren);
        f.field("modifier", &self.modif);
        f.field("lifetimes", &self.lifes);
        f.field("path", &self.path);
        f.finish()
    }
}
impl Debug for gen::bound::Modifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("gen::bound::Modifier::")?;
        match self {
            gen::bound::Modifier::None => f.write_str("None"),
            gen::bound::Modifier::Maybe(x) => {
                let mut f = f.debug_tuple("Maybe");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::trait_::Item {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("item::trait_::Item::")?;
        match self {
            item::trait_::Item::Const(x) => x.debug(f, "Const"),
            item::trait_::Item::Fn(x) => x.debug(f, "Fn"),
            item::trait_::Item::Type(x) => x.debug(f, "Type"),
            item::trait_::Item::Macro(x) => x.debug(f, "Macro"),
            item::trait_::Item::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::trait_::Const {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::trait_::Const {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("const_", &self.const_);
                f.field("ident", &self.ident);
                f.field("gens", &self.gens);
                f.field("colon", &self.colon);
                f.field("ty", &self.typ);
                f.field("default", &self.default);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::trait_::Const")
    }
}
impl Debug for item::trait_::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::trait_::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("sig", &self.sig);
                f.field("default", &self.default);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::trait_::Fn")
    }
}
impl Debug for item::trait_::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::trait_::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("attrs", &self.attrs);
                f.field("mac", &self.mac);
                f.field("semi", &self.semi);
                f.finish()
            }
        }
        self.debug(f, "item::trait_::Mac")
    }
}
impl Debug for item::trait_::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl item::trait_::Type {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("attrs", &self.attrs);
                y.field("type", &self.type_);
                y.field("ident", &self.ident);
                y.field("gens", &self.gens);
                y.field("colon", &self.colon);
                y.field("bounds", &self.bounds);
                y.field("default", &self.default);
                y.field("semi", &self.semi);
                y.finish()
            }
        }
        self.debug(f, "item::trait_::Type")
    }
}
impl Debug for typ::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Type::")?;
        match self {
            typ::Type::Array(x) => x.debug(f, "Array"),
            typ::Type::Fn(x) => x.debug(f, "Fn"),
            typ::Type::Group(x) => x.debug(f, "Group"),
            typ::Type::Impl(x) => x.debug(f, "ImplTrait"),
            typ::Type::Infer(x) => x.debug(f, "Infer"),
            typ::Type::Mac(x) => x.debug(f, "Macro"),
            typ::Type::Never(x) => x.debug(f, "Never"),
            typ::Type::Paren(x) => x.debug(f, "Paren"),
            typ::Type::Path(x) => x.debug(f, "Path"),
            typ::Type::Ptr(x) => x.debug(f, "Ptr"),
            typ::Type::Ref(x) => x.debug(f, "Reference"),
            typ::Type::Slice(x) => x.debug(f, "Slice"),
            typ::Type::Trait(x) => x.debug(f, "TraitObject"),
            typ::Type::Tuple(x) => x.debug(f, "Tuple"),
            typ::Type::Stream(x) => {
                let mut y = f.debug_tuple("Stream");
                y.field(x);
                y.finish()
            },
        }
    }
}
impl Debug for typ::Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Array {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("bracket", &self.bracket);
                y.field("elem", &self.elem);
                y.field("semi", &self.semi);
                y.field("len", &self.len);
                y.finish()
            }
        }
        self.debug(f, "typ::Array")
    }
}
impl Debug for typ::Fn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Fn {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("lifetimes", &self.lifes);
                y.field("unsafe_", &self.unsafe_);
                y.field("abi", &self.abi);
                y.field("fn_", &self.fn_);
                y.field("paren", &self.paren);
                y.field("inputs", &self.args);
                y.field("vari", &self.vari);
                y.field("output", &self.ret);
                y.finish()
            }
        }
        self.debug(f, "typ::Fn")
    }
}
impl Debug for typ::Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Group {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("group", &self.group);
                y.field("elem", &self.elem);
                y.finish()
            }
        }
        self.debug(f, "typ::Group")
    }
}
impl Debug for typ::Impl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Impl {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut y = f.debug_struct(name);
                y.field("impl_", &self.impl_);
                y.field("bounds", &self.bounds);
                y.finish()
            }
        }
        self.debug(f, "typ::Impl")
    }
}
impl Debug for typ::Infer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Infer {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("underscore", &self.underscore);
                f.finish()
            }
        }
        self.debug(f, "typ::Infer")
    }
}
impl Debug for typ::Mac {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Mac {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("mac", &self.mac);
                f.finish()
            }
        }
        self.debug(f, "typ::Mac")
    }
}
impl Debug for typ::Never {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Never {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("bang", &self.bang);
                f.finish()
            }
        }
        self.debug(f, "typ::Never")
    }
}
impl Debug for gen::param::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::param::Type");
        f.field("attrs", &self.attrs);
        f.field("ident", &self.ident);
        f.field("colon", &self.colon);
        f.field("bounds", &self.bounds);
        f.field("eq", &self.eq);
        f.field("default", &self.default);
        f.finish()
    }
}
impl Debug for gen::bound::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("gen::bound::Type::")?;
        match self {
            gen::bound::Type::Trait(x) => {
                let mut f = f.debug_tuple("Trait");
                f.field(x);
                f.finish()
            },
            gen::bound::Type::Life(x) => x.debug(f, "Life"),
            gen::bound::Type::Stream(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for typ::Paren {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Paren {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("paren", &self.paren);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Paren")
    }
}
impl Debug for typ::Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Path {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("qself", &self.qself);
                f.field("path", &self.path);
                f.finish()
            }
        }
        self.debug(f, "typ::Path")
    }
}
impl Debug for typ::Ptr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Ptr {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("star", &self.star);
                f.field("const_", &self.const_);
                f.field("mut_", &self.mut_);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Ptr")
    }
}
impl Debug for typ::Ref {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Ref {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("and", &self.and);
                f.field("life", &self.life);
                f.field("mut_", &self.mut_);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Ref")
    }
}
impl Debug for typ::Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Slice {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("bracket", &self.bracket);
                f.field("elem", &self.elem);
                f.finish()
            }
        }
        self.debug(f, "typ::Slice")
    }
}
impl Debug for typ::Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Trait {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("dyn_", &self.dyn_);
                f.field("bounds", &self.bounds);
                f.finish()
            }
        }
        self.debug(f, "typ::Trait")
    }
}
impl Debug for typ::Tuple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl typ::Tuple {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("paren", &self.paren);
                f.field("elems", &self.elems);
                f.finish()
            }
        }
        self.debug(f, "typ::Tuple")
    }
}
impl Debug for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("UnOp::")?;
        match self {
            UnOp::Deref(x) => {
                let mut f = f.debug_tuple("Deref");
                f.field(x);
                f.finish()
            },
            UnOp::Not(x) => {
                let mut f = f.debug_tuple("Not");
                f.field(x);
                f.finish()
            },
            UnOp::Neg(x) => {
                let mut f = f.debug_tuple("Neg");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::use_::Glob {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::use_::Glob");
        f.field("star", &self.star);
        f.finish()
    }
}
impl Debug for item::use_::Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::use_::Group");
        f.field("brace", &self.brace);
        f.field("items", &self.elems);
        f.finish()
    }
}
impl Debug for item::use_::Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::use_::Name");
        f.field("ident", &self.ident);
        f.finish()
    }
}
impl Debug for item::use_::Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::use_::Path");
        f.field("ident", &self.ident);
        f.field("colon2", &self.colon2);
        f.field("tree", &self.tree);
        f.finish()
    }
}
impl Debug for item::use_::Rename {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::use_::Rename");
        f.field("ident", &self.ident);
        f.field("as_", &self.as_);
        f.field("rename", &self.rename);
        f.finish()
    }
}
impl Debug for item::use_::Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("item::use_::Tree::")?;
        match self {
            item::use_::Tree::Path(x) => {
                let mut f = f.debug_tuple("Path");
                f.field(x);
                f.finish()
            },
            item::use_::Tree::Name(x) => {
                let mut f = f.debug_tuple("Name");
                f.field(x);
                f.finish()
            },
            item::use_::Tree::Rename(x) => {
                let mut f = f.debug_tuple("Rename");
                f.field(x);
                f.finish()
            },
            item::use_::Tree::Glob(x) => {
                let mut f = f.debug_tuple("Glob");
                f.field(x);
                f.finish()
            },
            item::use_::Tree::Group(x) => {
                let mut f = f.debug_tuple("Group");
                f.field(x);
                f.finish()
            },
        }
    }
}
impl Debug for item::Variadic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("item::Variadic");
        f.field("attrs", &self.attrs);
        f.field("pat", &self.pat);
        f.field("dots", &self.dots);
        f.field("comma", &self.comma);
        f.finish()
    }
}
impl Debug for data::Variant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("data::Variant");
        f.field("attrs", &self.attrs);
        f.field("ident", &self.ident);
        f.field("fields", &self.fields);
        f.field("discriminant", &self.discrim);
        f.finish()
    }
}
impl Debug for data::Restricted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl data::Restricted {
            fn debug(&self, f: &mut fmt::Formatter, name: &str) -> fmt::Result {
                let mut f = f.debug_struct(name);
                f.field("pub_", &self.pub_);
                f.field("paren", &self.paren);
                f.field("in_", &self.in_);
                f.field("path", &self.path);
                f.finish()
            }
        }
        self.debug(f, "data::Restricted")
    }
}
impl Debug for data::Visibility {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("data::Visibility::")?;
        match self {
            data::Visibility::Public(x) => {
                let mut f = f.debug_tuple("Public");
                f.field(x);
                f.finish()
            },
            data::Visibility::Restricted(x) => x.debug(f, "Restricted"),
            data::Visibility::Inherited => f.write_str("Inherited"),
        }
    }
}
impl Debug for gen::Where {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::Where");
        f.field("where_", &self.where_);
        f.field("predicates", &self.preds);
        f.finish()
    }
}
impl Debug for gen::Where::Pred {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("WherePredicate::")?;
        match self {
            gen::Where::Pred::Life(x) => {
                let mut f = f.debug_tuple("Life");
                f.field(x);
                f.finish()
            },
            gen::Where::Pred::Type(x) => {
                let mut f = f.debug_tuple("Type");
                f.field(x);
                f.finish()
            },
        }
    }
}
