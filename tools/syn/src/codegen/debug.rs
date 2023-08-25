use crate::*;

impl Debug for path::Angle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl path::Angle {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("colon2", &self.colon2);
                f.field("lt", &self.lt);
                f.field("args", &self.args);
                f.field("gt", &self.gt);
                f.finish()
            }
        }
        self.debug(f, "path::Angle")
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
impl Debug for gen::bound::Lifes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Bgen::bound::Lifes");
        f.field("for_", &self.for_);
        f.field("lt", &self.lt);
        f.field("lifes", &self.lifes);
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
impl Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("Index");
        f.field("index", &self.index);
        f.field("span", &self.span);
        f.finish()
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
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
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
impl Debug for path::Parenth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl path::Parenth {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("parenth", &self.parenth);
                f.field("args", &self.args);
                f.field("ret", &self.ret);
                f.finish()
            }
        }
        self.debug(f, "path::Parenth")
    }
}
impl Debug for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        impl Path {
            fn debug(&self, f: &mut fmt::Formatter, x: &str) -> fmt::Result {
                let mut f = f.debug_struct(x);
                f.field("colon", &self.colon);
                f.field("segs", &self.segs);
                f.finish()
            }
        }
        self.debug(f, "Path")
    }
}
impl Debug for path::Args {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("path::Args::")?;
        use path::Args::*;
        match self {
            None => f.write_str("None"),
            Angle(x) => x.debug(f, "Angle"),
            Parenth(x) => x.debug(f, "Parenth"),
        }
    }
}
impl Debug for path::Segment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("path::Segment");
        f.field("ident", &self.ident);
        f.field("args", &self.args);
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
        f.field("lifes", &self.lifes);
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
impl Debug for gen::bound::Trait {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut f = f.debug_struct("gen::bound::Trait");
        f.field("parenth", &self.parenth);
        f.field("modif", &self.modif);
        f.field("lifes", &self.lifes);
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
            gen::bound::Type::Verbatim(x) => {
                let mut f = f.debug_tuple("Stream");
                f.field(x);
                f.finish()
            },
        }
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
