use crate::*;

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
