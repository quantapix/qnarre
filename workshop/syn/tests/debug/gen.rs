#![allow(clippy::match_wildcard_for_single_variants)]
use super::{Lite, Present};
use ref_cast::RefCast;
use std::fmt::{self, Debug, Display};
impl Debug for Lite<syn::Abi> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Abi");
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::LitStr);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::AngleBracketedGenericArguments> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AngleBracketedGenericArguments");
        if self.value.colon2_token.is_some() {
            formatter.field("colon2_token", &Present);
        }
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Arm> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Arm");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.guard {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::If, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("guard", Print::ref_cast(val));
        }
        formatter.field("body", Lite(&self.value.body));
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::AssocConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocConst");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.generics {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngleBracketedGenericArguments);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("generics", Print::ref_cast(val));
        }
        formatter.field("value", Lite(&self.value.value));
        formatter.finish()
    }
}
impl Debug for Lite<syn::AssocType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("AssocType");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.generics {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngleBracketedGenericArguments);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("generics", Print::ref_cast(val));
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::AttrStyle> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::AttrStyle::Outer => formatter.write_str("AttrStyle::Outer"),
            syn::AttrStyle::Inner(_val) => {
                formatter.write_str("AttrStyle::Inner")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Attribute> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Attribute");
        formatter.field("style", Lite(&self.value.style));
        formatter.field("meta", Lite(&self.value.meta));
        formatter.finish()
    }
}
impl Debug for Lite<syn::BareFnArg> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BareFnArg");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::BareVariadic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BareVariadic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.name {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((proc_macro2::Ident, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("name", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::BinOp> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::BinOp::Add(_val) => {
                formatter.write_str("BinOp::Add")?;
                Ok(())
            },
            syn::BinOp::Sub(_val) => {
                formatter.write_str("BinOp::Sub")?;
                Ok(())
            },
            syn::BinOp::Mul(_val) => {
                formatter.write_str("BinOp::Mul")?;
                Ok(())
            },
            syn::BinOp::Div(_val) => {
                formatter.write_str("BinOp::Div")?;
                Ok(())
            },
            syn::BinOp::Rem(_val) => {
                formatter.write_str("BinOp::Rem")?;
                Ok(())
            },
            syn::BinOp::And(_val) => {
                formatter.write_str("BinOp::And")?;
                Ok(())
            },
            syn::BinOp::Or(_val) => {
                formatter.write_str("BinOp::Or")?;
                Ok(())
            },
            syn::BinOp::BitXor(_val) => {
                formatter.write_str("BinOp::BitXor")?;
                Ok(())
            },
            syn::BinOp::BitAnd(_val) => {
                formatter.write_str("BinOp::BitAnd")?;
                Ok(())
            },
            syn::BinOp::BitOr(_val) => {
                formatter.write_str("BinOp::BitOr")?;
                Ok(())
            },
            syn::BinOp::Shl(_val) => {
                formatter.write_str("BinOp::Shl")?;
                Ok(())
            },
            syn::BinOp::Shr(_val) => {
                formatter.write_str("BinOp::Shr")?;
                Ok(())
            },
            syn::BinOp::Eq(_val) => {
                formatter.write_str("BinOp::Eq")?;
                Ok(())
            },
            syn::BinOp::Lt(_val) => {
                formatter.write_str("BinOp::Lt")?;
                Ok(())
            },
            syn::BinOp::Le(_val) => {
                formatter.write_str("BinOp::Le")?;
                Ok(())
            },
            syn::BinOp::Ne(_val) => {
                formatter.write_str("BinOp::Ne")?;
                Ok(())
            },
            syn::BinOp::Ge(_val) => {
                formatter.write_str("BinOp::Ge")?;
                Ok(())
            },
            syn::BinOp::Gt(_val) => {
                formatter.write_str("BinOp::Gt")?;
                Ok(())
            },
            syn::BinOp::AddAssign(_val) => {
                formatter.write_str("BinOp::AddAssign")?;
                Ok(())
            },
            syn::BinOp::SubAssign(_val) => {
                formatter.write_str("BinOp::SubAssign")?;
                Ok(())
            },
            syn::BinOp::MulAssign(_val) => {
                formatter.write_str("BinOp::MulAssign")?;
                Ok(())
            },
            syn::BinOp::DivAssign(_val) => {
                formatter.write_str("BinOp::DivAssign")?;
                Ok(())
            },
            syn::BinOp::RemAssign(_val) => {
                formatter.write_str("BinOp::RemAssign")?;
                Ok(())
            },
            syn::BinOp::BitXorAssign(_val) => {
                formatter.write_str("BinOp::BitXorAssign")?;
                Ok(())
            },
            syn::BinOp::BitAndAssign(_val) => {
                formatter.write_str("BinOp::BitAndAssign")?;
                Ok(())
            },
            syn::BinOp::BitOrAssign(_val) => {
                formatter.write_str("BinOp::BitOrAssign")?;
                Ok(())
            },
            syn::BinOp::ShlAssign(_val) => {
                formatter.write_str("BinOp::ShlAssign")?;
                Ok(())
            },
            syn::BinOp::ShrAssign(_val) => {
                formatter.write_str("BinOp::ShrAssign")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::Block> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Block");
        if !self.value.stmts.is_empty() {
            formatter.field("stmts", Lite(&self.value.stmts));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::BoundLifetimes> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("BoundLifetimes");
        if !self.value.lifetimes.is_empty() {
            formatter.field("lifetimes", Lite(&self.value.lifetimes));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ConstParam> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ConstParam");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.ty));
        if self.value.eq_token.is_some() {
            formatter.field("eq_token", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Expr);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Constraint> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Constraint");
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.generics {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngleBracketedGenericArguments);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("generics", Print::ref_cast(val));
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Data> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Data::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Data::Struct");
                formatter.field("fields", Lite(&_val.fields));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::Data::Enum(_val) => {
                let mut formatter = formatter.debug_struct("Data::Enum");
                if !_val.variants.is_empty() {
                    formatter.field("variants", Lite(&_val.variants));
                }
                formatter.finish()
            },
            syn::Data::Union(_val) => {
                let mut formatter = formatter.debug_struct("Data::Union");
                formatter.field("fields", Lite(&_val.fields));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::DataEnum> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataEnum");
        if !self.value.variants.is_empty() {
            formatter.field("variants", Lite(&self.value.variants));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::DataStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataStruct");
        formatter.field("fields", Lite(&self.value.fields));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::DataUnion> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DataUnion");
        formatter.field("fields", Lite(&self.value.fields));
        formatter.finish()
    }
}
impl Debug for Lite<syn::DeriveInput> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("DeriveInput");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("data", Lite(&self.value.data));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Expr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Expr::Array(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Array");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Expr::Assign(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Assign");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("left", Lite(&_val.left));
                formatter.field("right", Lite(&_val.right));
                formatter.finish()
            },
            syn::Expr::Async(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Async");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.capture.is_some() {
                    formatter.field("capture", &Present);
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Await(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Await");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("base", Lite(&_val.base));
                formatter.finish()
            },
            syn::Expr::Binary(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Binary");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("left", Lite(&_val.left));
                formatter.field("op", Lite(&_val.op));
                formatter.field("right", Lite(&_val.right));
                formatter.finish()
            },
            syn::Expr::Block(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Block");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Break(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Break");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Lifetime);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Call(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Call");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("func", Lite(&_val.func));
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::Expr::Cast(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Cast");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("ty", Lite(&_val.ty));
                formatter.finish()
            },
            syn::Expr::Closure(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Closure");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.lifetimes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::BoundLifetimes);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("lifetimes", Print::ref_cast(val));
                }
                if _val.constness.is_some() {
                    formatter.field("constness", &Present);
                }
                if _val.movability.is_some() {
                    formatter.field("movability", &Present);
                }
                if _val.asyncness.is_some() {
                    formatter.field("asyncness", &Present);
                }
                if _val.capture.is_some() {
                    formatter.field("capture", &Present);
                }
                if !_val.inputs.is_empty() {
                    formatter.field("inputs", Lite(&_val.inputs));
                }
                formatter.field("output", Lite(&_val.output));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Const(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Continue(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Continue");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Lifetime);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Field(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Field");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("base", Lite(&_val.base));
                formatter.field("member", Lite(&_val.member));
                formatter.finish()
            },
            syn::Expr::ForLoop(_val) => {
                let mut formatter = formatter.debug_struct("Expr::ForLoop");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Group(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Group");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::If(_val) => {
                let mut formatter = formatter.debug_struct("Expr::If");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("cond", Lite(&_val.cond));
                formatter.field("then_branch", Lite(&_val.then_branch));
                if let Some(val) = &_val.else_branch {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Else, Box<syn::Expr>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("else_branch", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Index(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Index");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("index", Lite(&_val.index));
                formatter.finish()
            },
            syn::Expr::Infer(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Infer");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            syn::Expr::Let(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Let");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Lit(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Lit");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("lit", Lite(&_val.lit));
                formatter.finish()
            },
            syn::Expr::Loop(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Loop");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                formatter.finish()
            },
            syn::Expr::Match(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Match");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                if !_val.arms.is_empty() {
                    formatter.field("arms", Lite(&_val.arms));
                }
                formatter.finish()
            },
            syn::Expr::MethodCall(_val) => {
                let mut formatter = formatter.debug_struct("Expr::MethodCall");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("receiver", Lite(&_val.receiver));
                formatter.field("method", Lite(&_val.method));
                if let Some(val) = &_val.turbofish {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::AngleBracketedGenericArguments);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("turbofish", Print::ref_cast(val));
                }
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::Expr::Paren(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Paren");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Path(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Path");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::Expr::Range(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Range");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.start {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("start", Print::ref_cast(val));
                }
                formatter.field("limits", Lite(&_val.limits));
                if let Some(val) = &_val.end {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("end", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Reference(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Reference");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.mutability.is_some() {
                    formatter.field("mutability", &Present);
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Repeat(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Repeat");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.field("len", Lite(&_val.len));
                formatter.finish()
            },
            syn::Expr::Return(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Return");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                if _val.dot2_token.is_some() {
                    formatter.field("dot2_token", &Present);
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rest", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Expr::Try(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Try");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::TryBlock(_val) => {
                let mut formatter = formatter.debug_struct("Expr::TryBlock");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Tuple");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Expr::Unary(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Unary");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("op", Lite(&_val.op));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Expr::Unsafe(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Unsafe");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Expr::Verbatim(_val) => {
                formatter.write_str("Expr::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            syn::Expr::While(_val) => {
                let mut formatter = formatter.debug_struct("Expr::While");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.label {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Label);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("label", Print::ref_cast(val));
                }
                formatter.field("cond", Lite(&_val.cond));
                formatter.field("body", Lite(&_val.body));
                formatter.finish()
            },
            syn::Expr::Yield(_val) => {
                let mut formatter = formatter.debug_struct("Expr::Yield");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.expr {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(Box<syn::Expr>);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("expr", Print::ref_cast(val));
                }
                formatter.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::ExprArray> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprArray");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprAssign> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprAssign");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("left", Lite(&self.value.left));
        formatter.field("right", Lite(&self.value.right));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprAsync> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprAsync");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.capture.is_some() {
            formatter.field("capture", &Present);
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprAwait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprAwait");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("base", Lite(&self.value.base));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprBinary> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprBinary");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("left", Lite(&self.value.left));
        formatter.field("op", Lite(&self.value.op));
        formatter.field("right", Lite(&self.value.right));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprBlock> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprBlock");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprBreak> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprBreak");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Lifetime);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprCall> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprCall");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("func", Lite(&self.value.func));
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprCast> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprCast");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprClosure> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprClosure");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.lifetimes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::BoundLifetimes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.constness.is_some() {
            formatter.field("constness", &Present);
        }
        if self.value.movability.is_some() {
            formatter.field("movability", &Present);
        }
        if self.value.asyncness.is_some() {
            formatter.field("asyncness", &Present);
        }
        if self.value.capture.is_some() {
            formatter.field("capture", &Present);
        }
        if !self.value.inputs.is_empty() {
            formatter.field("inputs", Lite(&self.value.inputs));
        }
        formatter.field("output", Lite(&self.value.output));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprConst");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprContinue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprContinue");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Lifetime);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprField> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprField");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("base", Lite(&self.value.base));
        formatter.field("member", Lite(&self.value.member));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprForLoop> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprForLoop");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprGroup> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprGroup");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprIf> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprIf");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("cond", Lite(&self.value.cond));
        formatter.field("then_branch", Lite(&self.value.then_branch));
        if let Some(val) = &self.value.else_branch {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("else_branch", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprIndex> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprIndex");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("index", Lite(&self.value.index));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprInfer> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprInfer");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprLet> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprLet");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprLit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprLit");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("lit", Lite(&self.value.lit));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprLoop> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprLoop");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprMatch> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprMatch");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        if !self.value.arms.is_empty() {
            formatter.field("arms", Lite(&self.value.arms));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprMethodCall> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprMethodCall");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("receiver", Lite(&self.value.receiver));
        formatter.field("method", Lite(&self.value.method));
        if let Some(val) = &self.value.turbofish {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::AngleBracketedGenericArguments);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("turbofish", Print::ref_cast(val));
        }
        if !self.value.args.is_empty() {
            formatter.field("args", Lite(&self.value.args));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprParen> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprParen");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprPath> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprPath");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprRange> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprRange");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.start {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("start", Print::ref_cast(val));
        }
        formatter.field("limits", Lite(&self.value.limits));
        if let Some(val) = &self.value.end {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("end", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprReference> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprReference");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprRepeat> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprRepeat");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.field("len", Lite(&self.value.len));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprReturn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprReturn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        if self.value.dot2_token.is_some() {
            formatter.field("dot2_token", &Present);
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rest", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprTry> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprTry");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprTryBlock> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprTryBlock");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprTuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprTuple");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprUnary> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprUnary");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("op", Lite(&self.value.op));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprUnsafe> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprUnsafe");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprWhile> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprWhile");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.label {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Label);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("label", Print::ref_cast(val));
        }
        formatter.field("cond", Lite(&self.value.cond));
        formatter.field("body", Lite(&self.value.body));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ExprYield> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ExprYield");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.expr {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(Box<syn::Expr>);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("expr", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Field> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Field");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mutability {
            syn::FieldMutability::None => {},
            _ => {
                formatter.field("mutability", Lite(&self.value.mutability));
            },
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("ident", Print::ref_cast(val));
        }
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldMutability> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::FieldMutability::None => formatter.write_str("FieldMutability::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::FieldPat> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldPat");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("member", Lite(&self.value.member));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldValue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldValue");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("member", Lite(&self.value.member));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Fields> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Fields::Named(_val) => {
                let mut formatter = formatter.debug_struct("Fields::Named");
                if !_val.named.is_empty() {
                    formatter.field("named", Lite(&_val.named));
                }
                formatter.finish()
            },
            syn::Fields::Unnamed(_val) => {
                let mut formatter = formatter.debug_struct("Fields::Unnamed");
                if !_val.unnamed.is_empty() {
                    formatter.field("unnamed", Lite(&_val.unnamed));
                }
                formatter.finish()
            },
            syn::Fields::Unit => formatter.write_str("Fields::Unit"),
        }
    }
}
impl Debug for Lite<syn::FieldsNamed> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldsNamed");
        if !self.value.named.is_empty() {
            formatter.field("named", Lite(&self.value.named));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::FieldsUnnamed> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("FieldsUnnamed");
        if !self.value.unnamed.is_empty() {
            formatter.field("unnamed", Lite(&self.value.unnamed));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::File> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("File");
        if let Some(val) = &self.value.shebang {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(String);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("shebang", Print::ref_cast(val));
        }
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::FnArg> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::FnArg::Receiver(_val) => {
                formatter.write_str("FnArg::Receiver")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::FnArg::Typed(_val) => {
                formatter.write_str("FnArg::Typed")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::ForeignItem> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::ForeignItem::Fn(_val) => {
                let mut formatter = formatter.debug_struct("ForeignItem::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("sig", Lite(&_val.sig));
                formatter.finish()
            },
            syn::ForeignItem::Static(_val) => {
                let mut formatter = formatter.debug_struct("ForeignItem::Static");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                match _val.mutability {
                    syn::StaticMutability::None => {},
                    _ => {
                        formatter.field("mutability", Lite(&_val.mutability));
                    },
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("ty", Lite(&_val.ty));
                formatter.finish()
            },
            syn::ForeignItem::Type(_val) => {
                let mut formatter = formatter.debug_struct("ForeignItem::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.finish()
            },
            syn::ForeignItem::Macro(_val) => {
                let mut formatter = formatter.debug_struct("ForeignItem::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::ForeignItem::Verbatim(_val) => {
                formatter.write_str("ForeignItem::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::ForeignItemFn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ForeignItemFn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("sig", Lite(&self.value.sig));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ForeignItemMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ForeignItemMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ForeignItemStatic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ForeignItemStatic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mutability {
            syn::StaticMutability::None => {},
            _ => {
                formatter.field("mutability", Lite(&self.value.mutability));
            },
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ForeignItemType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ForeignItemType");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.finish()
    }
}
impl Debug for Lite<syn::GenericArgument> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::GenericArgument::Lifetime(_val) => {
                formatter.write_str("GenericArgument::Lifetime")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericArgument::Type(_val) => {
                formatter.write_str("GenericArgument::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericArgument::Const(_val) => {
                formatter.write_str("GenericArgument::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericArgument::AssocType(_val) => {
                formatter.write_str("GenericArgument::AssocType")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericArgument::AssocConst(_val) => {
                formatter.write_str("GenericArgument::AssocConst")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericArgument::Constraint(_val) => {
                formatter.write_str("GenericArgument::Constraint")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::GenericParam> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::GenericParam::Lifetime(_val) => {
                formatter.write_str("GenericParam::Lifetime")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericParam::Type(_val) => {
                formatter.write_str("GenericParam::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::GenericParam::Const(_val) => {
                formatter.write_str("GenericParam::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Generics> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Generics");
        if self.value.lt_token.is_some() {
            formatter.field("lt_token", &Present);
        }
        if !self.value.params.is_empty() {
            formatter.field("params", Lite(&self.value.params));
        }
        if self.value.gt_token.is_some() {
            formatter.field("gt_token", &Present);
        }
        if let Some(val) = &self.value.where_clause {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::WhereClause);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("where_clause", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ImplItem> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::ImplItem::Const(_val) => {
                let mut formatter = formatter.debug_struct("ImplItem::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.defaultness.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("ty", Lite(&_val.ty));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::ImplItem::Fn(_val) => {
                let mut formatter = formatter.debug_struct("ImplItem::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.defaultness.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("sig", Lite(&_val.sig));
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::ImplItem::Type(_val) => {
                let mut formatter = formatter.debug_struct("ImplItem::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.defaultness.is_some() {
                    formatter.field("defaultness", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("ty", Lite(&_val.ty));
                formatter.finish()
            },
            syn::ImplItem::Macro(_val) => {
                let mut formatter = formatter.debug_struct("ImplItem::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::ImplItem::Verbatim(_val) => {
                formatter.write_str("ImplItem::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::ImplItemConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ImplItemConst");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.defaultness.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ImplItemFn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ImplItemFn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.defaultness.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("sig", Lite(&self.value.sig));
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ImplItemMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ImplItemMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ImplItemType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ImplItemType");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.defaultness.is_some() {
            formatter.field("defaultness", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ImplRestriction> {
    fn fmt(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
        unreachable!()
    }
}
impl Debug for Lite<syn::Index> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Index");
        formatter.field("index", Lite(&self.value.index));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Item::Const(_val) => {
                let mut formatter = formatter.debug_struct("Item::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("ty", Lite(&_val.ty));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Item::Enum(_val) => {
                let mut formatter = formatter.debug_struct("Item::Enum");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                if !_val.variants.is_empty() {
                    formatter.field("variants", Lite(&_val.variants));
                }
                formatter.finish()
            },
            syn::Item::ExternCrate(_val) => {
                let mut formatter = formatter.debug_struct("Item::ExternCrate");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.rename {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::As, proc_macro2::Ident));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rename", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Item::Fn(_val) => {
                let mut formatter = formatter.debug_struct("Item::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("sig", Lite(&_val.sig));
                formatter.field("block", Lite(&_val.block));
                formatter.finish()
            },
            syn::Item::ForeignMod(_val) => {
                let mut formatter = formatter.debug_struct("Item::ForeignMod");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.unsafety.is_some() {
                    formatter.field("unsafety", &Present);
                }
                formatter.field("abi", Lite(&_val.abi));
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::Impl(_val) => {
                let mut formatter = formatter.debug_struct("Item::Impl");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.defaultness.is_some() {
                    formatter.field("defaultness", &Present);
                }
                if _val.unsafety.is_some() {
                    formatter.field("unsafety", &Present);
                }
                formatter.field("generics", Lite(&_val.generics));
                if let Some(val) = &_val.trait_ {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(
                                &(
                                    &super::Option {
                                        present: self.0 .0.is_some(),
                                    },
                                    Lite(&self.0 .1),
                                ),
                                formatter,
                            )?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("trait_", Print::ref_cast(val));
                }
                formatter.field("self_ty", Lite(&_val.self_ty));
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Item::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.ident {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(proc_macro2::Ident);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("ident", Print::ref_cast(val));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::Item::Mod(_val) => {
                let mut formatter = formatter.debug_struct("Item::Mod");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.unsafety.is_some() {
                    formatter.field("unsafety", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.content {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Brace, Vec<syn::Item>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("content", Print::ref_cast(val));
                }
                if _val.semi.is_some() {
                    formatter.field("semi", &Present);
                }
                formatter.finish()
            },
            syn::Item::Static(_val) => {
                let mut formatter = formatter.debug_struct("Item::Static");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                match _val.mutability {
                    syn::StaticMutability::None => {},
                    _ => {
                        formatter.field("mutability", Lite(&_val.mutability));
                    },
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("ty", Lite(&_val.ty));
                formatter.field("expr", Lite(&_val.expr));
                formatter.finish()
            },
            syn::Item::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Item::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("fields", Lite(&_val.fields));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::Item::Trait(_val) => {
                let mut formatter = formatter.debug_struct("Item::Trait");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.unsafety.is_some() {
                    formatter.field("unsafety", &Present);
                }
                if _val.auto_token.is_some() {
                    formatter.field("auto_token", &Present);
                }
                if let Some(val) = &_val.restriction {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::ImplRestriction);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("restriction", Print::ref_cast(val));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                if _val.colon_token.is_some() {
                    formatter.field("colon_token", &Present);
                }
                if !_val.supertraits.is_empty() {
                    formatter.field("supertraits", Lite(&_val.supertraits));
                }
                if !_val.items.is_empty() {
                    formatter.field("items", Lite(&_val.items));
                }
                formatter.finish()
            },
            syn::Item::TraitAlias(_val) => {
                let mut formatter = formatter.debug_struct("Item::TraitAlias");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::Item::Type(_val) => {
                let mut formatter = formatter.debug_struct("Item::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("ty", Lite(&_val.ty));
                formatter.finish()
            },
            syn::Item::Union(_val) => {
                let mut formatter = formatter.debug_struct("Item::Union");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("fields", Lite(&_val.fields));
                formatter.finish()
            },
            syn::Item::Use(_val) => {
                let mut formatter = formatter.debug_struct("Item::Use");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("vis", Lite(&_val.vis));
                if _val.leading_colon.is_some() {
                    formatter.field("leading_colon", &Present);
                }
                formatter.field("tree", Lite(&_val.tree));
                formatter.finish()
            },
            syn::Item::Verbatim(_val) => {
                formatter.write_str("Item::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::ItemConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemConst");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemEnum> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemEnum");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        if !self.value.variants.is_empty() {
            formatter.field("variants", Lite(&self.value.variants));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemExternCrate> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemExternCrate");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.rename {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::As, proc_macro2::Ident));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rename", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemFn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemFn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("sig", Lite(&self.value.sig));
        formatter.field("block", Lite(&self.value.block));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemForeignMod> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemForeignMod");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        formatter.field("abi", Lite(&self.value.abi));
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemImpl> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemImpl");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.defaultness.is_some() {
            formatter.field("defaultness", &Present);
        }
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        formatter.field("generics", Lite(&self.value.generics));
        if let Some(val) = &self.value.trait_ {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Option<syn::tok::Not>, syn::Path, syn::tok::For));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(
                        &(
                            &super::Option {
                                present: self.0 .0.is_some(),
                            },
                            Lite(&self.0 .1),
                        ),
                        formatter,
                    )?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("trait_", Print::ref_cast(val));
        }
        formatter.field("self_ty", Lite(&self.value.self_ty));
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.ident {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(proc_macro2::Ident);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("ident", Print::ref_cast(val));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemMod> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemMod");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.content {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Brace, Vec<syn::Item>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("content", Print::ref_cast(val));
        }
        if self.value.semi.is_some() {
            formatter.field("semi", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemStatic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemStatic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        match self.value.mutability {
            syn::StaticMutability::None => {},
            _ => {
                formatter.field("mutability", Lite(&self.value.mutability));
            },
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.field("expr", Lite(&self.value.expr));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("fields", Lite(&self.value.fields));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemTrait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemTrait");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        if self.value.auto_token.is_some() {
            formatter.field("auto_token", &Present);
        }
        if let Some(val) = &self.value.restriction {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::ImplRestriction);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("restriction", Print::ref_cast(val));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        if !self.value.supertraits.is_empty() {
            formatter.field("supertraits", Lite(&self.value.supertraits));
        }
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemTraitAlias> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemTraitAlias");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemType");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemUnion> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemUnion");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("fields", Lite(&self.value.fields));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ItemUse> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ItemUse");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("vis", Lite(&self.value.vis));
        if self.value.leading_colon.is_some() {
            formatter.field("leading_colon", &Present);
        }
        formatter.field("tree", Lite(&self.value.tree));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Label> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Label");
        formatter.field("name", Lite(&self.value.name));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Lifetime> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Lifetime");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.finish()
    }
}
impl Debug for Lite<syn::LifetimeParam> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("LifetimeParam");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("lifetime", Lite(&self.value.lifetime));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Lit::Str(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::ByteStr(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Byte(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Char(_val) => write!(formatter, "{:?}", _val.value()),
            syn::Lit::Int(_val) => write!(formatter, "{}", _val),
            syn::Lit::Float(_val) => write!(formatter, "{}", _val),
            syn::Lit::Bool(_val) => {
                let mut formatter = formatter.debug_struct("Lit::Bool");
                formatter.field("value", Lite(&_val.value));
                formatter.finish()
            },
            syn::Lit::Verbatim(_val) => {
                formatter.write_str("Lit::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::LitBool> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("LitBool");
        formatter.field("value", Lite(&self.value.value));
        formatter.finish()
    }
}
impl Debug for Lite<syn::LitByte> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::LitByteStr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::LitChar> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::LitFloat> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::LitInt> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}", &self.value)
    }
}
impl Debug for Lite<syn::LitStr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{:?}", self.value.value())
    }
}
impl Debug for Lite<syn::Local> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Local");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        if let Some(val) = &self.value.init {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::LocalInit);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("init", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::LocalInit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("LocalInit");
        formatter.field("expr", Lite(&self.value.expr));
        if let Some(val) = &self.value.diverge {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Else, Box<syn::Expr>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("diverge", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Macro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Macro");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("delimiter", Lite(&self.value.delimiter));
        formatter.field("tokens", Lite(&self.value.tokens));
        formatter.finish()
    }
}
impl Debug for Lite<syn::MacroDelimiter> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::MacroDelimiter::Paren(_val) => {
                formatter.write_str("MacroDelimiter::Paren")?;
                Ok(())
            },
            syn::MacroDelimiter::Brace(_val) => {
                formatter.write_str("MacroDelimiter::Brace")?;
                Ok(())
            },
            syn::MacroDelimiter::Bracket(_val) => {
                formatter.write_str("MacroDelimiter::Bracket")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Member> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Member::Named(_val) => {
                formatter.write_str("Member::Named")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Member::Unnamed(_val) => {
                formatter.write_str("Member::Unnamed")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Meta> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Meta::Path(_val) => {
                let mut formatter = formatter.debug_struct("Meta::Path");
                if _val.leading_colon.is_some() {
                    formatter.field("leading_colon", &Present);
                }
                if !_val.segments.is_empty() {
                    formatter.field("segments", Lite(&_val.segments));
                }
                formatter.finish()
            },
            syn::Meta::List(_val) => {
                let mut formatter = formatter.debug_struct("Meta::List");
                formatter.field("path", Lite(&_val.path));
                formatter.field("delimiter", Lite(&_val.delimiter));
                formatter.field("tokens", Lite(&_val.tokens));
                formatter.finish()
            },
            syn::Meta::NameValue(_val) => {
                let mut formatter = formatter.debug_struct("Meta::NameValue");
                formatter.field("path", Lite(&_val.path));
                formatter.field("value", Lite(&_val.value));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::MetaList> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("MetaList");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("delimiter", Lite(&self.value.delimiter));
        formatter.field("tokens", Lite(&self.value.tokens));
        formatter.finish()
    }
}
impl Debug for Lite<syn::MetaNameValue> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("MetaNameValue");
        formatter.field("path", Lite(&self.value.path));
        formatter.field("value", Lite(&self.value.value));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ParenthesizedGenericArguments> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("ParenthesizedGenericArguments");
        if !self.value.inputs.is_empty() {
            formatter.field("inputs", Lite(&self.value.inputs));
        }
        formatter.field("output", Lite(&self.value.output));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Pat> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Pat::Const(_val) => {
                formatter.write_str("Pat::Const")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Pat::Ident(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Ident");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.by_ref.is_some() {
                    formatter.field("by_ref", &Present);
                }
                if _val.mutability.is_some() {
                    formatter.field("mutability", &Present);
                }
                formatter.field("ident", Lite(&_val.ident));
                if let Some(val) = &_val.subpat {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::At, Box<syn::Pat>));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("subpat", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Pat::Lit(_val) => {
                formatter.write_str("Pat::Lit")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Pat::Macro(_val) => {
                formatter.write_str("Pat::Macro")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Pat::Or(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Or");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.leading_vert.is_some() {
                    formatter.field("leading_vert", &Present);
                }
                if !_val.cases.is_empty() {
                    formatter.field("cases", Lite(&_val.cases));
                }
                formatter.finish()
            },
            syn::Pat::Paren(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Paren");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.finish()
            },
            syn::Pat::Path(_val) => {
                formatter.write_str("Pat::Path")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Pat::Range(_val) => {
                formatter.write_str("Pat::Range")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Pat::Reference(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Reference");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if _val.mutability.is_some() {
                    formatter.field("mutability", &Present);
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.finish()
            },
            syn::Pat::Rest(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Rest");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            syn::Pat::Slice(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Slice");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Pat::Struct(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Struct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.fields.is_empty() {
                    formatter.field("fields", Lite(&_val.fields));
                }
                if let Some(val) = &_val.rest {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::PatRest);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("rest", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Pat::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Tuple");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Pat::TupleStruct(_val) => {
                let mut formatter = formatter.debug_struct("Pat::TupleStruct");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Pat::Type(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                formatter.field("ty", Lite(&_val.ty));
                formatter.finish()
            },
            syn::Pat::Verbatim(_val) => {
                formatter.write_str("Pat::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            syn::Pat::Wild(_val) => {
                let mut formatter = formatter.debug_struct("Pat::Wild");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.finish()
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::PatIdent> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatIdent");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.by_ref.is_some() {
            formatter.field("by_ref", &Present);
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        formatter.field("ident", Lite(&self.value.ident));
        if let Some(val) = &self.value.subpat {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::At, Box<syn::Pat>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("subpat", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatOr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatOr");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.leading_vert.is_some() {
            formatter.field("leading_vert", &Present);
        }
        if !self.value.cases.is_empty() {
            formatter.field("cases", Lite(&self.value.cases));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatParen> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatParen");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatReference> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatReference");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatRest> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatRest");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatSlice> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatSlice");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.fields.is_empty() {
            formatter.field("fields", Lite(&self.value.fields));
        }
        if let Some(val) = &self.value.rest {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::PatRest);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("rest", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatTuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatTuple");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatTupleStruct> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatTupleStruct");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatType");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("pat", Lite(&self.value.pat));
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::PatWild> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PatWild");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Path> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Path");
        if self.value.leading_colon.is_some() {
            formatter.field("leading_colon", &Present);
        }
        if !self.value.segments.is_empty() {
            formatter.field("segments", Lite(&self.value.segments));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PathArguments> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::PathArguments::None => formatter.write_str("PathArguments::None"),
            syn::PathArguments::AngleBracketed(_val) => {
                let mut formatter = formatter.debug_struct("PathArguments::AngleBracketed");
                if _val.colon2_token.is_some() {
                    formatter.field("colon2_token", &Present);
                }
                if !_val.args.is_empty() {
                    formatter.field("args", Lite(&_val.args));
                }
                formatter.finish()
            },
            syn::PathArguments::Parenthesized(_val) => {
                let mut formatter = formatter.debug_struct("PathArguments::Parenthesized");
                if !_val.inputs.is_empty() {
                    formatter.field("inputs", Lite(&_val.inputs));
                }
                formatter.field("output", Lite(&_val.output));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::PathSegment> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PathSegment");
        formatter.field("ident", Lite(&self.value.ident));
        match self.value.arguments {
            syn::PathArguments::None => {},
            _ => {
                formatter.field("arguments", Lite(&self.value.arguments));
            },
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PredicateLifetime> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateLifetime");
        formatter.field("lifetime", Lite(&self.value.lifetime));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::PredicateType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("PredicateType");
        if let Some(val) = &self.value.lifetimes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::BoundLifetimes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        formatter.field("bounded_ty", Lite(&self.value.bounded_ty));
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::QSelf> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("QSelf");
        formatter.field("ty", Lite(&self.value.ty));
        formatter.field("position", Lite(&self.value.position));
        if self.value.as_token.is_some() {
            formatter.field("as_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::RangeLimits> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::RangeLimits::HalfOpen(_val) => {
                formatter.write_str("RangeLimits::HalfOpen")?;
                Ok(())
            },
            syn::RangeLimits::Closed(_val) => {
                formatter.write_str("RangeLimits::Closed")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Receiver> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Receiver");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.reference {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::And, Option<syn::Lifetime>));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(
                        {
                            #[derive(RefCast)]
                            #[repr(transparent)]
                            struct Print(Option<syn::Lifetime>);
                            impl Debug for Print {
                                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                    match &self.0 {
                                        Some(_val) => {
                                            formatter.write_str("Some(")?;
                                            Debug::fmt(Lite(_val), formatter)?;
                                            formatter.write_str(")")?;
                                            Ok(())
                                        },
                                        None => formatter.write_str("None"),
                                    }
                                }
                            }
                            Print::ref_cast(&self.0 .1)
                        },
                        formatter,
                    )?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("reference", Print::ref_cast(val));
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        formatter.field("ty", Lite(&self.value.ty));
        formatter.finish()
    }
}
impl Debug for Lite<syn::ReturnType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::ReturnType::Default => formatter.write_str("ReturnType::Default"),
            syn::ReturnType::Type(_v0, _v1) => {
                let mut formatter = formatter.debug_tuple("ReturnType::Type");
                formatter.field(Lite(_v1));
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::Signature> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Signature");
        if self.value.constness.is_some() {
            formatter.field("constness", &Present);
        }
        if self.value.asyncness.is_some() {
            formatter.field("asyncness", &Present);
        }
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("abi", Print::ref_cast(val));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        if !self.value.inputs.is_empty() {
            formatter.field("inputs", Lite(&self.value.inputs));
        }
        if let Some(val) = &self.value.variadic {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Variadic);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("variadic", Print::ref_cast(val));
        }
        formatter.field("output", Lite(&self.value.output));
        formatter.finish()
    }
}
impl Debug for Lite<syn::StaticMutability> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::StaticMutability::Mut(_val) => {
                formatter.write_str("StaticMutability::Mut")?;
                Ok(())
            },
            syn::StaticMutability::None => formatter.write_str("StaticMutability::None"),
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::Stmt> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Stmt::Local(_val) => {
                let mut formatter = formatter.debug_struct("Stmt::Local");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("pat", Lite(&_val.pat));
                if let Some(val) = &_val.init {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::LocalInit);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("init", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::Stmt::Item(_val) => {
                formatter.write_str("Stmt::Item")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::Stmt::Expr(_v0, _v1) => {
                let mut formatter = formatter.debug_tuple("Stmt::Expr");
                formatter.field(Lite(_v0));
                formatter.field(&super::Option { present: _v1.is_some() });
                formatter.finish()
            },
            syn::Stmt::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Stmt::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
        }
    }
}
impl Debug for Lite<syn::StmtMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("StmtMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TraitBound> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitBound");
        if self.value.paren_token.is_some() {
            formatter.field("paren_token", &Present);
        }
        match self.value.modifier {
            syn::TraitBoundModifier::None => {},
            _ => {
                formatter.field("modifier", Lite(&self.value.modifier));
            },
        }
        if let Some(val) = &self.value.lifetimes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::BoundLifetimes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TraitBoundModifier> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::TraitBoundModifier::None => formatter.write_str("TraitBoundModifier::None"),
            syn::TraitBoundModifier::Maybe(_val) => {
                formatter.write_str("TraitBoundModifier::Maybe")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::TraitItem> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::TraitItem::Const(_val) => {
                let mut formatter = formatter.debug_struct("TraitItem::Const");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                formatter.field("ty", Lite(&_val.ty));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::Expr));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::TraitItem::Fn(_val) => {
                let mut formatter = formatter.debug_struct("TraitItem::Fn");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("sig", Lite(&_val.sig));
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Block);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::TraitItem::Type(_val) => {
                let mut formatter = formatter.debug_struct("TraitItem::Type");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("ident", Lite(&_val.ident));
                formatter.field("generics", Lite(&_val.generics));
                if _val.colon_token.is_some() {
                    formatter.field("colon_token", &Present);
                }
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                if let Some(val) = &_val.default {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print((syn::tok::Eq, syn::Type));
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0 .1), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("default", Print::ref_cast(val));
                }
                formatter.finish()
            },
            syn::TraitItem::Macro(_val) => {
                let mut formatter = formatter.debug_struct("TraitItem::Macro");
                if !_val.attrs.is_empty() {
                    formatter.field("attrs", Lite(&_val.attrs));
                }
                formatter.field("mac", Lite(&_val.mac));
                if _val.semi_token.is_some() {
                    formatter.field("semi_token", &Present);
                }
                formatter.finish()
            },
            syn::TraitItem::Verbatim(_val) => {
                formatter.write_str("TraitItem::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::TraitItemConst> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitItemConst");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        formatter.field("ty", Lite(&self.value.ty));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TraitItemFn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitItemFn");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("sig", Lite(&self.value.sig));
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Block);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TraitItemMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitItemMacro");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("mac", Lite(&self.value.mac));
        if self.value.semi_token.is_some() {
            formatter.field("semi_token", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TraitItemType> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TraitItemType");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("generics", Lite(&self.value.generics));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Type));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Type> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Type::Array(_val) => {
                let mut formatter = formatter.debug_struct("Type::Array");
                formatter.field("elem", Lite(&_val.elem));
                formatter.field("len", Lite(&_val.len));
                formatter.finish()
            },
            syn::Type::BareFn(_val) => {
                let mut formatter = formatter.debug_struct("Type::BareFn");
                if let Some(val) = &_val.lifetimes {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::BoundLifetimes);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("lifetimes", Print::ref_cast(val));
                }
                if _val.unsafety.is_some() {
                    formatter.field("unsafety", &Present);
                }
                if let Some(val) = &_val.abi {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Abi);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("abi", Print::ref_cast(val));
                }
                if !_val.inputs.is_empty() {
                    formatter.field("inputs", Lite(&_val.inputs));
                }
                if let Some(val) = &_val.variadic {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::BareVariadic);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("variadic", Print::ref_cast(val));
                }
                formatter.field("output", Lite(&_val.output));
                formatter.finish()
            },
            syn::Type::Group(_val) => {
                let mut formatter = formatter.debug_struct("Type::Group");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::Type::ImplTrait(_val) => {
                let mut formatter = formatter.debug_struct("Type::ImplTrait");
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::Type::Infer(_val) => {
                let mut formatter = formatter.debug_struct("Type::Infer");
                formatter.finish()
            },
            syn::Type::Macro(_val) => {
                let mut formatter = formatter.debug_struct("Type::Macro");
                formatter.field("mac", Lite(&_val.mac));
                formatter.finish()
            },
            syn::Type::Never(_val) => {
                let mut formatter = formatter.debug_struct("Type::Never");
                formatter.finish()
            },
            syn::Type::Paren(_val) => {
                let mut formatter = formatter.debug_struct("Type::Paren");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::Type::Path(_val) => {
                let mut formatter = formatter.debug_struct("Type::Path");
                if let Some(val) = &_val.qself {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::QSelf);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("qself", Print::ref_cast(val));
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::Type::Ptr(_val) => {
                let mut formatter = formatter.debug_struct("Type::Ptr");
                if _val.const_token.is_some() {
                    formatter.field("const_token", &Present);
                }
                if _val.mutability.is_some() {
                    formatter.field("mutability", &Present);
                }
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::Type::Reference(_val) => {
                let mut formatter = formatter.debug_struct("Type::Reference");
                if let Some(val) = &_val.lifetime {
                    #[derive(RefCast)]
                    #[repr(transparent)]
                    struct Print(syn::Lifetime);
                    impl Debug for Print {
                        fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str("Some(")?;
                            Debug::fmt(Lite(&self.0), formatter)?;
                            formatter.write_str(")")?;
                            Ok(())
                        }
                    }
                    formatter.field("lifetime", Print::ref_cast(val));
                }
                if _val.mutability.is_some() {
                    formatter.field("mutability", &Present);
                }
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::Type::Slice(_val) => {
                let mut formatter = formatter.debug_struct("Type::Slice");
                formatter.field("elem", Lite(&_val.elem));
                formatter.finish()
            },
            syn::Type::TraitObject(_val) => {
                let mut formatter = formatter.debug_struct("Type::TraitObject");
                if _val.dyn_token.is_some() {
                    formatter.field("dyn_token", &Present);
                }
                if !_val.bounds.is_empty() {
                    formatter.field("bounds", Lite(&_val.bounds));
                }
                formatter.finish()
            },
            syn::Type::Tuple(_val) => {
                let mut formatter = formatter.debug_struct("Type::Tuple");
                if !_val.elems.is_empty() {
                    formatter.field("elems", Lite(&_val.elems));
                }
                formatter.finish()
            },
            syn::Type::Verbatim(_val) => {
                formatter.write_str("Type::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::TypeArray> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeArray");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.field("len", Lite(&self.value.len));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeBareFn> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeBareFn");
        if let Some(val) = &self.value.lifetimes {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::BoundLifetimes);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetimes", Print::ref_cast(val));
        }
        if self.value.unsafety.is_some() {
            formatter.field("unsafety", &Present);
        }
        if let Some(val) = &self.value.abi {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Abi);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("abi", Print::ref_cast(val));
        }
        if !self.value.inputs.is_empty() {
            formatter.field("inputs", Lite(&self.value.inputs));
        }
        if let Some(val) = &self.value.variadic {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::BareVariadic);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("variadic", Print::ref_cast(val));
        }
        formatter.field("output", Lite(&self.value.output));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeGroup> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeGroup");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeImplTrait> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeImplTrait");
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeInfer> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeInfer");
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeMacro> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeMacro");
        formatter.field("mac", Lite(&self.value.mac));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeNever> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeNever");
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeParam> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeParam");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        if self.value.colon_token.is_some() {
            formatter.field("colon_token", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        if self.value.eq_token.is_some() {
            formatter.field("eq_token", &Present);
        }
        if let Some(val) = &self.value.default {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Type);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("default", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeParamBound> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::TypeParamBound::Trait(_val) => {
                formatter.write_str("TypeParamBound::Trait")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::TypeParamBound::Lifetime(_val) => {
                let mut formatter = formatter.debug_struct("TypeParamBound::Lifetime");
                formatter.field("ident", Lite(&_val.ident));
                formatter.finish()
            },
            syn::TypeParamBound::Verbatim(_val) => {
                formatter.write_str("TypeParamBound::Verbatim")?;
                formatter.write_str("(`")?;
                Display::fmt(_val, formatter)?;
                formatter.write_str("`)")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::TypeParen> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeParen");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypePath> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypePath");
        if let Some(val) = &self.value.qself {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::QSelf);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("qself", Print::ref_cast(val));
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypePtr> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypePtr");
        if self.value.const_token.is_some() {
            formatter.field("const_token", &Present);
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeReference> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeReference");
        if let Some(val) = &self.value.lifetime {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print(syn::Lifetime);
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("lifetime", Print::ref_cast(val));
        }
        if self.value.mutability.is_some() {
            formatter.field("mutability", &Present);
        }
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeSlice> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeSlice");
        formatter.field("elem", Lite(&self.value.elem));
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeTraitObject> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeTraitObject");
        if self.value.dyn_token.is_some() {
            formatter.field("dyn_token", &Present);
        }
        if !self.value.bounds.is_empty() {
            formatter.field("bounds", Lite(&self.value.bounds));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::TypeTuple> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("TypeTuple");
        if !self.value.elems.is_empty() {
            formatter.field("elems", Lite(&self.value.elems));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::UnOp> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::UnOp::Deref(_val) => {
                formatter.write_str("UnOp::Deref")?;
                Ok(())
            },
            syn::UnOp::Not(_val) => {
                formatter.write_str("UnOp::Not")?;
                Ok(())
            },
            syn::UnOp::Neg(_val) => {
                formatter.write_str("UnOp::Neg")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
impl Debug for Lite<syn::UseGlob> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseGlob");
        formatter.finish()
    }
}
impl Debug for Lite<syn::UseGroup> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseGroup");
        if !self.value.items.is_empty() {
            formatter.field("items", Lite(&self.value.items));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::UseName> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseName");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.finish()
    }
}
impl Debug for Lite<syn::UsePath> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UsePath");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("tree", Lite(&self.value.tree));
        formatter.finish()
    }
}
impl Debug for Lite<syn::UseRename> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("UseRename");
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("rename", Lite(&self.value.rename));
        formatter.finish()
    }
}
impl Debug for Lite<syn::UseTree> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::UseTree::Path(_val) => {
                formatter.write_str("UseTree::Path")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::UseTree::Name(_val) => {
                formatter.write_str("UseTree::Name")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::UseTree::Rename(_val) => {
                formatter.write_str("UseTree::Rename")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::UseTree::Glob(_val) => {
                formatter.write_str("UseTree::Glob")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::UseTree::Group(_val) => {
                formatter.write_str("UseTree::Group")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
        }
    }
}
impl Debug for Lite<syn::Variadic> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Variadic");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        if let Some(val) = &self.value.pat {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((Box<syn::Pat>, syn::tok::Colon));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .0), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("pat", Print::ref_cast(val));
        }
        if self.value.comma.is_some() {
            formatter.field("comma", &Present);
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::Variant> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("Variant");
        if !self.value.attrs.is_empty() {
            formatter.field("attrs", Lite(&self.value.attrs));
        }
        formatter.field("ident", Lite(&self.value.ident));
        formatter.field("fields", Lite(&self.value.fields));
        if let Some(val) = &self.value.discriminant {
            #[derive(RefCast)]
            #[repr(transparent)]
            struct Print((syn::tok::Eq, syn::Expr));
            impl Debug for Print {
                fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Some(")?;
                    Debug::fmt(Lite(&self.0 .1), formatter)?;
                    formatter.write_str(")")?;
                    Ok(())
                }
            }
            formatter.field("discriminant", Print::ref_cast(val));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::VisRestricted> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("VisRestricted");
        if self.value.in_token.is_some() {
            formatter.field("in_token", &Present);
        }
        formatter.field("path", Lite(&self.value.path));
        formatter.finish()
    }
}
impl Debug for Lite<syn::Visibility> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::Visibility::Public(_val) => {
                formatter.write_str("Visibility::Public")?;
                Ok(())
            },
            syn::Visibility::Restricted(_val) => {
                let mut formatter = formatter.debug_struct("Visibility::Restricted");
                if _val.in_token.is_some() {
                    formatter.field("in_token", &Present);
                }
                formatter.field("path", Lite(&_val.path));
                formatter.finish()
            },
            syn::Visibility::Inherited => formatter.write_str("Visibility::Inherited"),
        }
    }
}
impl Debug for Lite<syn::WhereClause> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let mut formatter = formatter.debug_struct("WhereClause");
        if !self.value.predicates.is_empty() {
            formatter.field("predicates", Lite(&self.value.predicates));
        }
        formatter.finish()
    }
}
impl Debug for Lite<syn::WherePredicate> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            syn::WherePredicate::Lifetime(_val) => {
                formatter.write_str("WherePredicate::Lifetime")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            syn::WherePredicate::Type(_val) => {
                formatter.write_str("WherePredicate::Type")?;
                formatter.write_str("(")?;
                Debug::fmt(Lite(_val), formatter)?;
                formatter.write_str(")")?;
                Ok(())
            },
            _ => unreachable!(),
        }
    }
}
