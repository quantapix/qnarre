use super::super::codegen::MacroTypeVariation;
use super::context::{BindgenContext, TypeId};
use super::dot::DotAttrs;
use super::function::cursor_mangling;
use super::int::IntKind;
use super::item::Item;
use super::ty::{FloatKind, TyKind};
use crate::callbacks::{ItemInfo, ItemKind, MacroParsing};
use crate::clang;
use crate::clang::Token;
use crate::parse;
use std::io;
use std::num::Wrapping;

#[derive(Debug)]
pub enum VarType {
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(u8),
    String(Vec<u8>),
}
#[derive(Debug)]
pub struct Var {
    name: String,
    mangled: Option<String>,
    link: Option<String>,
    ty: TypeId,
    val: Option<VarType>,
    is_const: bool,
}
impl Var {
    pub fn new(
        name: String,
        mangled: Option<String>,
        link: Option<String>,
        ty: TypeId,
        val: Option<VarType>,
        is_const: bool,
    ) -> Var {
        assert!(!name.is_empty());
        Var {
            name,
            mangled,
            link,
            ty,
            val,
            is_const,
        }
    }
    pub fn is_const(&self) -> bool {
        self.is_const
    }
    pub fn val(&self) -> Option<&VarType> {
        self.val.as_ref()
    }
    pub fn ty(&self) -> TypeId {
        self.ty
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn mangled(&self) -> Option<&str> {
        self.mangled.as_deref()
    }
    pub fn link(&self) -> Option<&str> {
        self.link.as_deref()
    }
}
impl DotAttrs for Var {
    fn dot_attrs<W>(&self, _ctx: &BindgenContext, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if self.is_const {
            writeln!(y, "<tr><td>const</td><td>true</td></tr>")?;
        }
        if let Some(ref x) = self.mangled {
            writeln!(y, "<tr><td>mangled name</td><td>{}</td></tr>", x)?;
        }
        Ok(())
    }
}
fn default_macro_constant_type(ctx: &BindgenContext, value: i64) -> IntKind {
    if value < 0 || ctx.opts().default_macro_constant_type == MacroTypeVariation::Signed {
        if value < i32::min_value() as i64 || value > i32::max_value() as i64 {
            IntKind::I64
        } else if !ctx.opts().fit_macro_constants || value < i16::min_value() as i64 || value > i16::max_value() as i64
        {
            IntKind::I32
        } else if value < i8::min_value() as i64 || value > i8::max_value() as i64 {
            IntKind::I16
        } else {
            IntKind::I8
        }
    } else if value > u32::max_value() as i64 {
        IntKind::U64
    } else if !ctx.opts().fit_macro_constants || value > u16::max_value() as i64 {
        IntKind::U32
    } else if value > u8::max_value() as i64 {
        IntKind::U16
    } else {
        IntKind::U8
    }
}
fn handle_function_macro(cur: &clang::Cursor, callbacks: &dyn crate::callbacks::Parse) {
    let is_closing_paren = |t: &Token| t.kind == clang_lib::CXToken_Punctuation && t.spelling() == b")";
    let tokens: Vec<_> = cur.tokens().iter().collect();
    if let Some(boundary) = tokens.iter().position(is_closing_paren) {
        let mut spelled = tokens.iter().map(Token::spelling);
        let left = spelled.by_ref().take(boundary + 1);
        let left = left.collect::<Vec<_>>().concat();
        if let Ok(left) = String::from_utf8(left) {
            let right: Vec<_> = spelled.collect();
            callbacks.func_macro(&left, &right);
        }
    }
}
impl parse::SubItem for Var {
    fn parse(cur: clang::Cursor, ctx: &mut BindgenContext) -> Result<parse::Result<Self>, parse::Error> {
        use cexpr::expr::EvalResult;
        use cexpr::literal::CChar;
        use clang_lib::*;
        match cur.kind() {
            CXCursor_MacroDefinition => {
                for callbacks in &ctx.opts().parse_callbacks {
                    match callbacks.will_parse_macro(&cur.spelling()) {
                        MacroParsing::Ignore => {
                            return Err(parse::Error::Continue);
                        },
                        MacroParsing::Default => {},
                    }
                    if cur.is_macro_function_like() {
                        handle_function_macro(&cur, callbacks.as_ref());
                        return Err(parse::Error::Continue);
                    }
                }
                let value = parse_macro(ctx, &cur);
                let (id, value) = match value {
                    Some(v) => v,
                    None => return Err(parse::Error::Continue),
                };
                assert!(!id.is_empty(), "Empty macro name?");
                let previously_defined = ctx.parsed_macro(&id);
                ctx.note_parsed_macro(id.clone(), value.clone());
                if previously_defined {
                    let name = String::from_utf8(id).unwrap();
                    duplicated_macro_diagnostic(&name, cur.location(), ctx);
                    return Err(parse::Error::Continue);
                }
                let name = String::from_utf8(id).unwrap();
                let (type_kind, val) = match value {
                    EvalResult::Invalid => return Err(parse::Error::Continue),
                    EvalResult::Float(f) => (TyKind::Float(FloatKind::Double), VarType::Float(f)),
                    EvalResult::Char(c) => {
                        let c = match c {
                            CChar::Char(c) => {
                                assert_eq!(c.len_utf8(), 1);
                                c as u8
                            },
                            CChar::Raw(c) => {
                                assert!(c <= ::std::u8::MAX as u64);
                                c as u8
                            },
                        };
                        (TyKind::Int(IntKind::U8), VarType::Char(c))
                    },
                    EvalResult::Str(val) => {
                        let char_ty = Item::builtin_type(TyKind::Int(IntKind::U8), true, ctx);
                        for callbacks in &ctx.opts().parse_callbacks {
                            callbacks.str_macro(&name, &val);
                        }
                        (TyKind::Pointer(char_ty), VarType::String(val))
                    },
                    EvalResult::Int(Wrapping(value)) => {
                        let kind = ctx
                            .opts()
                            .last_callback(|c| c.int_macro(&name, value))
                            .unwrap_or_else(|| default_macro_constant_type(ctx, value));
                        (TyKind::Int(kind), VarType::Int(value))
                    },
                };
                let ty = Item::builtin_type(type_kind, true, ctx);
                Ok(parse::Result::New(
                    Var::new(name, None, None, ty, Some(val), true),
                    Some(cur),
                ))
            },
            CXCursor_VarDecl => {
                let mut name = cur.spelling();
                if cur.linkage() == CXLinkage_External {
                    if let Some(nm) = ctx.opts().last_callback(|callbacks| {
                        callbacks.generated_name_override(ItemInfo {
                            name: name.as_str(),
                            kind: ItemKind::Var,
                        })
                    }) {
                        name = nm;
                    }
                }
                let name = name;
                if name.is_empty() {
                    warn!("Empty constant name?");
                    return Err(parse::Error::Continue);
                }
                let link_name = ctx.opts().last_callback(|callbacks| {
                    callbacks.generated_link_name_override(ItemInfo {
                        name: name.as_str(),
                        kind: ItemKind::Var,
                    })
                });
                let ty = cur.cur_type();
                let is_const = ty.is_const()
                    || ([CXType_ConstantArray, CXType_IncompleteArray].contains(&ty.kind())
                        && ty.elem_type().map_or(false, |element| element.is_const()));
                let ty = match Item::from_ty(&ty, cur, None, ctx) {
                    Ok(ty) => ty,
                    Err(e) => {
                        assert!(
                            matches!(ty.kind(), CXType_Auto | CXType_Unexposed),
                            "Couldn't resolve constant type, and it \
                             wasn't an nondeductible auto type or unexposed \
                             type!"
                        );
                        return Err(e);
                    },
                };
                let canonical_ty = ctx.safe_resolve_type(ty).and_then(|t| t.safe_canonical_type(ctx));
                let is_integer = canonical_ty.map_or(false, |t| t.is_integer());
                let is_float = canonical_ty.map_or(false, |t| t.is_float());
                let value = if is_integer {
                    let kind = match *canonical_ty.unwrap().kind() {
                        TyKind::Int(kind) => kind,
                        _ => unreachable!(),
                    };
                    let mut val = cur.evaluate().and_then(|v| v.as_int());
                    if val.is_none() || !kind.signedness_matches(val.unwrap()) {
                        val = get_integer_literal_from_cursor(&cur);
                    }
                    val.map(|val| {
                        if kind == IntKind::Bool {
                            VarType::Bool(val != 0)
                        } else {
                            VarType::Int(val)
                        }
                    })
                } else if is_float {
                    cur.evaluate().and_then(|v| v.as_double()).map(VarType::Float)
                } else {
                    cur.evaluate().and_then(|v| v.as_literal_string()).map(VarType::String)
                };
                let mangling = cursor_mangling(ctx, &cur);
                let var = Var::new(name, mangling, link_name, ty, value, is_const);
                Ok(parse::Result::New(var, Some(cur)))
            },
            _ => {
                /* TODO */
                Err(parse::Error::Continue)
            },
        }
    }
}
fn parse_macro(ctx: &BindgenContext, cur: &clang::Cursor) -> Option<(Vec<u8>, cexpr::expr::EvalResult)> {
    use cexpr::expr;
    let cexpr_tokens = cur.cexpr_tokens();
    let y = expr::IdentifierParser::new(ctx.parsed_macros());
    match y.macro_definition(&cexpr_tokens) {
        Ok((_, (id, val))) => Some((id.into(), val)),
        _ => None,
    }
}
fn parse_int_literal_tokens(cur: &clang::Cursor) -> Option<i64> {
    use cexpr::expr;
    use cexpr::expr::EvalResult;
    let y = cur.cexpr_tokens();
    match expr::expr(&y) {
        Ok((_, EvalResult::Int(Wrapping(x)))) => Some(x),
        _ => None,
    }
}
fn get_integer_literal_from_cursor(cur: &clang::Cursor) -> Option<i64> {
    use clang_lib::*;
    let mut y = None;
    cur.visit(|c| {
        match c.kind() {
            CXCursor_IntegerLiteral | CXCursor_UnaryOperator => {
                y = parse_int_literal_tokens(&c);
            },
            CXCursor_UnexposedExpr => {
                y = get_integer_literal_from_cursor(&c);
            },
            _ => (),
        }
        if y.is_some() {
            CXChildVisit_Break
        } else {
            CXChildVisit_Continue
        }
    });
    y
}
fn duplicated_macro_diagnostic(macro_name: &str, _location: crate::clang::SrcLoc, _ctx: &BindgenContext) {
    warn!("Duplicated macro definition: {}", macro_name);
}
