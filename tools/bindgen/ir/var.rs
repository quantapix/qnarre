use super::dot::DotAttrs;
use super::function::cursor_mangling;
use super::int::IntKind;
use super::item::Item;
use super::typ::{FloatKind, TypeKind};
use super::{Context, TypeId};
use crate::callbacks::{ItemInfo, ItemKind, MacroParsing};
use crate::clang;
use crate::clang::Token;
use crate::codegen::utils::variation;
use crate::parse;
use std::io;
use std::num::Wrapping;

#[derive(Debug)]
pub enum VarKind {
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(u8),
    String(Vec<u8>),
}

#[derive(Debug)]
pub struct Var {
    is_const: bool,
    link: Option<String>,
    mangled: Option<String>,
    name: String,
    ty: TypeId,
    val: Option<VarKind>,
}
impl Var {
    pub fn new(
        name: String,
        mangled: Option<String>,
        link: Option<String>,
        ty: TypeId,
        val: Option<VarKind>,
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
    pub fn link(&self) -> Option<&str> {
        self.link.as_deref()
    }
    pub fn mangled(&self) -> Option<&str> {
        self.mangled.as_deref()
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn ty(&self) -> TypeId {
        self.ty
    }
    pub fn val(&self) -> Option<&VarKind> {
        self.val.as_ref()
    }
}
impl DotAttrs for Var {
    fn dot_attrs<W>(&self, _ctx: &Context, y: &mut W) -> io::Result<()>
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
impl parse::SubItem for Var {
    fn parse(cur: clang::Cursor, ctx: &mut Context) -> Result<parse::Resolved<Self>, parse::Error> {
        use cexpr::expr::EvalResult;
        use cexpr::literal::CChar;
        use clang_lib::*;
        match cur.kind() {
            CXCursor_MacroDefinition => {
                for x in &ctx.opts().parse_callbacks {
                    match x.will_parse_macro(&cur.spelling()) {
                        MacroParsing::Ignore => {
                            return Err(parse::Error::Continue);
                        },
                        MacroParsing::Default => {},
                    }
                    if cur.is_macro_function_like() {
                        handle_fn_macro(&cur, x.as_ref());
                        return Err(parse::Error::Continue);
                    }
                }
                let y = parse_macro(ctx, &cur);
                let (id, y) = match y {
                    Some(x) => x,
                    None => return Err(parse::Error::Continue),
                };
                assert!(!id.is_empty(), "Empty macro name?");
                let prev = ctx.parsed_macro(&id);
                ctx.note_parsed_macro(id.clone(), y.clone());
                if prev {
                    return Err(parse::Error::Continue);
                }
                let name = String::from_utf8(id).unwrap();
                let (ty, y) = match y {
                    EvalResult::Invalid => return Err(parse::Error::Continue),
                    EvalResult::Float(x) => (TypeKind::Float(FloatKind::Double), VarKind::Float(x)),
                    EvalResult::Char(x) => {
                        let x = match x {
                            CChar::Char(x) => {
                                assert_eq!(x.len_utf8(), 1);
                                x as u8
                            },
                            CChar::Raw(x) => {
                                assert!(x <= ::std::u8::MAX as u64);
                                x as u8
                            },
                        };
                        (TypeKind::Int(IntKind::U8), VarKind::Char(x))
                    },
                    EvalResult::Str(x) => {
                        let ty = Item::builtin_type(TypeKind::Int(IntKind::U8), true, ctx);
                        for x2 in &ctx.opts().parse_callbacks {
                            x2.str_macro(&name, &x);
                        }
                        (TypeKind::Pointer(ty), VarKind::String(x))
                    },
                    EvalResult::Int(Wrapping(x)) => {
                        let ty = ctx
                            .opts()
                            .last_callback(|x| x.int_macro(&name, x))
                            .unwrap_or_else(|| default_macro_const(ctx, x));
                        (TypeKind::Int(ty), VarKind::Int(x))
                    },
                };
                let ty = Item::builtin_type(ty, true, ctx);
                Ok(parse::Resolved::New(
                    Var::new(name, None, None, ty, Some(y), true),
                    Some(cur),
                ))
            },
            CXCursor_VarDecl => {
                let mut name = cur.spelling();
                if cur.linkage() == CXLinkage_External {
                    if let Some(x) = ctx.opts().last_callback(|x| {
                        x.generated_name_override(ItemInfo {
                            name: name.as_str(),
                            kind: ItemKind::Var,
                        })
                    }) {
                        name = x;
                    }
                }
                let name = name;
                if name.is_empty() {
                    warn!("Empty constant name?");
                    return Err(parse::Error::Continue);
                }
                let link = ctx.opts().last_callback(|x| {
                    x.generated_link_name_override(ItemInfo {
                        name: name.as_str(),
                        kind: ItemKind::Var,
                    })
                });
                let ty = cur.cur_type();
                let is_const = ty.is_const()
                    || ([CXType_ConstantArray, CXType_IncompleteArray].contains(&ty.kind())
                        && ty.elem_type().map_or(false, |x| x.is_const()));
                let ty = match Item::from_ty(&ty, cur, None, ctx) {
                    Ok(x) => x,
                    Err(x) => {
                        assert!(matches!(ty.kind(), CXType_Auto | CXType_Unexposed));
                        return Err(x);
                    },
                };
                let ty2 = ctx.safe_resolve_type(ty).and_then(|x| x.safe_canon_type(ctx));
                let is_integer = ty2.map_or(false, |x| x.is_integer());
                let is_float = ty2.map_or(false, |x| x.is_float());
                let y = if is_integer {
                    let kind = match *ty2.unwrap().kind() {
                        TypeKind::Int(x) => x,
                        _ => unreachable!(),
                    };
                    let mut y = cur.evaluate().and_then(|x| x.as_int());
                    if y.is_none() || !kind.signedness_matches(y.unwrap()) {
                        y = get_int_literal(&cur);
                    }
                    y.map(|x| {
                        if kind == IntKind::Bool {
                            VarKind::Bool(x != 0)
                        } else {
                            VarKind::Int(x)
                        }
                    })
                } else if is_float {
                    cur.evaluate().and_then(|x| x.as_double()).map(VarKind::Float)
                } else {
                    cur.evaluate().and_then(|x| x.as_literal_string()).map(VarKind::String)
                };
                let mangled = cursor_mangling(ctx, &cur);
                let y = Var::new(name, mangled, link, ty, y, is_const);
                Ok(parse::Resolved::New(y, Some(cur)))
            },
            _ => Err(parse::Error::Continue),
        }
    }
}

fn default_macro_const(ctx: &Context, x: i64) -> IntKind {
    if x < 0 || ctx.opts().default_macro_const == variation::MacroType::Signed {
        if x < i32::min_value() as i64 || x > i32::max_value() as i64 {
            IntKind::I64
        } else if !ctx.opts().fit_macro_const || x < i16::min_value() as i64 || x > i16::max_value() as i64 {
            IntKind::I32
        } else if x < i8::min_value() as i64 || x > i8::max_value() as i64 {
            IntKind::I16
        } else {
            IntKind::I8
        }
    } else if x > u32::max_value() as i64 {
        IntKind::U64
    } else if !ctx.opts().fit_macro_const || x > u16::max_value() as i64 {
        IntKind::U32
    } else if x > u8::max_value() as i64 {
        IntKind::U16
    } else {
        IntKind::U8
    }
}

fn handle_fn_macro(cur: &clang::Cursor, cb: &dyn crate::callbacks::Parse) {
    let is_closing = |x: &Token| x.kind == clang_lib::CXToken_Punctuation && x.spelling() == b")";
    let toks: Vec<_> = cur.tokens().iter().collect();
    if let Some(x) = toks.iter().position(is_closing) {
        let mut y = toks.iter().map(Token::spelling);
        let left = y.by_ref().take(x + 1);
        let left = left.collect::<Vec<_>>().concat();
        if let Ok(x) = String::from_utf8(left) {
            let right: Vec<_> = y.collect();
            cb.func_macro(&x, &right);
        }
    }
}

fn parse_macro(ctx: &Context, cur: &clang::Cursor) -> Option<(Vec<u8>, cexpr::expr::EvalResult)> {
    use cexpr::expr;
    let toks = cur.cexpr_tokens();
    let y = expr::IdentifierParser::new(ctx.parsed_macros());
    match y.macro_definition(&toks) {
        Ok((_, (id, x))) => Some((id.into(), x)),
        _ => None,
    }
}

fn parse_int_literal(cur: &clang::Cursor) -> Option<i64> {
    use cexpr::expr;
    use cexpr::expr::EvalResult;
    let y = cur.cexpr_tokens();
    match expr::expr(&y) {
        Ok((_, EvalResult::Int(Wrapping(x)))) => Some(x),
        _ => None,
    }
}

fn get_int_literal(cur: &clang::Cursor) -> Option<i64> {
    use clang_lib::*;
    let mut y = None;
    cur.visit(|x| {
        match x.kind() {
            CXCursor_IntegerLiteral | CXCursor_UnaryOperator => {
                y = parse_int_literal(&x);
            },
            CXCursor_UnexposedExpr => {
                y = get_int_literal(&x);
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
