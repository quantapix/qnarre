use super::super::codegen::MacroTypeVariation;
use super::context::{BindgenContext, TypeId};
use super::dot::DotAttrs;
use super::function::cursor_mangling;
use super::int::IntKind;
use super::item::Item;
use super::ty::{FloatKind, TypeKind};
use crate::callbacks::{ItemInfo, ItemKind, MacroParsing};
use crate::clang;
use crate::clang::ClangToken;
use crate::parse::{ClangSubItemParser, ParseError, ParseResult};

use std::io;
use std::num::Wrapping;

#[derive(Debug)]
pub(crate) enum VarType {
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(u8),
    String(Vec<u8>),
}

#[derive(Debug)]
pub(crate) struct Var {
    name: String,
    mangled_name: Option<String>,
    link_name: Option<String>,
    ty: TypeId,
    val: Option<VarType>,
    is_const: bool,
}

impl Var {
    pub(crate) fn new(
        name: String,
        mangled_name: Option<String>,
        link_name: Option<String>,
        ty: TypeId,
        val: Option<VarType>,
        is_const: bool,
    ) -> Var {
        assert!(!name.is_empty());
        Var {
            name,
            mangled_name,
            link_name,
            ty,
            val,
            is_const,
        }
    }

    pub(crate) fn is_const(&self) -> bool {
        self.is_const
    }

    pub(crate) fn val(&self) -> Option<&VarType> {
        self.val.as_ref()
    }

    pub(crate) fn ty(&self) -> TypeId {
        self.ty
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn mangled_name(&self) -> Option<&str> {
        self.mangled_name.as_deref()
    }

    pub fn link_name(&self) -> Option<&str> {
        self.link_name.as_deref()
    }
}

impl DotAttrs for Var {
    fn dot_attributes<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if self.is_const {
            writeln!(out, "<tr><td>const</td><td>true</td></tr>")?;
        }

        if let Some(ref mangled) = self.mangled_name {
            writeln!(out, "<tr><td>mangled name</td><td>{}</td></tr>", mangled)?;
        }

        Ok(())
    }
}

fn default_macro_constant_type(ctx: &BindgenContext, value: i64) -> IntKind {
    if value < 0 || ctx.options().default_macro_constant_type == MacroTypeVariation::Signed {
        if value < i32::min_value() as i64 || value > i32::max_value() as i64 {
            IntKind::I64
        } else if !ctx.options().fit_macro_constants
            || value < i16::min_value() as i64
            || value > i16::max_value() as i64
        {
            IntKind::I32
        } else if value < i8::min_value() as i64 || value > i8::max_value() as i64 {
            IntKind::I16
        } else {
            IntKind::I8
        }
    } else if value > u32::max_value() as i64 {
        IntKind::U64
    } else if !ctx.options().fit_macro_constants || value > u16::max_value() as i64 {
        IntKind::U32
    } else if value > u8::max_value() as i64 {
        IntKind::U16
    } else {
        IntKind::U8
    }
}

fn handle_function_macro(cursor: &clang::Cursor, callbacks: &dyn crate::callbacks::ParseCallbacks) {
    let is_closing_paren = |t: &ClangToken| t.kind == clang_lib::CXToken_Punctuation && t.spelling() == b")";
    let tokens: Vec<_> = cursor.tokens().iter().collect();
    if let Some(boundary) = tokens.iter().position(is_closing_paren) {
        let mut spelled = tokens.iter().map(ClangToken::spelling);
        let left = spelled.by_ref().take(boundary + 1);
        let left = left.collect::<Vec<_>>().concat();
        if let Ok(left) = String::from_utf8(left) {
            let right: Vec<_> = spelled.collect();
            callbacks.func_macro(&left, &right);
        }
    }
}

impl ClangSubItemParser for Var {
    fn parse(cursor: clang::Cursor, ctx: &mut BindgenContext) -> Result<ParseResult<Self>, ParseError> {
        use cexpr::expr::EvalResult;
        use cexpr::literal::CChar;
        use clang_lib::*;
        match cursor.kind() {
            CXCursor_MacroDefinition => {
                for callbacks in &ctx.options().parse_callbacks {
                    match callbacks.will_parse_macro(&cursor.spelling()) {
                        MacroParsing::Ignore => {
                            return Err(ParseError::Continue);
                        },
                        MacroParsing::Default => {},
                    }

                    if cursor.is_macro_function_like() {
                        handle_function_macro(&cursor, callbacks.as_ref());
                        return Err(ParseError::Continue);
                    }
                }

                let value = parse_macro(ctx, &cursor);

                let (id, value) = match value {
                    Some(v) => v,
                    None => return Err(ParseError::Continue),
                };

                assert!(!id.is_empty(), "Empty macro name?");

                let previously_defined = ctx.parsed_macro(&id);

                ctx.note_parsed_macro(id.clone(), value.clone());

                if previously_defined {
                    let name = String::from_utf8(id).unwrap();
                    duplicated_macro_diagnostic(&name, cursor.location(), ctx);
                    return Err(ParseError::Continue);
                }

                let name = String::from_utf8(id).unwrap();
                let (type_kind, val) = match value {
                    EvalResult::Invalid => return Err(ParseError::Continue),
                    EvalResult::Float(f) => (TypeKind::Float(FloatKind::Double), VarType::Float(f)),
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

                        (TypeKind::Int(IntKind::U8), VarType::Char(c))
                    },
                    EvalResult::Str(val) => {
                        let char_ty = Item::builtin_type(TypeKind::Int(IntKind::U8), true, ctx);
                        for callbacks in &ctx.options().parse_callbacks {
                            callbacks.str_macro(&name, &val);
                        }
                        (TypeKind::Pointer(char_ty), VarType::String(val))
                    },
                    EvalResult::Int(Wrapping(value)) => {
                        let kind = ctx
                            .options()
                            .last_callback(|c| c.int_macro(&name, value))
                            .unwrap_or_else(|| default_macro_constant_type(ctx, value));

                        (TypeKind::Int(kind), VarType::Int(value))
                    },
                };

                let ty = Item::builtin_type(type_kind, true, ctx);

                Ok(ParseResult::New(
                    Var::new(name, None, None, ty, Some(val), true),
                    Some(cursor),
                ))
            },
            CXCursor_VarDecl => {
                let mut name = cursor.spelling();
                if cursor.linkage() == CXLinkage_External {
                    if let Some(nm) = ctx.options().last_callback(|callbacks| {
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
                    return Err(ParseError::Continue);
                }

                let link_name = ctx.options().last_callback(|callbacks| {
                    callbacks.generated_link_name_override(ItemInfo {
                        name: name.as_str(),
                        kind: ItemKind::Var,
                    })
                });

                let ty = cursor.cur_type();

                let is_const = ty.is_const()
                    || ([CXType_ConstantArray, CXType_IncompleteArray].contains(&ty.kind())
                        && ty.elem_type().map_or(false, |element| element.is_const()));

                let ty = match Item::from_ty(&ty, cursor, None, ctx) {
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
                        TypeKind::Int(kind) => kind,
                        _ => unreachable!(),
                    };

                    let mut val = cursor.evaluate().and_then(|v| v.as_int());
                    if val.is_none() || !kind.signedness_matches(val.unwrap()) {
                        val = get_integer_literal_from_cursor(&cursor);
                    }

                    val.map(|val| {
                        if kind == IntKind::Bool {
                            VarType::Bool(val != 0)
                        } else {
                            VarType::Int(val)
                        }
                    })
                } else if is_float {
                    cursor.evaluate().and_then(|v| v.as_double()).map(VarType::Float)
                } else {
                    cursor
                        .evaluate()
                        .and_then(|v| v.as_literal_string())
                        .map(VarType::String)
                };

                let mangling = cursor_mangling(ctx, &cursor);
                let var = Var::new(name, mangling, link_name, ty, value, is_const);

                Ok(ParseResult::New(var, Some(cursor)))
            },
            _ => {
                /* TODO */
                Err(ParseError::Continue)
            },
        }
    }
}

fn parse_macro(ctx: &BindgenContext, cursor: &clang::Cursor) -> Option<(Vec<u8>, cexpr::expr::EvalResult)> {
    use cexpr::expr;

    let cexpr_tokens = cursor.cexpr_tokens();

    let parser = expr::IdentifierParser::new(ctx.parsed_macros());

    match parser.macro_definition(&cexpr_tokens) {
        Ok((_, (id, val))) => Some((id.into(), val)),
        _ => None,
    }
}

fn parse_int_literal_tokens(cursor: &clang::Cursor) -> Option<i64> {
    use cexpr::expr;
    use cexpr::expr::EvalResult;

    let cexpr_tokens = cursor.cexpr_tokens();

    match expr::expr(&cexpr_tokens) {
        Ok((_, EvalResult::Int(Wrapping(val)))) => Some(val),
        _ => None,
    }
}

fn get_integer_literal_from_cursor(cursor: &clang::Cursor) -> Option<i64> {
    use clang_lib::*;
    let mut value = None;
    cursor.visit(|c| {
        match c.kind() {
            CXCursor_IntegerLiteral | CXCursor_UnaryOperator => {
                value = parse_int_literal_tokens(&c);
            },
            CXCursor_UnexposedExpr => {
                value = get_integer_literal_from_cursor(&c);
            },
            _ => (),
        }
        if value.is_some() {
            CXChildVisit_Break
        } else {
            CXChildVisit_Continue
        }
    });
    value
}

fn duplicated_macro_diagnostic(macro_name: &str, _location: crate::clang::SourceLocation, _ctx: &BindgenContext) {
    warn!("Duplicated macro definition: {}", macro_name);

    #[cfg(feature = "experimental")]
    #[allow(clippy::overly_complex_bool_expr)]
    if false && _ctx.options().emit_diagnostics {
        use crate::diagnostics::{get_line, Diagnostic, Level, Slice};
        use std::borrow::Cow;

        let mut slice = Slice::default();
        let mut source = Cow::from(macro_name);

        let (file, line, col, _) = _location.location();
        if let Some(filename) = file.name() {
            if let Ok(Some(code)) = get_line(&filename, line) {
                source = code.into();
            }
            slice.with_location(filename, line, col);
        }

        slice.with_source(source);

        Diagnostic::default()
            .with_title("Duplicated macro definition.", Level::Warn)
            .add_slice(slice)
            .add_annotation("This macro had a duplicate.", Level::Note)
            .display();
    }
}
