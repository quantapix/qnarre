use super::comp::MethodKind;
use super::context::{BindgenContext, TypeId};
use super::dot::DotAttributes;
use super::item::Item;
use super::traversal::{EdgeKind, Trace, Tracer};
use super::ty::TypeKind;
use crate::callbacks::{ItemInfo, ItemKind};
use crate::clang::{self, Attribute};
use crate::parse::{ClangSubItemParser, ParseError, ParseResult};
use clang::{self, CXCallingConv};

use quote::TokenStreamExt;
use std::io;
use std::str::FromStr;

const RUST_DERIVE_FUNPTR_LIMIT: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum FunctionKind {
    Function,
    Method(MethodKind),
}

impl FunctionKind {
    pub(crate) fn from_cursor(cursor: &clang::Cursor) -> Option<FunctionKind> {
        Some(match cursor.kind() {
            clang::CXCursor_FunctionDecl => FunctionKind::Function,
            clang::CXCursor_Constructor => FunctionKind::Method(MethodKind::Constructor),
            clang::CXCursor_Destructor => FunctionKind::Method(if cursor.method_is_virtual() {
                MethodKind::VirtualDestructor {
                    pure_virtual: cursor.method_is_pure_virtual(),
                }
            } else {
                MethodKind::Destructor
            }),
            clang::CXCursor_CXXMethod => {
                if cursor.method_is_virtual() {
                    FunctionKind::Method(MethodKind::Virtual {
                        pure_virtual: cursor.method_is_pure_virtual(),
                    })
                } else if cursor.method_is_static() {
                    FunctionKind::Method(MethodKind::Static)
                } else {
                    FunctionKind::Method(MethodKind::Normal)
                }
            },
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Linkage {
    External,
    Internal,
}

#[derive(Debug)]
pub(crate) struct Function {
    name: String,

    mangled_name: Option<String>,

    link_name: Option<String>,

    signature: TypeId,

    kind: FunctionKind,

    linkage: Linkage,
}

impl Function {
    pub(crate) fn new(
        name: String,
        mangled_name: Option<String>,
        link_name: Option<String>,
        signature: TypeId,
        kind: FunctionKind,
        linkage: Linkage,
    ) -> Self {
        Function {
            name,
            mangled_name,
            link_name,
            signature,
            kind,
            linkage,
        }
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

    pub(crate) fn signature(&self) -> TypeId {
        self.signature
    }

    pub(crate) fn kind(&self) -> FunctionKind {
        self.kind
    }

    pub(crate) fn linkage(&self) -> Linkage {
        self.linkage
    }
}

impl DotAttributes for Function {
    fn dot_attributes<W>(&self, _ctx: &BindgenContext, out: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if let Some(ref mangled) = self.mangled_name {
            let mangled: String = mangled.chars().flat_map(|c| c.escape_default()).collect();
            writeln!(out, "<tr><td>mangled name</td><td>{}</td></tr>", mangled)?;
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum Abi {
    C,
    Stdcall,
    EfiApi,
    Fastcall,
    ThisCall,
    Vectorcall,
    Aapcs,
    Win64,
    CUnwind,
}

impl FromStr for Abi {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "C" => Ok(Self::C),
            "stdcall" => Ok(Self::Stdcall),
            "efiapi" => Ok(Self::EfiApi),
            "fastcall" => Ok(Self::Fastcall),
            "thiscall" => Ok(Self::ThisCall),
            "vectorcall" => Ok(Self::Vectorcall),
            "aapcs" => Ok(Self::Aapcs),
            "win64" => Ok(Self::Win64),
            "C-unwind" => Ok(Self::CUnwind),
            _ => Err(format!("Invalid or unknown ABI {:?}", s)),
        }
    }
}

impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match *self {
            Self::C => "C",
            Self::Stdcall => "stdcall",
            Self::EfiApi => "efiapi",
            Self::Fastcall => "fastcall",
            Self::ThisCall => "thiscall",
            Self::Vectorcall => "vectorcall",
            Self::Aapcs => "aapcs",
            Self::Win64 => "win64",
            Self::CUnwind => "C-unwind",
        };

        s.fmt(f)
    }
}

impl quote::ToTokens for Abi {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let abi = self.to_string();
        tokens.append_all(quote! { #abi });
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum ClangAbi {
    Known(Abi),
    Unknown(CXCallingConv),
}

impl ClangAbi {
    fn is_unknown(&self) -> bool {
        matches!(*self, ClangAbi::Unknown(..))
    }
}

impl quote::ToTokens for ClangAbi {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match *self {
            Self::Known(abi) => abi.to_tokens(tokens),
            Self::Unknown(cc) => panic!("Cannot turn unknown calling convention to tokens: {:?}", cc),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FunctionSig {
    name: String,

    return_type: TypeId,

    argument_types: Vec<(Option<String>, TypeId)>,

    is_variadic: bool,
    is_divergent: bool,

    must_use: bool,

    abi: ClangAbi,
}

fn get_abi(cc: CXCallingConv) -> ClangAbi {
    use clang::*;
    match cc {
        CXCallingConv_Default => ClangAbi::Known(Abi::C),
        CXCallingConv_C => ClangAbi::Known(Abi::C),
        CXCallingConv_X86StdCall => ClangAbi::Known(Abi::Stdcall),
        CXCallingConv_X86FastCall => ClangAbi::Known(Abi::Fastcall),
        CXCallingConv_X86ThisCall => ClangAbi::Known(Abi::ThisCall),
        CXCallingConv_X86VectorCall => ClangAbi::Known(Abi::Vectorcall),
        CXCallingConv_AAPCS => ClangAbi::Known(Abi::Aapcs),
        CXCallingConv_X86_64Win64 => ClangAbi::Known(Abi::Win64),
        other => ClangAbi::Unknown(other),
    }
}

pub(crate) fn cursor_mangling(ctx: &BindgenContext, cursor: &clang::Cursor) -> Option<String> {
    if !ctx.options().enable_mangling {
        return None;
    }

    if cursor.is_in_non_fully_specialized_template() {
        return None;
    }

    let is_destructor = cursor.kind() == clang::CXCursor_Destructor;
    if let Ok(mut manglings) = cursor.cxx_manglings() {
        while let Some(m) = manglings.pop() {
            if is_destructor && !m.ends_with("D1Ev") {
                continue;
            }

            return Some(m);
        }
    }

    let mut mangling = cursor.mangling();
    if mangling.is_empty() {
        return None;
    }

    if is_destructor {
        if mangling.ends_with("D0Ev") {
            let new_len = mangling.len() - 4;
            mangling.truncate(new_len);
            mangling.push_str("D1Ev");
        }
    }

    Some(mangling)
}

fn args_from_ty_and_cursor(
    ty: &clang::Type,
    cursor: &clang::Cursor,
    ctx: &mut BindgenContext,
) -> Vec<(Option<String>, TypeId)> {
    let cursor_args = cursor.args().unwrap_or_default().into_iter();
    let type_args = ty.args().unwrap_or_default().into_iter();

    cursor_args
        .map(Some)
        .chain(std::iter::repeat(None))
        .zip(type_args.map(Some).chain(std::iter::repeat(None)))
        .take_while(|(cur, ty)| cur.is_some() || ty.is_some())
        .map(|(arg_cur, arg_ty)| {
            let name = arg_cur
                .map(|a| a.spelling())
                .and_then(|name| if name.is_empty() { None } else { Some(name) });

            let cursor = arg_cur.unwrap_or(*cursor);
            let ty = arg_ty.unwrap_or_else(|| cursor.cur_type());
            (name, Item::from_ty_or_ref(ty, cursor, None, ctx))
        })
        .collect()
}

impl FunctionSig {
    pub(crate) fn from_ty(
        ty: &clang::Type,
        cursor: &clang::Cursor,
        ctx: &mut BindgenContext,
    ) -> Result<Self, ParseError> {
        use clang::*;
        debug!("FunctionSig::from_ty {:?} {:?}", ty, cursor);

        let kind = cursor.kind();
        if kind == CXCursor_FunctionTemplate {
            return Err(ParseError::Continue);
        }

        let spelling = cursor.spelling();

        let is_operator = |spelling: &str| spelling.starts_with("operator") && !clang::is_valid_identifier(spelling);
        if is_operator(&spelling) {
            return Err(ParseError::Continue);
        }

        if (kind == CXCursor_Constructor || kind == CXCursor_Destructor) && spelling.contains('<') {
            return Err(ParseError::Continue);
        }

        let cursor = if cursor.is_valid() { *cursor } else { ty.declaration() };

        let mut args = match kind {
            CXCursor_FunctionDecl
            | CXCursor_Constructor
            | CXCursor_CXXMethod
            | CXCursor_ObjCInstanceMethodDecl
            | CXCursor_ObjCClassMethodDecl => args_from_ty_and_cursor(ty, &cursor, ctx),
            _ => {
                let mut args = vec![];
                cursor.visit(|c| {
                    if c.kind() == CXCursor_ParmDecl {
                        let ty = Item::from_ty_or_ref(c.cur_type(), c, None, ctx);
                        let name = c.spelling();
                        let name = if name.is_empty() { None } else { Some(name) };
                        args.push((name, ty));
                    }
                    CXChildVisit_Continue
                });

                if args.is_empty() {
                    args_from_ty_and_cursor(ty, &cursor, ctx)
                } else {
                    args
                }
            },
        };

        let (must_use, mut is_divergent) = if ctx.options().enable_function_attribute_detection {
            let [must_use, no_return, no_return_cpp] =
                cursor.has_attrs(&[Attribute::MUST_USE, Attribute::NO_RETURN, Attribute::NO_RETURN_CPP]);
            (must_use, no_return || no_return_cpp)
        } else {
            Default::default()
        };

        is_divergent = is_divergent || ty.spelling().contains("__attribute__((noreturn))");

        let is_method = kind == CXCursor_CXXMethod;
        let is_constructor = kind == CXCursor_Constructor;
        let is_destructor = kind == CXCursor_Destructor;
        if (is_constructor || is_destructor || is_method) && cursor.lexical_parent() != cursor.semantic_parent() {
            return Err(ParseError::Continue);
        }

        if is_method || is_constructor || is_destructor {
            let is_const = is_method && cursor.method_is_const();
            let is_virtual = is_method && cursor.method_is_virtual();
            let is_static = is_method && cursor.method_is_static();
            if !is_static && !is_virtual {
                let parent = cursor.semantic_parent();
                let class = Item::parse(parent, None, ctx).expect("Expected to parse the class");
                let class = class.as_type_id_unchecked();

                let class = if is_const {
                    let const_class_id = ctx.next_item_id();
                    ctx.build_const_wrapper(const_class_id, class, None, &parent.cur_type())
                } else {
                    class
                };

                let ptr = Item::builtin_type(TypeKind::Pointer(class), false, ctx);
                args.insert(0, (Some("this".into()), ptr));
            } else if is_virtual {
                let void = Item::builtin_type(TypeKind::Void, false, ctx);
                let ptr = Item::builtin_type(TypeKind::Pointer(void), false, ctx);
                args.insert(0, (Some("this".into()), ptr));
            }
        }

        let ty_ret_type = if kind == CXCursor_ObjCInstanceMethodDecl || kind == CXCursor_ObjCClassMethodDecl {
            ty.ret_type()
                .or_else(|| cursor.ret_type())
                .ok_or(ParseError::Continue)?
        } else {
            ty.ret_type().ok_or(ParseError::Continue)?
        };

        let ret = if is_constructor && ctx.is_target_wasm32() {
            let void = Item::builtin_type(TypeKind::Void, false, ctx);
            Item::builtin_type(TypeKind::Pointer(void), false, ctx)
        } else {
            Item::from_ty_or_ref(ty_ret_type, cursor, None, ctx)
        };

        let mut call_conv = ty.call_conv();
        if let Some(ty) = cursor.cur_type().canonical_type().pointee_type() {
            let cursor_call_conv = ty.call_conv();
            if cursor_call_conv != CXCallingConv_Invalid {
                call_conv = cursor_call_conv;
            }
        }

        let abi = get_abi(call_conv);

        if abi.is_unknown() {
            warn!("Unknown calling convention: {:?}", call_conv);
        }

        Ok(Self {
            name: spelling,
            return_type: ret,
            argument_types: args,
            is_variadic: ty.is_variadic(),
            is_divergent,
            must_use,
            abi,
        })
    }

    pub(crate) fn return_type(&self) -> TypeId {
        self.return_type
    }

    pub(crate) fn argument_types(&self) -> &[(Option<String>, TypeId)] {
        &self.argument_types
    }

    pub(crate) fn abi(&self, ctx: &BindgenContext, name: Option<&str>) -> ClangAbi {
        if let Some(name) = name {
            if let Some((abi, _)) = ctx
                .options()
                .abi_overrides
                .iter()
                .find(|(_, regex_set)| regex_set.matches(name))
            {
                ClangAbi::Known(*abi)
            } else {
                self.abi
            }
        } else if let Some((abi, _)) = ctx
            .options()
            .abi_overrides
            .iter()
            .find(|(_, regex_set)| regex_set.matches(&self.name))
        {
            ClangAbi::Known(*abi)
        } else {
            self.abi
        }
    }

    pub(crate) fn is_variadic(&self) -> bool {
        self.is_variadic && !self.argument_types.is_empty()
    }

    pub(crate) fn must_use(&self) -> bool {
        self.must_use
    }

    pub(crate) fn function_pointers_can_derive(&self) -> bool {
        if self.argument_types.len() > RUST_DERIVE_FUNPTR_LIMIT {
            return false;
        }

        matches!(self.abi, ClangAbi::Known(Abi::C) | ClangAbi::Unknown(..))
    }

    pub(crate) fn is_divergent(&self) -> bool {
        self.is_divergent
    }
}

impl ClangSubItemParser for Function {
    fn parse(cursor: clang::Cursor, context: &mut BindgenContext) -> Result<ParseResult<Self>, ParseError> {
        use clang::*;

        let kind = match FunctionKind::from_cursor(&cursor) {
            None => return Err(ParseError::Continue),
            Some(k) => k,
        };

        debug!("Function::parse({:?}, {:?})", cursor, cursor.cur_type());
        let visibility = cursor.visibility();
        if visibility != CXVisibility_Default {
            return Err(ParseError::Continue);
        }

        if cursor.access_specifier() == CX_CXXPrivate {
            return Err(ParseError::Continue);
        }

        let linkage = cursor.linkage();
        let linkage = match linkage {
            CXLinkage_External | CXLinkage_UniqueExternal => Linkage::External,
            CXLinkage_Internal => Linkage::Internal,
            _ => return Err(ParseError::Continue),
        };

        if cursor.is_inlined_function() || cursor.definition().map_or(false, |x| x.is_inlined_function()) {
            if !context.options().generate_inline_fns && !context.options().wrap_static_fns {
                return Err(ParseError::Continue);
            }

            if cursor.is_deleted_function() {
                return Err(ParseError::Continue);
            }

            if context.options().wrap_static_fns && cursor.is_inlined_function() && matches!(linkage, Linkage::External)
            {
                return Err(ParseError::Continue);
            }
        }

        let sig = Item::from_ty(&cursor.cur_type(), cursor, None, context)?;

        let mut name = cursor.spelling();
        assert!(!name.is_empty(), "Empty function name?");

        if cursor.kind() == CXCursor_Destructor {
            if name.starts_with('~') {
                name.remove(0);
            }

            name.push_str("_destructor");
        }
        if let Some(nm) = context.options().last_callback(|callbacks| {
            callbacks.generated_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Function,
            })
        }) {
            name = nm;
        }
        assert!(!name.is_empty(), "Empty function name.");

        let mangled_name = cursor_mangling(context, &cursor);

        let link_name = context.options().last_callback(|callbacks| {
            callbacks.generated_link_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Function,
            })
        });

        let function = Self::new(name.clone(), mangled_name, link_name, sig, kind, linkage);

        Ok(ParseResult::New(function, Some(cursor)))
    }
}

impl Trace for FunctionSig {
    type Extra = ();

    fn trace<T>(&self, _: &BindgenContext, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        tracer.visit_kind(self.return_type().into(), EdgeKind::FunctionReturn);

        for &(_, ty) in self.argument_types() {
            tracer.visit_kind(ty.into(), EdgeKind::FunctionParameter);
        }
    }
}
