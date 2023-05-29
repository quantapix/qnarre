use super::comp::MethodKind;
use super::context::{BindgenContext, TypeId};
use super::dot::DotAttrs;
use super::item::Item;
use super::traversal::{EdgeKind, Trace, Tracer};
use super::ty::TyKind;
use crate::callbacks::{ItemInfo, ItemKind};
use crate::clang::{self, Attribute};
use crate::parse;
use clang_lib::{self, CXCallingConv};
use quote::TokenStreamExt;
use std::io;
use std::str::FromStr;
const RUST_DERIVE_FUNPTR_LIMIT: usize = 12;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FnKind {
    Function,
    Method(MethodKind),
}
impl FnKind {
    pub fn from_cursor(cur: &clang::Cursor) -> Option<FnKind> {
        Some(match cur.kind() {
            clang_lib::CXCursor_FunctionDecl => FnKind::Function,
            clang_lib::CXCursor_Constructor => FnKind::Method(MethodKind::Constructor),
            clang_lib::CXCursor_Destructor => FnKind::Method(if cur.method_is_virtual() {
                MethodKind::VirtualDestructor {
                    pure_virtual: cur.method_is_pure_virtual(),
                }
            } else {
                MethodKind::Destructor
            }),
            clang_lib::CXCursor_CXXMethod => {
                if cur.method_is_virtual() {
                    FnKind::Method(MethodKind::Virtual {
                        pure_virtual: cur.method_is_pure_virtual(),
                    })
                } else if cur.method_is_static() {
                    FnKind::Method(MethodKind::Static)
                } else {
                    FnKind::Method(MethodKind::Normal)
                }
            },
            _ => return None,
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub enum Linkage {
    External,
    Internal,
}
#[derive(Debug)]
pub struct Function {
    name: String,
    mangled_name: Option<String>,
    link_name: Option<String>,
    sig: TypeId,
    kind: FnKind,
    linkage: Linkage,
}
impl Function {
    pub fn new(
        name: String,
        mangled_name: Option<String>,
        link_name: Option<String>,
        signature: TypeId,
        kind: FnKind,
        linkage: Linkage,
    ) -> Self {
        Function {
            name,
            mangled_name,
            link_name,
            sig: signature,
            kind,
            linkage,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn mangled_name(&self) -> Option<&str> {
        self.mangled_name.as_deref()
    }
    pub fn link_name(&self) -> Option<&str> {
        self.link_name.as_deref()
    }
    pub fn sig(&self) -> TypeId {
        self.sig
    }
    pub fn kind(&self) -> FnKind {
        self.kind
    }
    pub fn linkage(&self) -> Linkage {
        self.linkage
    }
}
impl DotAttrs for Function {
    fn dot_attrs<W>(&self, _ctx: &BindgenContext, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if let Some(ref x) = self.mangled_name {
            let x: String = x.chars().flat_map(|c| c.escape_default()).collect();
            writeln!(y, "<tr><td>mangled name</td><td>{}</td></tr>", x)?;
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
    fn from_str(x: &str) -> Result<Self, Self::Err> {
        match x {
            "C" => Ok(Self::C),
            "stdcall" => Ok(Self::Stdcall),
            "efiapi" => Ok(Self::EfiApi),
            "fastcall" => Ok(Self::Fastcall),
            "thiscall" => Ok(Self::ThisCall),
            "vectorcall" => Ok(Self::Vectorcall),
            "aapcs" => Ok(Self::Aapcs),
            "win64" => Ok(Self::Win64),
            "C-unwind" => Ok(Self::CUnwind),
            _ => Err(format!("Invalid or unknown ABI {:?}", x)),
        }
    }
}
impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let y = match *self {
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
        y.fmt(f)
    }
}
impl quote::ToTokens for Abi {
    fn to_tokens(&self, toks: &mut proc_macro2::TokenStream) {
        let abi = self.to_string();
        toks.append_all(quote! { #abi });
    }
}
#[derive(Debug, Copy, Clone)]
pub enum ClangAbi {
    Known(Abi),
    Unknown(CXCallingConv),
}
impl ClangAbi {
    fn is_unknown(&self) -> bool {
        matches!(*self, ClangAbi::Unknown(..))
    }
}
impl quote::ToTokens for ClangAbi {
    fn to_tokens(&self, toks: &mut proc_macro2::TokenStream) {
        match *self {
            Self::Known(x) => x.to_tokens(toks),
            Self::Unknown(x) => panic!("Cannot turn unknown calling convention to tokens: {:?}", x),
        }
    }
}
#[derive(Debug)]
pub struct FnSig {
    name: String,
    return_type: TypeId,
    argument_types: Vec<(Option<String>, TypeId)>,
    is_variadic: bool,
    is_divergent: bool,
    must_use: bool,
    abi: ClangAbi,
}
fn get_abi(x: CXCallingConv) -> ClangAbi {
    use clang_lib::*;
    match x {
        CXCallingConv_Default => ClangAbi::Known(Abi::C),
        CXCallingConv_C => ClangAbi::Known(Abi::C),
        CXCallingConv_X86StdCall => ClangAbi::Known(Abi::Stdcall),
        CXCallingConv_X86FastCall => ClangAbi::Known(Abi::Fastcall),
        CXCallingConv_X86ThisCall => ClangAbi::Known(Abi::ThisCall),
        CXCallingConv_X86VectorCall => ClangAbi::Known(Abi::Vectorcall),
        CXCallingConv_AAPCS => ClangAbi::Known(Abi::Aapcs),
        CXCallingConv_X86_64Win64 => ClangAbi::Known(Abi::Win64),
        x => ClangAbi::Unknown(x),
    }
}
pub fn cursor_mangling(ctx: &BindgenContext, cur: &clang::Cursor) -> Option<String> {
    if !ctx.opts().enable_mangling {
        return None;
    }
    if cur.is_in_non_fully_specialized_template() {
        return None;
    }
    let is_destructor = cur.kind() == clang_lib::CXCursor_Destructor;
    if let Ok(mut manglings) = cur.cxx_manglings() {
        while let Some(m) = manglings.pop() {
            if is_destructor && !m.ends_with("D1Ev") {
                continue;
            }
            return Some(m);
        }
    }
    let mut mangling = cur.mangling();
    if mangling.is_empty() {
        return None;
    }
    if is_destructor && mangling.ends_with("D0Ev") {
        let new_len = mangling.len() - 4;
        mangling.truncate(new_len);
        mangling.push_str("D1Ev");
    }
    Some(mangling)
}
fn args_from_ty_and_cursor(
    ty: &clang::Type,
    cur: &clang::Cursor,
    ctx: &mut BindgenContext,
) -> Vec<(Option<String>, TypeId)> {
    let cursor_args = cur.args().unwrap_or_default().into_iter();
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
            let cursor = arg_cur.unwrap_or(*cur);
            let ty = arg_ty.unwrap_or_else(|| cursor.cur_type());
            (name, Item::from_ty_or_ref(ty, cursor, None, ctx))
        })
        .collect()
}
impl FnSig {
    pub fn from_ty(ty: &clang::Type, cur: &clang::Cursor, ctx: &mut BindgenContext) -> Result<Self, parse::Error> {
        use clang_lib::*;
        debug!("FunctionSig::from_ty {:?} {:?}", ty, cur);
        let kind = cur.kind();
        if kind == CXCursor_FunctionTemplate {
            return Err(parse::Error::Continue);
        }
        let spelling = cur.spelling();
        let is_operator = |spelling: &str| spelling.starts_with("operator") && !clang::is_valid_identifier(spelling);
        if is_operator(&spelling) {
            return Err(parse::Error::Continue);
        }
        if (kind == CXCursor_Constructor || kind == CXCursor_Destructor) && spelling.contains('<') {
            return Err(parse::Error::Continue);
        }
        let cur2 = if cur.is_valid() { *cur } else { ty.declaration() };
        let mut args = match kind {
            CXCursor_FunctionDecl | CXCursor_Constructor | CXCursor_CXXMethod => {
                args_from_ty_and_cursor(ty, &cur2, ctx)
            },
            _ => {
                let mut args = vec![];
                cur2.visit(|c| {
                    if c.kind() == CXCursor_ParmDecl {
                        let ty = Item::from_ty_or_ref(c.cur_type(), c, None, ctx);
                        let name = c.spelling();
                        let name = if name.is_empty() { None } else { Some(name) };
                        args.push((name, ty));
                    }
                    CXChildVisit_Continue
                });
                if args.is_empty() {
                    args_from_ty_and_cursor(ty, &cur2, ctx)
                } else {
                    args
                }
            },
        };
        let (must_use, mut is_divergent) = if ctx.opts().enable_function_attribute_detection {
            let [must_use, no_return, no_return_cpp] =
                cur2.has_attrs(&[Attribute::MUST_USE, Attribute::NO_RETURN, Attribute::NO_RETURN_CPP]);
            (must_use, no_return || no_return_cpp)
        } else {
            Default::default()
        };
        is_divergent = is_divergent || ty.spelling().contains("__attribute__((noreturn))");
        let is_method = kind == CXCursor_CXXMethod;
        let is_constructor = kind == CXCursor_Constructor;
        let is_destructor = kind == CXCursor_Destructor;
        if (is_constructor || is_destructor || is_method) && cur2.lexical_parent() != cur2.semantic_parent() {
            return Err(parse::Error::Continue);
        }
        if is_method || is_constructor || is_destructor {
            let is_const = is_method && cur2.method_is_const();
            let is_virtual = is_method && cur2.method_is_virtual();
            let is_static = is_method && cur2.method_is_static();
            if !is_static && !is_virtual {
                let parent = cur2.semantic_parent();
                let class = Item::parse(parent, None, ctx).expect("Expected to parse the class");
                let class = class.as_type_id_unchecked();
                let class = if is_const {
                    let const_class_id = ctx.next_item_id();
                    ctx.build_const_wrapper(const_class_id, class, None, &parent.cur_type())
                } else {
                    class
                };
                let ptr = Item::builtin_type(TyKind::Pointer(class), false, ctx);
                args.insert(0, (Some("this".into()), ptr));
            } else if is_virtual {
                let void = Item::builtin_type(TyKind::Void, false, ctx);
                let ptr = Item::builtin_type(TyKind::Pointer(void), false, ctx);
                args.insert(0, (Some("this".into()), ptr));
            }
        }
        let ty_ret_type = ty.ret_type().ok_or(parse::Error::Continue)?;
        let ret = if is_constructor && ctx.is_target_wasm32() {
            let void = Item::builtin_type(TyKind::Void, false, ctx);
            Item::builtin_type(TyKind::Pointer(void), false, ctx)
        } else {
            Item::from_ty_or_ref(ty_ret_type, cur2, None, ctx)
        };
        let mut call_conv = ty.call_conv();
        if let Some(ty) = cur2.cur_type().canonical_type().pointee_type() {
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
    pub fn return_type(&self) -> TypeId {
        self.return_type
    }
    pub fn argument_types(&self) -> &[(Option<String>, TypeId)] {
        &self.argument_types
    }
    pub fn abi(&self, ctx: &BindgenContext, name: Option<&str>) -> ClangAbi {
        if let Some(name) = name {
            if let Some((abi, _)) = ctx
                .opts()
                .abi_overrides
                .iter()
                .find(|(_, regex_set)| regex_set.matches(name))
            {
                ClangAbi::Known(*abi)
            } else {
                self.abi
            }
        } else if let Some((abi, _)) = ctx
            .opts()
            .abi_overrides
            .iter()
            .find(|(_, regex_set)| regex_set.matches(&self.name))
        {
            ClangAbi::Known(*abi)
        } else {
            self.abi
        }
    }
    pub fn is_variadic(&self) -> bool {
        self.is_variadic && !self.argument_types.is_empty()
    }
    pub fn must_use(&self) -> bool {
        self.must_use
    }
    pub fn function_pointers_can_derive(&self) -> bool {
        if self.argument_types.len() > RUST_DERIVE_FUNPTR_LIMIT {
            return false;
        }
        matches!(self.abi, ClangAbi::Known(Abi::C) | ClangAbi::Unknown(..))
    }
    pub fn is_divergent(&self) -> bool {
        self.is_divergent
    }
}
impl parse::SubItem for Function {
    fn parse(cur: clang::Cursor, ctx: &mut BindgenContext) -> Result<parse::Result<Self>, parse::Error> {
        use clang_lib::*;
        let kind = match FnKind::from_cursor(&cur) {
            None => return Err(parse::Error::Continue),
            Some(k) => k,
        };
        debug!("Function::parse({:?}, {:?})", cur, cur.cur_type());
        let visibility = cur.visibility();
        if visibility != CXVisibility_Default {
            return Err(parse::Error::Continue);
        }
        if cur.access_specifier() == CX_CXXPrivate {
            return Err(parse::Error::Continue);
        }
        let linkage = cur.linkage();
        let linkage = match linkage {
            CXLinkage_External | CXLinkage_UniqueExternal => Linkage::External,
            CXLinkage_Internal => Linkage::Internal,
            _ => return Err(parse::Error::Continue),
        };
        if cur.is_inlined_function() || cur.definition().map_or(false, |x| x.is_inlined_function()) {
            return Err(parse::Error::Continue);
        }
        let sig = Item::from_ty(&cur.cur_type(), cur, None, ctx)?;
        let mut name = cur.spelling();
        assert!(!name.is_empty(), "Empty function name?");
        if cur.kind() == CXCursor_Destructor {
            if name.starts_with('~') {
                name.remove(0);
            }
            name.push_str("_destructor");
        }
        if let Some(nm) = ctx.opts().last_callback(|callbacks| {
            callbacks.generated_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Function,
            })
        }) {
            name = nm;
        }
        assert!(!name.is_empty(), "Empty function name.");
        let mangled_name = cursor_mangling(ctx, &cur);
        let link_name = ctx.opts().last_callback(|callbacks| {
            callbacks.generated_link_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Function,
            })
        });
        let function = Self::new(name.clone(), mangled_name, link_name, sig, kind, linkage);
        Ok(parse::Result::New(function, Some(cur)))
    }
}
impl Trace for FnSig {
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
