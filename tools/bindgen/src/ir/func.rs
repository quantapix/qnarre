use super::{comp::MethKind, dot::DotAttrs, item::Item, typ::TypeKind, Context, EdgeKind, Trace, Tracer, TypeId};
use crate::{
    callbacks::{ItemInfo, ItemKind},
    clang::{self, Attribute},
    parse,
};
use clang_lib::{self, CXCallingConv};
use quote::TokenStreamExt;
use std::{io, str::FromStr};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FnKind {
    Func,
    Method(MethKind),
}
impl FnKind {
    pub fn from_cursor(cur: &clang::Cursor) -> Option<FnKind> {
        Some(match cur.kind() {
            clang_lib::CXCursor_FunctionDecl => FnKind::Func,
            clang_lib::CXCursor_Constructor => FnKind::Method(MethKind::Constr),
            clang_lib::CXCursor_Destructor => FnKind::Method(if cur.method_is_virt() {
                MethKind::VirtDestr {
                    pure: cur.method_is_pure_virt(),
                }
            } else {
                MethKind::Destr
            }),
            clang_lib::CXCursor_CXXMethod => {
                if cur.method_is_virt() {
                    FnKind::Method(MethKind::Virt {
                        pure: cur.method_is_pure_virt(),
                    })
                } else if cur.method_is_static() {
                    FnKind::Method(MethKind::Static)
                } else {
                    FnKind::Method(MethKind::Normal)
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
pub struct Func {
    name: String,
    mangled: Option<String>,
    link: Option<String>,
    sig: TypeId,
    kind: FnKind,
    linkage: Linkage,
}
impl Func {
    pub fn new(
        name: String,
        mangled: Option<String>,
        link: Option<String>,
        sig: TypeId,
        kind: FnKind,
        linkage: Linkage,
    ) -> Self {
        Func {
            name,
            mangled,
            link,
            sig,
            kind,
            linkage,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn mangled_name(&self) -> Option<&str> {
        self.mangled.as_deref()
    }
    pub fn link_name(&self) -> Option<&str> {
        self.link.as_deref()
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
impl DotAttrs for Func {
    fn dot_attrs<W>(&self, _ctx: &Context, y: &mut W) -> io::Result<()>
    where
        W: io::Write,
    {
        if let Some(ref x) = self.mangled {
            let x: String = x.chars().flat_map(|c| c.escape_default()).collect();
            writeln!(y, "<tr><td>mangled name</td><td>{}</td></tr>", x)?;
        }
        Ok(())
    }
}
impl parse::SubItem for Func {
    fn parse(cur: clang::Cursor, ctx: &mut Context) -> Result<parse::Resolved<Self>, parse::Error> {
        use clang_lib::*;
        let kind = match FnKind::from_cursor(&cur) {
            None => return Err(parse::Error::Continue),
            Some(x) => x,
        };
        if cur.visibility() != CXVisibility_Default {
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
        if cur.is_inlined_fn() || cur.definition().map_or(false, |x| x.is_inlined_fn()) {
            return Err(parse::Error::Continue);
        }
        let sig = Item::from_ty(&cur.cur_type(), cur, None, ctx)?;
        let mut name = cur.spelling();
        assert!(!name.is_empty());
        if cur.kind() == CXCursor_Destructor {
            if name.starts_with('~') {
                name.remove(0);
            }
            name.push_str("_destructor");
        }
        if let Some(x) = ctx.opts().last_callback(|x| {
            x.gen_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Func,
            })
        }) {
            name = x;
        }
        assert!(!name.is_empty());
        let mangled = cursor_mangling(ctx, &cur);
        let link = ctx.opts().last_callback(|x| {
            x.gen_link_name_override(ItemInfo {
                name: name.as_str(),
                kind: ItemKind::Func,
            })
        });
        let y = Self::new(name.clone(), mangled, link, sig, kind, linkage);
        Ok(parse::Resolved::New(y, Some(cur)))
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
impl quote::ToTokens for Abi {
    fn to_tokens(&self, toks: &mut proc_macro2::TokenStream) {
        let abi = self.to_string();
        toks.append_all(quote! { #abi });
    }
}
impl std::fmt::Display for Abi {
    fn fmt(&self, x: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        y.fmt(x)
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

const RUST_DERIVE_FUNPTR_LIMIT: usize = 12;

#[derive(Debug)]
pub struct FnSig {
    name: String,
    ret_type: TypeId,
    arg_types: Vec<(Option<String>, TypeId)>,
    is_variadic: bool,
    is_divergent: bool,
    must_use: bool,
    abi: ClangAbi,
}
impl FnSig {
    pub fn from_ty(ty: &clang::Type, cur: &clang::Cursor, ctx: &mut Context) -> Result<Self, parse::Error> {
        use clang_lib::*;
        let kind = cur.kind();
        if kind == CXCursor_FunctionTemplate {
            return Err(parse::Error::Continue);
        }
        let name = cur.spelling();
        let is_oper = |x: &str| x.starts_with("operator") && !clang::is_valid_ident(x);
        if is_oper(&name) {
            return Err(parse::Error::Continue);
        }
        if (kind == CXCursor_Constructor || kind == CXCursor_Destructor) && name.contains('<') {
            return Err(parse::Error::Continue);
        }
        let cur = if cur.is_valid() { *cur } else { ty.decl() };
        let mut arg_types = match kind {
            CXCursor_FunctionDecl | CXCursor_Constructor | CXCursor_CXXMethod => args_from_ty_and_cursor(ty, &cur, ctx),
            _ => {
                let mut ys = vec![];
                cur.visit(|x| {
                    if x.kind() == CXCursor_ParmDecl {
                        let ty = Item::from_ty_or_ref(x.cur_type(), x, None, ctx);
                        let name = x.spelling();
                        let name = if name.is_empty() { None } else { Some(name) };
                        ys.push((name, ty));
                    }
                    CXChildVisit_Continue
                });
                if ys.is_empty() {
                    args_from_ty_and_cursor(ty, &cur, ctx)
                } else {
                    ys
                }
            },
        };
        let (must_use, mut is_divergent) = if ctx.opts().enable_fn_attr_detection {
            let [must_use, no_return, no_return_cpp] =
                cur.has_attrs(&[Attribute::MUST_USE, Attribute::NO_RETURN, Attribute::NO_RETURN_CPP]);
            (must_use, no_return || no_return_cpp)
        } else {
            Default::default()
        };
        is_divergent = is_divergent || ty.spelling().contains("__attribute__((noreturn))");
        let is_method = kind == CXCursor_CXXMethod;
        let is_constr = kind == CXCursor_Constructor;
        let is_destr = kind == CXCursor_Destructor;
        if (is_constr || is_destr || is_method) && cur.lexical_parent() != cur.semantic_parent() {
            return Err(parse::Error::Continue);
        }
        if is_method || is_constr || is_destr {
            let is_const = is_method && cur.method_is_const();
            let is_virt = is_method && cur.method_is_virt();
            let is_static = is_method && cur.method_is_static();
            if !is_static && !is_virt {
                let parent = cur.semantic_parent();
                let class = Item::parse(parent, None, ctx).expect("Expected to parse the class");
                let class = class.as_type_id_unchecked();
                let class = if is_const {
                    let id = ctx.next_item_id();
                    ctx.build_const_wrapper(id, class, None, &parent.cur_type())
                } else {
                    class
                };
                let ptr = Item::builtin_type(TypeKind::Pointer(class), false, ctx);
                arg_types.insert(0, (Some("this".into()), ptr));
            } else if is_virt {
                let void = Item::builtin_type(TypeKind::Void, false, ctx);
                let ptr = Item::builtin_type(TypeKind::Pointer(void), false, ctx);
                arg_types.insert(0, (Some("this".into()), ptr));
            }
        }
        let ret_type = ty.ret_type().ok_or(parse::Error::Continue)?;
        let ret_type = if is_constr && ctx.is_target_wasm32() {
            let void = Item::builtin_type(TypeKind::Void, false, ctx);
            Item::builtin_type(TypeKind::Pointer(void), false, ctx)
        } else {
            Item::from_ty_or_ref(ret_type, cur, None, ctx)
        };
        let mut cc = ty.call_conv();
        if let Some(x) = cur.cur_type().canon_type().pointee_type() {
            let cc2 = x.call_conv();
            if cc2 != CXCallingConv_Invalid {
                cc = cc2;
            }
        }
        let abi = get_abi(cc);
        if abi.is_unknown() {
            warn!("Unknown calling convention: {:?}", cc);
        }
        Ok(Self {
            name,
            ret_type,
            arg_types,
            is_variadic: ty.is_variadic(),
            is_divergent,
            must_use,
            abi,
        })
    }
    pub fn ret_type(&self) -> TypeId {
        self.ret_type
    }
    pub fn arg_types(&self) -> &[(Option<String>, TypeId)] {
        &self.arg_types
    }
    pub fn abi(&self, ctx: &Context, name: Option<&str>) -> ClangAbi {
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
        self.is_variadic && !self.arg_types.is_empty()
    }
    pub fn must_use(&self) -> bool {
        self.must_use
    }
    pub fn fn_ptrs_can_derive(&self) -> bool {
        if self.arg_types.len() > RUST_DERIVE_FUNPTR_LIMIT {
            return false;
        }
        matches!(self.abi, ClangAbi::Known(Abi::C) | ClangAbi::Unknown(..))
    }
    pub fn is_divergent(&self) -> bool {
        self.is_divergent
    }
}
impl Trace for FnSig {
    type Extra = ();
    fn trace<T>(&self, _: &Context, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        tracer.visit_kind(self.ret_type().into(), EdgeKind::FnReturn);
        for &(_, ty) in self.arg_types() {
            tracer.visit_kind(ty.into(), EdgeKind::FnParameter);
        }
    }
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

pub fn cursor_mangling(ctx: &Context, cur: &clang::Cursor) -> Option<String> {
    if !ctx.opts().enable_mangling {
        return None;
    }
    if cur.is_in_non_fully_specialized_templ() {
        return None;
    }
    let is_destructor = cur.kind() == clang_lib::CXCursor_Destructor;
    if let Ok(mut xs) = cur.cxx_manglings() {
        while let Some(x) = xs.pop() {
            if is_destructor && !x.ends_with("D1Ev") {
                continue;
            }
            return Some(x);
        }
    }
    let mut y = cur.mangling();
    if y.is_empty() {
        return None;
    }
    if is_destructor && y.ends_with("D0Ev") {
        y.truncate(y.len() - 4);
        y.push_str("D1Ev");
    }
    Some(y)
}

fn args_from_ty_and_cursor(ty: &clang::Type, cur: &clang::Cursor, ctx: &mut Context) -> Vec<(Option<String>, TypeId)> {
    let ys = cur.args().unwrap_or_default().into_iter();
    let ty_args = ty.args().unwrap_or_default().into_iter();
    ys.map(Some)
        .chain(std::iter::repeat(None))
        .zip(ty_args.map(Some).chain(std::iter::repeat(None)))
        .take_while(|(cur, ty)| cur.is_some() || ty.is_some())
        .map(|(cur2, ty2)| {
            let name = cur2
                .map(|x| x.spelling())
                .and_then(|x| if x.is_empty() { None } else { Some(x) });
            let cur2 = cur2.unwrap_or(*cur);
            let ty2 = ty2.unwrap_or_else(|| cur2.cur_type());
            (name, Item::from_ty_or_ref(ty2, cur2, None, ctx))
        })
        .collect()
}
