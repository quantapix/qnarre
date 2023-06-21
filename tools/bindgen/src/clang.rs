use clang_lib::*;
use std::{
    ffi::{CStr, CString},
    fmt,
    hash::Hash,
    hash::Hasher,
    mem,
    os::raw::{c_char, c_int, c_longlong, c_uint, c_ulong, c_ulonglong},
    ptr, slice,
};

use crate::ir::Context;

pub struct Attribute {
    name: &'static [u8],
    kind: Option<CXCursorKind>,
    tok_kind: CXTokenKind,
}
impl Attribute {
    pub const MUST_USE: Self = Self {
        name: b"warn_unused_result",
        kind: Some(440),
        tok_kind: CXToken_Identifier,
    };
    pub const NO_RETURN: Self = Self {
        name: b"_Noreturn",
        kind: None,
        tok_kind: CXToken_Keyword,
    };
    pub const NO_RETURN_CPP: Self = Self {
        name: b"noreturn",
        kind: None,
        tok_kind: CXToken_Identifier,
    };
}

#[derive(Debug)]
pub struct MiscErr;
impl fmt::Display for MiscErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Misc error")
    }
}
impl std::error::Error for MiscErr {}

#[derive(Copy, Clone)]
pub struct Cursor {
    cur: CXCursor,
}
impl Cursor {
    pub fn usr(&self) -> Option<String> {
        let y = unsafe { cxstring_into_string(clang_getCursorUSR(self.cur)) };
        if y.is_empty() {
            None
        } else {
            Some(y)
        }
    }
    pub fn is_declaration(&self) -> bool {
        unsafe { clang_isDeclaration(self.kind()) != 0 }
    }
    pub fn is_anonymous(&self) -> bool {
        unsafe { clang_Cursor_isAnonymous(self.cur) != 0 }
    }
    pub fn spelling(&self) -> String {
        unsafe { cxstring_into_string(clang_getCursorSpelling(self.cur)) }
    }
    pub fn display_name(&self) -> String {
        unsafe { cxstring_into_string(clang_getCursorDisplayName(self.cur)) }
    }
    pub fn mangling(&self) -> String {
        unsafe { cxstring_into_string(clang_Cursor_getMangling(self.cur)) }
    }
    pub fn cxx_manglings(&self) -> Result<Vec<String>, MiscErr> {
        unsafe {
            let xs = clang_Cursor_getCXXManglings(self.cur);
            if xs.is_null() {
                return Err(MiscErr);
            }
            let count = (*xs).Count as usize;
            let mut ys = Vec::with_capacity(count);
            for i in 0..count {
                let string_ptr = (*xs).Strings.add(i);
                ys.push(cxstring_to_string_leaky(*string_ptr));
            }
            clang_disposeStringSet(xs);
            Ok(ys)
        }
    }
    pub fn is_builtin(&self) -> bool {
        let (file, _, _, _) = self.location().location();
        file.name().is_none()
    }
    pub fn lexical_parent(&self) -> Cursor {
        unsafe {
            Cursor {
                cur: clang_getCursorLexicalParent(self.cur),
            }
        }
    }
    pub fn fallible_semantic_parent(&self) -> Option<Cursor> {
        let y = unsafe {
            Cursor {
                cur: clang_getCursorSemanticParent(self.cur),
            }
        };
        if y == *self || !y.is_valid() {
            return None;
        }
        Some(y)
    }
    pub fn semantic_parent(&self) -> Cursor {
        self.fallible_semantic_parent().unwrap()
    }
    pub fn num_templ_args(&self) -> Option<u32> {
        self.cur_type()
            .num_templ_args()
            .or_else(|| {
                let y: c_int = unsafe { clang_Cursor_getNumTemplateArguments(self.cur) };
                if y >= 0 {
                    Some(y as u32)
                } else {
                    debug_assert_eq!(y, -1);
                    None
                }
            })
            .or_else(|| {
                let y = self.canonical();
                if y != *self {
                    y.num_templ_args()
                } else {
                    None
                }
            })
    }
    pub fn translation_unit(&self) -> Cursor {
        assert!(self.is_valid());
        unsafe {
            let tu = clang_Cursor_getTranslationUnit(self.cur);
            let y = Cursor {
                cur: clang_getTranslationUnitCursor(tu),
            };
            assert!(y.is_valid());
            y
        }
    }
    pub fn is_toplevel(&self) -> bool {
        let mut y = self.fallible_semantic_parent();
        while y.is_some()
            && (y.unwrap().kind() == CXCursor_Namespace
                || y.unwrap().kind() == CXCursor_NamespaceAlias
                || y.unwrap().kind() == CXCursor_NamespaceRef)
        {
            y = y.unwrap().fallible_semantic_parent();
        }
        let tu = self.translation_unit();
        y == tu.fallible_semantic_parent()
    }
    pub fn is_templ_like(&self) -> bool {
        matches!(
            self.kind(),
            CXCursor_ClassTemplate | CXCursor_ClassTemplatePartialSpecialization | CXCursor_TypeAliasTemplateDecl
        )
    }
    pub fn is_macro_fn_like(&self) -> bool {
        unsafe { clang_Cursor_isMacroFunctionLike(self.cur) != 0 }
    }
    pub fn kind(&self) -> CXCursorKind {
        self.cur.kind
    }
    pub fn is_definition(&self) -> bool {
        unsafe { clang_isCursorDefinition(self.cur) != 0 }
    }
    pub fn is_templ_specialization(&self) -> bool {
        self.specialized().is_some()
    }
    pub fn is_fully_specialized_templ(&self) -> bool {
        self.is_templ_specialization()
            && self.kind() != CXCursor_ClassTemplatePartialSpecialization
            && self.num_templ_args().unwrap_or(0) > 0
    }
    pub fn is_in_non_fully_specialized_templ(&self) -> bool {
        if self.is_toplevel() {
            return false;
        }
        let y = self.semantic_parent();
        if y.is_fully_specialized_templ() {
            return false;
        }
        if !y.is_templ_like() {
            return y.is_in_non_fully_specialized_templ();
        }
        true
    }
    pub fn is_templ_param(&self) -> bool {
        matches!(
            self.kind(),
            CXCursor_TemplateTemplateParameter | CXCursor_TemplateTypeParameter | CXCursor_NonTypeTemplateParameter
        )
    }
    pub fn is_dependent_on_templ_param(&self) -> bool {
        fn visitor(y: &mut bool, cur: Cursor) -> CXChildVisitResult {
            if cur.is_templ_param() {
                *y = true;
                return CXChildVisit_Break;
            }
            if let Some(x) = cur.referenced() {
                if x.is_templ_param() {
                    *y = true;
                    return CXChildVisit_Break;
                }
                x.visit(|x| visitor(y, x));
                if *y {
                    return CXChildVisit_Break;
                }
            }
            CXChildVisit_Recurse
        }
        if self.is_templ_param() {
            return true;
        }
        let mut y = false;
        self.visit(|x| visitor(&mut y, x));
        y
    }
    pub fn is_valid(&self) -> bool {
        unsafe { clang_isInvalid(self.kind()) == 0 }
    }
    pub fn location(&self) -> SrcLoc {
        unsafe {
            SrcLoc {
                x: clang_getCursorLocation(self.cur),
            }
        }
    }
    pub fn extent(&self) -> CXSourceRange {
        unsafe { clang_getCursorExtent(self.cur) }
    }
    pub fn raw_comment(&self) -> Option<String> {
        let y = unsafe { cxstring_into_string(clang_Cursor_getRawCommentText(self.cur)) };
        if y.is_empty() {
            None
        } else {
            Some(y)
        }
    }
    pub fn comment(&self) -> Comment {
        unsafe {
            Comment {
                comm: clang_Cursor_getParsedComment(self.cur),
            }
        }
    }
    pub fn cur_type(&self) -> Type {
        unsafe {
            Type {
                ty: clang_getCursorType(self.cur),
            }
        }
    }
    pub fn definition(&self) -> Option<Cursor> {
        unsafe {
            let y = Cursor {
                cur: clang_getCursorDefinition(self.cur),
            };
            if y.is_valid() && y.kind() != CXCursor_NoDeclFound {
                Some(y)
            } else {
                None
            }
        }
    }
    pub fn referenced(&self) -> Option<Cursor> {
        unsafe {
            let y = Cursor {
                cur: clang_getCursorReferenced(self.cur),
            };
            if y.is_valid() {
                Some(y)
            } else {
                None
            }
        }
    }
    pub fn canonical(&self) -> Cursor {
        unsafe {
            Cursor {
                cur: clang_getCanonicalCursor(self.cur),
            }
        }
    }
    pub fn specialized(&self) -> Option<Cursor> {
        unsafe {
            let y = Cursor {
                cur: clang_getSpecializedCursorTemplate(self.cur),
            };
            if y.is_valid() {
                Some(y)
            } else {
                None
            }
        }
    }
    pub fn templ_kind(&self) -> CXCursorKind {
        unsafe { clang_getTemplateCursorKind(self.cur) }
    }
    pub fn visit<Visitor>(&self, mut v: Visitor)
    where
        Visitor: FnMut(Cursor) -> CXChildVisitResult,
    {
        let v = &mut v as *mut Visitor;
        unsafe {
            clang_visitChildren(self.cur, visit_children::<Visitor>, v.cast());
        }
    }
    pub fn collect_children(&self) -> Vec<Cursor> {
        let mut ys = vec![];
        self.visit(|x| {
            ys.push(x);
            CXChildVisit_Continue
        });
        ys
    }
    pub fn has_children(&self) -> bool {
        let mut y = false;
        self.visit(|_| {
            y = true;
            CXChildVisit_Break
        });
        y
    }
    pub fn has_at_least_num_children(&self, n: usize) -> bool {
        assert!(n > 0);
        let mut y = n;
        self.visit(|_| {
            y -= 1;
            if y == 0 {
                CXChildVisit_Break
            } else {
                CXChildVisit_Continue
            }
        });
        y == 0
    }
    pub fn contains_cursor(&self, kind: CXCursorKind) -> bool {
        let mut y = false;
        self.visit(|x| {
            if x.kind() == kind {
                y = true;
                CXChildVisit_Break
            } else {
                CXChildVisit_Continue
            }
        });
        y
    }
    pub fn is_inlined_fn(&self) -> bool {
        unsafe { clang_Cursor_isFunctionInlined(self.cur) != 0 }
    }
    pub fn is_defaulted_fn(&self) -> bool {
        unsafe { clang_CXXMethod_isDefaulted(self.cur) != 0 }
    }
    pub fn is_deleted_fn(&self) -> bool {
        self.is_inlined_fn() && self.definition().is_none() && !self.is_defaulted_fn()
    }
    pub fn is_bit_field(&self) -> bool {
        unsafe { clang_Cursor_isBitField(self.cur) != 0 }
    }
    pub fn bit_width_expr(&self) -> Option<Cursor> {
        if !self.is_bit_field() {
            return None;
        }
        let mut y = None;
        self.visit(|x| {
            if x.kind() == CXCursor_TypeRef {
                return CXChildVisit_Continue;
            }
            y = Some(x);
            CXChildVisit_Break
        });
        y
    }
    pub fn bit_width(&self) -> Option<u32> {
        if self.bit_width_expr()?.is_dependent_on_templ_param() {
            return None;
        }
        unsafe {
            let y = clang_getFieldDeclBitWidth(self.cur);
            if y == -1 {
                None
            } else {
                Some(y as u32)
            }
        }
    }
    pub fn enum_type(&self) -> Option<Type> {
        unsafe {
            let y = Type {
                ty: clang_getEnumDeclIntegerType(self.cur),
            };
            if y.is_valid() {
                Some(y)
            } else {
                None
            }
        }
    }
    pub fn enum_val_bool(&self) -> Option<bool> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                Some(clang_getEnumConstantDeclValue(self.cur) != 0)
            } else {
                None
            }
        }
    }
    pub fn enum_val_signed(&self) -> Option<i64> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                #[allow(clippy::unnecessary_cast)]
                Some(clang_getEnumConstantDeclValue(self.cur) as i64)
            } else {
                None
            }
        }
    }
    pub fn enum_val_unsigned(&self) -> Option<u64> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                #[allow(clippy::unnecessary_cast)]
                Some(clang_getEnumConstantDeclUnsignedValue(self.cur) as u64)
            } else {
                None
            }
        }
    }
    pub fn has_attrs<const N: usize>(&self, attrs: &[Attribute; N]) -> [bool; N] {
        let mut ys = [false; N];
        let mut count = 0;
        self.visit(|x| {
            let kind = x.kind();
            for (i, attr) in attrs.iter().enumerate() {
                let y = &mut ys[i];
                if !*y
                    && (attr.kind.map_or(false, |x| x == kind)
                        || (kind == CXCursor_UnexposedAttr
                            && x.toks()
                                .iter()
                                .any(|x| x.kind == attr.tok_kind && x.spelling() == attr.name)))
                {
                    *y = true;
                    count += 1;
                    if count == N {
                        return CXChildVisit_Break;
                    }
                }
            }
            CXChildVisit_Continue
        });
        ys
    }
    pub fn typedef_type(&self) -> Option<Type> {
        let y = Type {
            ty: unsafe { clang_getTypedefDeclUnderlyingType(self.cur) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn linkage(&self) -> CXLinkageKind {
        unsafe { clang_getCursorLinkage(self.cur) }
    }
    pub fn visibility(&self) -> CXVisibilityKind {
        unsafe { clang_getCursorVisibility(self.cur) }
    }
    pub fn args(&self) -> Option<Vec<Cursor>> {
        self.num_args().ok().map(|x| {
            (0..x)
                .map(|i| Cursor {
                    cur: unsafe { clang_Cursor_getArgument(self.cur, i as c_uint) },
                })
                .collect()
        })
    }
    pub fn num_args(&self) -> Result<u32, MiscErr> {
        unsafe {
            let y = clang_Cursor_getNumArguments(self.cur);
            if y == -1 {
                Err(MiscErr)
            } else {
                Ok(y as u32)
            }
        }
    }
    pub fn access_specifier(&self) -> CX_CXXAccessSpecifier {
        unsafe { clang_getCXXAccessSpecifier(self.cur) }
    }
    pub fn public_accessible(&self) -> bool {
        let y = self.access_specifier();
        y == CX_CXXPublic || y == CX_CXXInvalidAccessSpecifier
    }
    pub fn is_mut_field(&self) -> bool {
        unsafe { clang_CXXField_isMutable(self.cur) != 0 }
    }
    pub fn offset_of_field(&self) -> Result<usize, LayoutErr> {
        let y = unsafe { clang_Cursor_getOffsetOfField(self.cur) };
        if y < 0 {
            Err(LayoutErr::from(y as i32))
        } else {
            Ok(y as usize)
        }
    }
    pub fn method_is_static(&self) -> bool {
        unsafe { clang_CXXMethod_isStatic(self.cur) != 0 }
    }
    pub fn method_is_const(&self) -> bool {
        unsafe { clang_CXXMethod_isConst(self.cur) != 0 }
    }
    pub fn method_is_virt(&self) -> bool {
        unsafe { clang_CXXMethod_isVirtual(self.cur) != 0 }
    }
    pub fn method_is_pure_virt(&self) -> bool {
        unsafe { clang_CXXMethod_isPureVirtual(self.cur) != 0 }
    }
    pub fn is_virt_base(&self) -> bool {
        unsafe { clang_isVirtualBase(self.cur) != 0 }
    }
    pub fn evaluate(&self) -> Option<EvalResult> {
        EvalResult::new(*self)
    }
    pub fn ret_type(&self) -> Option<Type> {
        let y = Type {
            ty: unsafe { clang_getCursorResultType(self.cur) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn toks(&self) -> RawToks {
        RawToks::new(self)
    }
    pub fn cexpr_toks(self) -> Vec<cexpr::token::Token> {
        self.toks().iter().filter_map(|x| x.as_cexpr_token()).collect()
    }
    pub fn get_included_file_name(&self) -> Option<String> {
        let y = unsafe { clang_getIncludedFile(self.cur) };
        if y.is_null() {
            None
        } else {
            Some(unsafe { cxstring_into_string(clang_getFileName(y)) })
        }
    }
}
impl PartialEq for Cursor {
    fn eq(&self, x: &Cursor) -> bool {
        unsafe { clang_equalCursors(self.cur, x.cur) == 1 }
    }
}
impl Eq for Cursor {}
impl Hash for Cursor {
    fn hash<H: Hasher>(&self, x: &mut H) {
        unsafe { clang_hashCursor(self.cur) }.hash(x)
    }
}
impl fmt::Debug for Cursor {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(
            x,
            "Cursor({} kind: {}, loc: {}, usr: {:?})",
            self.spelling(),
            kind_to_str(self.kind()),
            self.location(),
            self.usr()
        )
    }
}

pub struct RawToks<'a> {
    cur: &'a Cursor,
    tu: CXTranslationUnit,
    toks: *mut CXToken,
    count: c_uint,
}
impl<'a> RawToks<'a> {
    fn new(cur: &'a Cursor) -> Self {
        let mut toks = ptr::null_mut();
        let mut count = 0;
        let range = cur.extent();
        let tu = unsafe { clang_Cursor_getTranslationUnit(cur.cur) };
        unsafe { clang_tokenize(tu, range, &mut toks, &mut count) };
        Self { cur, tu, toks, count }
    }
    fn as_slice(&self) -> &[CXToken] {
        if self.toks.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.toks, self.count as usize) }
    }
    pub fn iter(&self) -> TokIter {
        TokIter {
            tu: self.tu,
            raw: self.as_slice().iter(),
        }
    }
}
impl<'a> Drop for RawToks<'a> {
    fn drop(&mut self) {
        if !self.toks.is_null() {
            unsafe {
                clang_disposeTokens(self.tu, self.toks, self.count as c_uint);
            }
        }
    }
}

#[derive(Debug)]
pub struct Token {
    spelling: CXString,
    pub ext: CXSourceRange,
    pub kind: CXTokenKind,
}
impl Token {
    pub fn spelling(&self) -> &[u8] {
        let y = unsafe { CStr::from_ptr(clang_getCString(self.spelling) as *const _) };
        y.to_bytes()
    }
    pub fn as_cexpr_token(&self) -> Option<cexpr::token::Token> {
        use cexpr::token;
        let kind = match self.kind {
            CXToken_Punctuation => token::Kind::Punctuation,
            CXToken_Literal => token::Kind::Literal,
            CXToken_Identifier => token::Kind::Identifier,
            CXToken_Keyword => token::Kind::Keyword,
            CXToken_Comment => return None,
            _ => {
                warn!("Found unexpected token kind: {:?}", self);
                return None;
            },
        };
        Some(token::Token {
            kind,
            raw: self.spelling().to_vec().into_boxed_slice(),
        })
    }
}
impl Drop for Token {
    fn drop(&mut self) {
        unsafe { clang_disposeString(self.spelling) }
    }
}

pub struct TokIter<'a> {
    tu: CXTranslationUnit,
    raw: slice::Iter<'a, CXToken>,
}
impl<'a> Iterator for TokIter<'a> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        let y = self.raw.next()?;
        unsafe {
            let kind = clang_getTokenKind(*y);
            let spelling = clang_getTokenSpelling(self.tu, *y);
            let ext = clang_getTokenExtent(self.tu, *y);
            Some(Token { kind, ext, spelling })
        }
    }
}

#[derive(Clone, Copy)]
pub struct Type {
    ty: CXType,
}
impl Type {
    pub fn kind(&self) -> CXTypeKind {
        self.ty.kind
    }
    pub fn decl(&self) -> Cursor {
        unsafe {
            Cursor {
                cur: clang_getTypeDeclaration(self.ty),
            }
        }
    }
    pub fn canon_decl(&self, x: Option<&Cursor>) -> Option<CanonTyDecl> {
        let mut y = self.decl();
        if !y.is_valid() {
            if let Some(x) = x {
                let mut loc = *x;
                if let Some(x) = loc.referenced() {
                    loc = x;
                }
                if loc.is_templ_like() {
                    y = loc;
                }
            }
        }
        let y = y.canonical();
        if y.is_valid() && y.kind() != CXCursor_NoDeclFound {
            Some(CanonTyDecl(*self, y))
        } else {
            None
        }
    }
    pub fn spelling(&self) -> String {
        let y = unsafe { cxstring_into_string(clang_getTypeSpelling(self.ty)) };
        if y.split("::").all(is_valid_ident) {
            if let Some(y) = y.split("::").last() {
                return y.to_owned();
            }
        }
        y
    }
    pub fn is_const(&self) -> bool {
        unsafe { clang_isConstQualifiedType(self.ty) != 0 }
    }
    #[inline]
    fn is_non_deductible_auto_type(&self) -> bool {
        debug_assert_eq!(self.kind(), CXType_Auto);
        self.canon_type() == *self
    }
    #[inline]
    fn clang_size_of(&self, ctx: &Context) -> c_longlong {
        match self.kind() {
            CXType_RValueReference | CXType_LValueReference => ctx.target_ptr_size() as c_longlong,
            CXType_Auto if self.is_non_deductible_auto_type() => -6,
            _ => unsafe { clang_Type_getSizeOf(self.ty) },
        }
    }
    #[inline]
    fn clang_align_of(&self, ctx: &Context) -> c_longlong {
        match self.kind() {
            CXType_RValueReference | CXType_LValueReference => ctx.target_ptr_size() as c_longlong,
            CXType_Auto if self.is_non_deductible_auto_type() => -6,
            _ => unsafe { clang_Type_getAlignOf(self.ty) },
        }
    }
    pub fn size(&self, ctx: &Context) -> usize {
        let y = self.clang_size_of(ctx);
        if y < 0 {
            0
        } else {
            y as usize
        }
    }
    pub fn fallible_size(&self, ctx: &Context) -> Result<usize, LayoutErr> {
        let y = self.clang_size_of(ctx);
        if y < 0 {
            Err(LayoutErr::from(y as i32))
        } else {
            Ok(y as usize)
        }
    }
    pub fn align(&self, ctx: &Context) -> usize {
        let y = self.clang_align_of(ctx);
        if y < 0 {
            0
        } else {
            y as usize
        }
    }
    pub fn fallible_align(&self, ctx: &Context) -> Result<usize, LayoutErr> {
        let y = self.clang_align_of(ctx);
        if y < 0 {
            Err(LayoutErr::from(y as i32))
        } else {
            Ok(y as usize)
        }
    }
    pub fn fallible_layout(&self, ctx: &Context) -> Result<crate::ir::Layout, LayoutErr> {
        use crate::ir::Layout;
        let size = self.fallible_size(ctx)?;
        let align = self.fallible_align(ctx)?;
        Ok(Layout::new(size, align))
    }
    pub fn num_templ_args(&self) -> Option<u32> {
        let y = unsafe { clang_Type_getNumTemplateArguments(self.ty) };
        if y >= 0 {
            Some(y as u32)
        } else {
            debug_assert_eq!(y, -1);
            None
        }
    }
    pub fn templ_args(&self) -> Option<TemplArgIter> {
        self.num_templ_args().map(|length| TemplArgIter {
            x: self.ty,
            length,
            idx: 0,
        })
    }
    pub fn args(&self) -> Option<Vec<Type>> {
        self.num_args().ok().map(|x| {
            (0..x)
                .map(|x| Type {
                    ty: unsafe { clang_getArgType(self.ty, x as c_uint) },
                })
                .collect()
        })
    }
    pub fn num_args(&self) -> Result<u32, MiscErr> {
        unsafe {
            let y = clang_getNumArgTypes(self.ty);
            if y == -1 {
                Err(MiscErr)
            } else {
                Ok(y as u32)
            }
        }
    }
    pub fn pointee_type(&self) -> Option<Type> {
        match self.kind() {
            CXType_Pointer
            | CXType_RValueReference
            | CXType_LValueReference
            | CXType_MemberPointer
            | CXType_BlockPointer => {
                let y = Type {
                    ty: unsafe { clang_getPointeeType(self.ty) },
                };
                debug_assert!(y.is_valid());
                Some(y)
            },
            _ => None,
        }
    }
    pub fn elem_type(&self) -> Option<Type> {
        let y = Type {
            ty: unsafe { clang_getElementType(self.ty) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn num_elems(&self) -> Option<usize> {
        let y = unsafe { clang_getNumElements(self.ty) };
        if y != -1 {
            Some(y as usize)
        } else {
            None
        }
    }
    pub fn canon_type(&self) -> Type {
        unsafe {
            Type {
                ty: clang_getCanonicalType(self.ty),
            }
        }
    }
    pub fn is_variadic(&self) -> bool {
        unsafe { clang_isFunctionTypeVariadic(self.ty) != 0 }
    }
    pub fn ret_type(&self) -> Option<Type> {
        let y = Type {
            ty: unsafe { clang_getResultType(self.ty) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn call_conv(&self) -> CXCallingConv {
        unsafe { clang_getFunctionTypeCallingConv(self.ty) }
    }
    pub fn named(&self) -> Type {
        unsafe {
            Type {
                ty: clang_Type_getNamedType(self.ty),
            }
        }
    }
    pub fn is_valid(&self) -> bool {
        self.kind() != CXType_Invalid
    }
    pub fn is_valid_and_exposed(&self) -> bool {
        self.is_valid() && self.kind() != CXType_Unexposed
    }
    pub fn is_fully_inst_templ(&self) -> bool {
        self.templ_args().map_or(false, |x| x.len() > 0)
            && !matches!(
                self.decl().kind(),
                CXCursor_ClassTemplatePartialSpecialization
                    | CXCursor_TypeAliasTemplateDecl
                    | CXCursor_TemplateTemplateParameter
            )
    }
    pub fn is_associated_type(&self) -> bool {
        fn hacky_parse_associated_type<S: AsRef<str>>(x: S) -> bool {
            lazy_static! {
                static ref ASSOC_TYPE_RE: regex::Regex =
                    regex::Regex::new(r"typename type\-parameter\-\d+\-\d+::.+").unwrap();
            }
            ASSOC_TYPE_RE.is_match(x.as_ref())
        }
        self.kind() == CXType_Unexposed
            && (hacky_parse_associated_type(self.spelling())
                || hacky_parse_associated_type(self.canon_type().spelling()))
    }
}
impl PartialEq for Type {
    fn eq(&self, x: &Self) -> bool {
        unsafe { clang_equalTypes(self.ty, x.ty) != 0 }
    }
}
impl Eq for Type {}
impl fmt::Debug for Type {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(
            x,
            "Type({}, kind: {}, cconv: {}, decl: {:?}, canon: {:?})",
            self.spelling(),
            type_to_str(self.kind()),
            self.call_conv(),
            self.decl(),
            self.decl().canonical()
        )
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum LayoutErr {
    Invalid,
    Incomplete,
    Dependent,
    NotConstantSize,
    InvalidFieldName,
    Unknown,
}
impl ::std::convert::From<i32> for LayoutErr {
    fn from(x: i32) -> Self {
        use self::LayoutErr::*;
        match x {
            CXTypeLayoutError_Invalid => Invalid,
            CXTypeLayoutError_Incomplete => Incomplete,
            CXTypeLayoutError_Dependent => Dependent,
            CXTypeLayoutError_NotConstantSize => NotConstantSize,
            CXTypeLayoutError_InvalidFieldName => InvalidFieldName,
            _ => Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonTyDecl(Type, Cursor);
impl CanonTyDecl {
    pub fn ty(&self) -> &Type {
        &self.0
    }
    pub fn cursor(&self) -> &Cursor {
        &self.1
    }
}

pub struct TemplArgIter {
    x: CXType,
    length: u32,
    idx: u32,
}
impl Iterator for TemplArgIter {
    type Item = Type;
    fn next(&mut self) -> Option<Type> {
        if self.idx < self.length {
            let i = self.idx as c_uint;
            self.idx += 1;
            Some(Type {
                ty: unsafe { clang_Type_getTemplateArgumentAsType(self.x, i) },
            })
        } else {
            None
        }
    }
}
impl ExactSizeIterator for TemplArgIter {
    fn len(&self) -> usize {
        assert!(self.idx <= self.length);
        (self.length - self.idx) as usize
    }
}

pub struct SrcLoc {
    x: CXSourceLocation,
}
impl SrcLoc {
    pub fn location(&self) -> (File, usize, usize, usize) {
        unsafe {
            let mut file = mem::zeroed();
            let mut line = 0;
            let mut col = 0;
            let mut off = 0;
            clang_getSpellingLocation(self.x, &mut file, &mut line, &mut col, &mut off);
            (File { file }, line as usize, col as usize, off as usize)
        }
    }
}
impl fmt::Display for SrcLoc {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        let (file, line, col, _) = self.location();
        if let Some(name) = file.name() {
            write!(x, "{}:{}:{}", name, line, col)
        } else {
            "builtin definitions".fmt(x)
        }
    }
}
impl fmt::Debug for SrcLoc {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(x, "{}", self)
    }
}

pub struct Comment {
    comm: CXComment,
}
impl Comment {
    pub fn kind(&self) -> CXCommentKind {
        unsafe { clang_Comment_getKind(self.comm) }
    }
    pub fn get_children(&self) -> CommChildrenIter {
        CommChildrenIter {
            parent: self.comm,
            length: unsafe { clang_Comment_getNumChildren(self.comm) },
            idx: 0,
        }
    }
    pub fn get_tag_name(&self) -> String {
        unsafe { cxstring_into_string(clang_HTMLTagComment_getTagName(self.comm)) }
    }
    pub fn get_tag_attrs(&self) -> CommAttrsIter {
        CommAttrsIter {
            x: self.comm,
            length: unsafe { clang_HTMLStartTag_getNumAttrs(self.comm) },
            idx: 0,
        }
    }
}

pub struct CommChildrenIter {
    parent: CXComment,
    length: c_uint,
    idx: c_uint,
}
impl Iterator for CommChildrenIter {
    type Item = Comment;
    fn next(&mut self) -> Option<Comment> {
        if self.idx < self.length {
            let i = self.idx;
            self.idx += 1;
            Some(Comment {
                comm: unsafe { clang_Comment_getChild(self.parent, i) },
            })
        } else {
            None
        }
    }
}

pub struct CommAttr {
    pub name: String,
    pub value: String,
}
pub struct CommAttrsIter {
    x: CXComment,
    length: c_uint,
    idx: c_uint,
}
impl Iterator for CommAttrsIter {
    type Item = CommAttr;
    fn next(&mut self) -> Option<CommAttr> {
        if self.idx < self.length {
            let idx = self.idx;
            self.idx += 1;
            Some(CommAttr {
                name: unsafe { cxstring_into_string(clang_HTMLStartTag_getAttrName(self.x, idx)) },
                value: unsafe { cxstring_into_string(clang_HTMLStartTag_getAttrValue(self.x, idx)) },
            })
        } else {
            None
        }
    }
}

pub struct File {
    file: CXFile,
}
impl File {
    pub fn name(&self) -> Option<String> {
        if self.file.is_null() {
            return None;
        }
        Some(unsafe { cxstring_into_string(clang_getFileName(self.file)) })
    }
}

pub struct Index {
    idx: CXIndex,
}
impl Index {
    pub fn new(pch: bool, diag: bool) -> Index {
        unsafe {
            Index {
                idx: clang_createIndex(pch as c_int, diag as c_int),
            }
        }
    }
}
impl fmt::Debug for Index {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(x, "Index {{ }}")
    }
}
impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            clang_disposeIndex(self.idx);
        }
    }
}

pub struct TranslationUnit {
    tu: CXTranslationUnit,
}
impl fmt::Debug for TranslationUnit {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(x, "TranslationUnit {{ }}")
    }
}
impl TranslationUnit {
    pub fn parse(
        ix: &Index,
        file: &str,
        cmd_args: &[String],
        unsaved: &[UnsavedFile],
        opts: CXTranslationUnit_Flags,
    ) -> Option<TranslationUnit> {
        let fname = CString::new(file).unwrap();
        let _c_args: Vec<CString> = cmd_args.iter().map(|s| CString::new(s.clone()).unwrap()).collect();
        let c_args: Vec<*const c_char> = _c_args.iter().map(|s| s.as_ptr()).collect();
        let mut c_unsaved: Vec<CXUnsavedFile> = unsaved.iter().map(|f| f.file).collect();
        let tu = unsafe {
            clang_parseTranslationUnit(
                ix.idx,
                fname.as_ptr(),
                c_args.as_ptr(),
                c_args.len() as c_int,
                c_unsaved.as_mut_ptr(),
                c_unsaved.len() as c_uint,
                opts,
            )
        };
        if tu.is_null() {
            None
        } else {
            Some(TranslationUnit { tu })
        }
    }
    pub fn diags(&self) -> Vec<Diagnostic> {
        unsafe {
            let x = clang_getNumDiagnostics(self.tu) as usize;
            let mut ys = vec![];
            for i in 0..x {
                ys.push(Diagnostic {
                    diag: clang_getDiagnostic(self.tu, i as c_uint),
                });
            }
            ys
        }
    }
    pub fn cursor(&self) -> Cursor {
        unsafe {
            Cursor {
                cur: clang_getTranslationUnitCursor(self.tu),
            }
        }
    }
    pub fn is_null(&self) -> bool {
        self.tu.is_null()
    }
}
impl Drop for TranslationUnit {
    fn drop(&mut self) {
        unsafe {
            clang_disposeTranslationUnit(self.tu);
        }
    }
}

pub struct Diagnostic {
    diag: CXDiagnostic,
}
impl Diagnostic {
    pub fn format(&self) -> String {
        unsafe {
            let opts = clang_defaultDiagnosticDisplayOptions();
            cxstring_into_string(clang_formatDiagnostic(self.diag, opts))
        }
    }
    pub fn severity(&self) -> CXDiagnosticSeverity {
        unsafe { clang_getDiagnosticSeverity(self.diag) }
    }
}
impl Drop for Diagnostic {
    fn drop(&mut self) {
        unsafe {
            clang_disposeDiagnostic(self.diag);
        }
    }
}

pub struct UnsavedFile {
    file: CXUnsavedFile,
    pub name: CString,
    cont: CString,
}
impl UnsavedFile {
    pub fn new(name: String, cont: String) -> UnsavedFile {
        let name = CString::new(name).unwrap();
        let cont = CString::new(cont).unwrap();
        let file = CXUnsavedFile {
            Filename: name.as_ptr(),
            Contents: cont.as_ptr(),
            Length: cont.as_bytes().len() as c_ulong,
        };
        UnsavedFile { file, name, cont }
    }
}
impl fmt::Debug for UnsavedFile {
    fn fmt(&self, x: &mut fmt::Formatter) -> fmt::Result {
        write!(x, "UnsavedFile(name: {:?}, contents: {:?})", self.name, self.cont)
    }
}

#[derive(Debug)]
pub struct EvalResult {
    x: CXEvalResult,
    ty: Type,
}
impl EvalResult {
    pub fn new(cur: Cursor) -> Option<Self> {
        {
            let mut cant_eval = false;
            cur.visit(|x| {
                if x.kind() == CXCursor_TypeRef && x.cur_type().canon_type().kind() == CXType_Unexposed {
                    cant_eval = true;
                    return CXChildVisit_Break;
                }
                CXChildVisit_Recurse
            });
            if cant_eval {
                return None;
            }
        }
        Some(EvalResult {
            x: unsafe { clang_Cursor_Evaluate(cur.cur) },
            ty: cur.cur_type().canon_type(),
        })
    }
    fn kind(&self) -> CXEvalResultKind {
        unsafe { clang_EvalResult_getKind(self.x) }
    }
    pub fn as_double(&self) -> Option<f64> {
        match self.kind() {
            CXEval_Float => Some(unsafe { clang_EvalResult_getAsDouble(self.x) }),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        if self.kind() != CXEval_Int {
            return None;
        }
        if unsafe { clang_EvalResult_isUnsignedInt(self.x) } != 0 {
            let value = unsafe { clang_EvalResult_getAsUnsigned(self.x) };
            if value > i64::max_value() as c_ulonglong {
                return None;
            }
            return Some(value as i64);
        }
        let value = unsafe { clang_EvalResult_getAsLongLong(self.x) };
        if value > i64::max_value() as c_longlong {
            return None;
        }
        if value < i64::min_value() as c_longlong {
            return None;
        }
        Some(value)
    }
    pub fn as_literal_string(&self) -> Option<Vec<u8>> {
        if self.kind() != CXEval_StrLiteral {
            return None;
        }
        let char_ty = self.ty.pointee_type().or_else(|| self.ty.elem_type())?;
        match char_ty.kind() {
            CXType_Char_S | CXType_SChar | CXType_Char_U | CXType_UChar => {
                let ret = unsafe { CStr::from_ptr(clang_EvalResult_getAsStr(self.x)) };
                Some(ret.to_bytes().to_vec())
            },
            CXType_Char16 => None,
            CXType_Char32 => None,
            CXType_WChar => None,
            _ => None,
        }
    }
}
impl Drop for EvalResult {
    fn drop(&mut self) {
        unsafe { clang_EvalResult_dispose(self.x) };
    }
}

#[derive(Debug)]
pub struct Target {
    pub triple: String,
    pub ptr_size: usize,
}
impl Target {
    pub fn new(tu: &TranslationUnit) -> Self {
        let triple;
        let ptr_size;
        unsafe {
            let ti = clang_getTranslationUnitTargetInfo(tu.tu);
            triple = cxstring_into_string(clang_TargetInfo_getTriple(ti));
            ptr_size = clang_TargetInfo_getPointerWidth(ti);
            clang_TargetInfo_dispose(ti);
        }
        assert!(ptr_size > 0);
        assert_eq!(ptr_size % 8, 0);
        Target {
            triple,
            ptr_size: ptr_size as usize,
        }
    }
}

pub fn is_valid_ident(x: &str) -> bool {
    let mut ys = x.chars();
    let first_valid = ys.next().map(|x| x.is_alphabetic() || x == '_').unwrap_or(false);
    first_valid && ys.all(|x| x.is_alphanumeric() || x == '_')
}
extern "C" fn visit_children<Visitor>(cur: CXCursor, _parent: CXCursor, data: CXClientData) -> CXChildVisitResult
where
    Visitor: FnMut(Cursor) -> CXChildVisitResult,
{
    let v: &mut Visitor = unsafe { &mut *(data as *mut Visitor) };
    let child = Cursor { cur };
    (*v)(child)
}

pub fn kind_to_str(x: CXCursorKind) -> String {
    unsafe { cxstring_into_string(clang_getCursorKindSpelling(x)) }
}
pub fn type_to_str(x: CXTypeKind) -> String {
    unsafe { cxstring_into_string(clang_getTypeKindSpelling(x)) }
}
pub fn ast_dump(c: &Cursor, depth: isize) -> CXChildVisitResult {
    fn print_indent<S: AsRef<str>>(depth: isize, s: S) {
        for _ in 0..depth {
            print!("    ");
        }
        println!("{}", s.as_ref());
    }
    fn print_cursor<S: AsRef<str>>(depth: isize, prefix: S, c: &Cursor) {
        let prefix = prefix.as_ref();
        print_indent(depth, format!(" {}kind = {}", prefix, kind_to_str(c.kind())));
        print_indent(depth, format!(" {}spelling = \"{}\"", prefix, c.spelling()));
        print_indent(depth, format!(" {}location = {}", prefix, c.location()));
        print_indent(depth, format!(" {}is-definition? {}", prefix, c.is_definition()));
        print_indent(depth, format!(" {}is-declaration? {}", prefix, c.is_declaration()));
        print_indent(depth, format!(" {}is-inlined-function? {}", prefix, c.is_inlined_fn()));
        let templ_kind = c.templ_kind();
        if templ_kind != CXCursor_NoDeclFound {
            print_indent(depth, format!(" {}templ-kind = {}", prefix, kind_to_str(templ_kind)));
        }
        if let Some(usr) = c.usr() {
            print_indent(depth, format!(" {}usr = \"{}\"", prefix, usr));
        }
        if let Ok(num) = c.num_args() {
            print_indent(depth, format!(" {}number-of-args = {}", prefix, num));
        }
        if let Some(num) = c.num_templ_args() {
            print_indent(depth, format!(" {}number-of-template-args = {}", prefix, num));
        }
        if c.is_bit_field() {
            let width = match c.bit_width() {
                Some(w) => w.to_string(),
                None => "<unevaluable>".to_string(),
            };
            print_indent(depth, format!(" {}bit-width = {}", prefix, width));
        }
        if let Some(ty) = c.enum_type() {
            print_indent(depth, format!(" {}enum-type = {}", prefix, type_to_str(ty.kind())));
        }
        if let Some(val) = c.enum_val_signed() {
            print_indent(depth, format!(" {}enum-val = {}", prefix, val));
        }
        if let Some(ty) = c.typedef_type() {
            print_indent(depth, format!(" {}typedef-type = {}", prefix, type_to_str(ty.kind())));
        }
        if let Some(ty) = c.ret_type() {
            print_indent(depth, format!(" {}ret-type = {}", prefix, type_to_str(ty.kind())));
        }
        if let Some(refd) = c.referenced() {
            if refd != *c {
                println!();
                print_cursor(depth, String::from(prefix) + "referenced.", &refd);
            }
        }
        let canon = c.canonical();
        if canon != *c {
            println!();
            print_cursor(depth, String::from(prefix) + "canonical.", &canon);
        }
        if let Some(specialized) = c.specialized() {
            if specialized != *c {
                println!();
                print_cursor(depth, String::from(prefix) + "specialized.", &specialized);
            }
        }
        if let Some(parent) = c.fallible_semantic_parent() {
            println!();
            print_cursor(depth, String::from(prefix) + "semantic-parent.", &parent);
        }
    }
    fn print_type<S: AsRef<str>>(depth: isize, prefix: S, ty: &Type) {
        let prefix = prefix.as_ref();
        let kind = ty.kind();
        print_indent(depth, format!(" {}kind = {}", prefix, type_to_str(kind)));
        if kind == CXType_Invalid {
            return;
        }
        print_indent(depth, format!(" {}cconv = {}", prefix, ty.call_conv()));
        print_indent(depth, format!(" {}spelling = \"{}\"", prefix, ty.spelling()));
        let n = unsafe { clang_Type_getNumTemplateArguments(ty.ty) };
        if n >= 0 {
            print_indent(depth, format!(" {}number-of-template-args = {}", prefix, n));
        }
        if let Some(num) = ty.num_elems() {
            print_indent(depth, format!(" {}number-of-elements = {}", prefix, num));
        }
        print_indent(depth, format!(" {}is-variadic? {}", prefix, ty.is_variadic()));
        let canonical = ty.canon_type();
        if canonical != *ty {
            println!();
            print_type(depth, String::from(prefix) + "canonical.", &canonical);
        }
        if let Some(pointee) = ty.pointee_type() {
            if pointee != *ty {
                println!();
                print_type(depth, String::from(prefix) + "pointee.", &pointee);
            }
        }
        if let Some(elem) = ty.elem_type() {
            if elem != *ty {
                println!();
                print_type(depth, String::from(prefix) + "elements.", &elem);
            }
        }
        if let Some(ret) = ty.ret_type() {
            if ret != *ty {
                println!();
                print_type(depth, String::from(prefix) + "return.", &ret);
            }
        }
        let named = ty.named();
        if named != *ty && named.is_valid() {
            println!();
            print_type(depth, String::from(prefix) + "named.", &named);
        }
    }
    print_indent(depth, "(");
    print_cursor(depth, "", c);
    println!();
    let ty = c.cur_type();
    print_type(depth, "type.", &ty);
    let declaration = ty.decl();
    if declaration != *c && declaration.kind() != CXCursor_NoDeclFound {
        println!();
        print_cursor(depth, "type.declaration.", &declaration);
    }
    let mut found_children = false;
    c.visit(|s| {
        if !found_children {
            println!();
            found_children = true;
        }
        ast_dump(&s, depth + 1)
    });
    print_indent(depth, ")");
    CXChildVisit_Continue
}
pub fn extract_clang_version() -> String {
    unsafe { cxstring_into_string(clang_getClangVersion()) }
}

fn cxstring_to_string_leaky(x: CXString) -> String {
    if x.data.is_null() {
        return "".to_owned();
    }
    let y = unsafe { CStr::from_ptr(clang_getCString(x) as *const _) };
    y.to_string_lossy().into_owned()
}
fn cxstring_into_string(s: CXString) -> String {
    let y = cxstring_to_string_leaky(s);
    unsafe { clang_disposeString(s) };
    y
}
