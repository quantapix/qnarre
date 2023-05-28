use std::ffi::{CStr, CString};
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::os::raw::{c_char, c_int, c_longlong, c_uint, c_ulong, c_ulonglong};
use std::{mem, ptr, slice};

use clang_lib::*;

use crate::ir::context::BindgenContext;

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

#[derive(Copy, Clone)]
pub struct Cursor {
    c: CXCursor,
}

impl Cursor {
    pub fn usr(&self) -> Option<String> {
        let s = unsafe { cxstring_into_string(clang_getCursorUSR(self.c)) };
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    pub fn is_declaration(&self) -> bool {
        unsafe { clang_isDeclaration(self.kind()) != 0 }
    }

    pub fn is_anonymous(&self) -> bool {
        unsafe { clang_Cursor_isAnonymous(self.c) != 0 }
    }

    pub fn spelling(&self) -> String {
        unsafe { cxstring_into_string(clang_getCursorSpelling(self.c)) }
    }

    pub fn display_name(&self) -> String {
        unsafe { cxstring_into_string(clang_getCursorDisplayName(self.c)) }
    }

    pub fn mangling(&self) -> String {
        unsafe { cxstring_into_string(clang_Cursor_getMangling(self.c)) }
    }

    pub fn cxx_manglings(&self) -> Result<Vec<String>, ()> {
        use clang_lib::*;
        unsafe {
            let manglings = clang_Cursor_getCXXManglings(self.c);
            if manglings.is_null() {
                return Err(());
            }
            let count = (*manglings).Count as usize;

            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let string_ptr = (*manglings).Strings.add(i);
                result.push(cxstring_to_string_leaky(*string_ptr));
            }
            clang_disposeStringSet(manglings);
            Ok(result)
        }
    }

    pub fn is_builtin(&self) -> bool {
        let (file, _, _, _) = self.location().location();
        file.name().is_none()
    }

    pub fn lexical_parent(&self) -> Cursor {
        unsafe {
            Cursor {
                c: clang_getCursorLexicalParent(self.c),
            }
        }
    }

    pub fn fallible_semantic_parent(&self) -> Option<Cursor> {
        let sp = unsafe {
            Cursor {
                c: clang_getCursorSemanticParent(self.c),
            }
        };
        if sp == *self || !sp.is_valid() {
            return None;
        }
        Some(sp)
    }

    pub fn semantic_parent(&self) -> Cursor {
        self.fallible_semantic_parent().unwrap()
    }

    pub fn num_template_args(&self) -> Option<u32> {
        self.cur_type()
            .num_template_args()
            .or_else(|| {
                let n: c_int = unsafe { clang_Cursor_getNumTemplateArguments(self.c) };

                if n >= 0 {
                    Some(n as u32)
                } else {
                    debug_assert_eq!(n, -1);
                    None
                }
            })
            .or_else(|| {
                let canonical = self.canonical();
                if canonical != *self {
                    canonical.num_template_args()
                } else {
                    None
                }
            })
    }

    pub fn translation_unit(&self) -> Cursor {
        assert!(self.is_valid());
        unsafe {
            let tu = clang_Cursor_getTranslationUnit(self.c);
            let cursor = Cursor {
                c: clang_getTranslationUnitCursor(tu),
            };
            assert!(cursor.is_valid());
            cursor
        }
    }

    pub fn is_toplevel(&self) -> bool {
        let mut semantic_parent = self.fallible_semantic_parent();

        while semantic_parent.is_some()
            && (semantic_parent.unwrap().kind() == CXCursor_Namespace
                || semantic_parent.unwrap().kind() == CXCursor_NamespaceAlias
                || semantic_parent.unwrap().kind() == CXCursor_NamespaceRef)
        {
            semantic_parent = semantic_parent.unwrap().fallible_semantic_parent();
        }

        let tu = self.translation_unit();
        semantic_parent == tu.fallible_semantic_parent()
    }

    pub fn is_template_like(&self) -> bool {
        matches!(
            self.kind(),
            CXCursor_ClassTemplate | CXCursor_ClassTemplatePartialSpecialization | CXCursor_TypeAliasTemplateDecl
        )
    }

    pub fn is_macro_function_like(&self) -> bool {
        unsafe { clang_Cursor_isMacroFunctionLike(self.c) != 0 }
    }

    pub fn kind(&self) -> CXCursorKind {
        self.c.kind
    }

    pub fn is_definition(&self) -> bool {
        unsafe { clang_isCursorDefinition(self.c) != 0 }
    }

    pub fn is_template_specialization(&self) -> bool {
        self.specialized().is_some()
    }

    pub fn is_fully_specialized_template(&self) -> bool {
        self.is_template_specialization()
            && self.kind() != CXCursor_ClassTemplatePartialSpecialization
            && self.num_template_args().unwrap_or(0) > 0
    }

    pub fn is_in_non_fully_specialized_template(&self) -> bool {
        if self.is_toplevel() {
            return false;
        }

        let parent = self.semantic_parent();
        if parent.is_fully_specialized_template() {
            return false;
        }

        if !parent.is_template_like() {
            return parent.is_in_non_fully_specialized_template();
        }

        true
    }

    pub fn is_template_parameter(&self) -> bool {
        matches!(
            self.kind(),
            CXCursor_TemplateTemplateParameter | CXCursor_TemplateTypeParameter | CXCursor_NonTypeTemplateParameter
        )
    }

    pub fn is_dependent_on_template_parameter(&self) -> bool {
        fn visitor(found_template_parameter: &mut bool, cur: Cursor) -> CXChildVisitResult {
            if cur.is_template_parameter() {
                *found_template_parameter = true;
                return CXChildVisit_Break;
            }

            if let Some(referenced) = cur.referenced() {
                if referenced.is_template_parameter() {
                    *found_template_parameter = true;
                    return CXChildVisit_Break;
                }

                referenced.visit(|next| visitor(found_template_parameter, next));
                if *found_template_parameter {
                    return CXChildVisit_Break;
                }
            }

            CXChildVisit_Recurse
        }

        if self.is_template_parameter() {
            return true;
        }

        let mut found_template_parameter = false;
        self.visit(|next| visitor(&mut found_template_parameter, next));

        found_template_parameter
    }

    pub fn is_valid(&self) -> bool {
        unsafe { clang_isInvalid(self.kind()) == 0 }
    }

    pub fn location(&self) -> SrcLoc {
        unsafe {
            SrcLoc {
                x: clang_getCursorLocation(self.c),
            }
        }
    }

    pub fn extent(&self) -> CXSourceRange {
        unsafe { clang_getCursorExtent(self.c) }
    }

    pub fn raw_comment(&self) -> Option<String> {
        let s = unsafe { cxstring_into_string(clang_Cursor_getRawCommentText(self.c)) };
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    pub fn comment(&self) -> Comment {
        unsafe {
            Comment {
                c: clang_Cursor_getParsedComment(self.c),
            }
        }
    }

    pub fn cur_type(&self) -> Type {
        unsafe {
            Type {
                t: clang_getCursorType(self.c),
            }
        }
    }

    pub fn definition(&self) -> Option<Cursor> {
        unsafe {
            let ret = Cursor {
                c: clang_getCursorDefinition(self.c),
            };

            if ret.is_valid() && ret.kind() != CXCursor_NoDeclFound {
                Some(ret)
            } else {
                None
            }
        }
    }

    pub fn referenced(&self) -> Option<Cursor> {
        unsafe {
            let ret = Cursor {
                c: clang_getCursorReferenced(self.c),
            };

            if ret.is_valid() {
                Some(ret)
            } else {
                None
            }
        }
    }

    pub fn canonical(&self) -> Cursor {
        unsafe {
            Cursor {
                c: clang_getCanonicalCursor(self.c),
            }
        }
    }

    pub fn specialized(&self) -> Option<Cursor> {
        unsafe {
            let ret = Cursor {
                c: clang_getSpecializedCursorTemplate(self.c),
            };
            if ret.is_valid() {
                Some(ret)
            } else {
                None
            }
        }
    }

    pub fn template_kind(&self) -> CXCursorKind {
        unsafe { clang_getTemplateCursorKind(self.c) }
    }

    pub fn visit<Visitor>(&self, mut visitor: Visitor)
    where
        Visitor: FnMut(Cursor) -> CXChildVisitResult,
    {
        let data = &mut visitor as *mut Visitor;
        unsafe {
            clang_visitChildren(self.c, visit_children::<Visitor>, data.cast());
        }
    }

    pub fn collect_children(&self) -> Vec<Cursor> {
        let mut children = vec![];
        self.visit(|c| {
            children.push(c);
            CXChildVisit_Continue
        });
        children
    }

    pub fn has_children(&self) -> bool {
        let mut has_children = false;
        self.visit(|_| {
            has_children = true;
            CXChildVisit_Break
        });
        has_children
    }

    pub fn has_at_least_num_children(&self, n: usize) -> bool {
        assert!(n > 0);
        let mut num_left = n;
        self.visit(|_| {
            num_left -= 1;
            if num_left == 0 {
                CXChildVisit_Break
            } else {
                CXChildVisit_Continue
            }
        });
        num_left == 0
    }

    pub fn contains_cursor(&self, kind: CXCursorKind) -> bool {
        let mut found = false;

        self.visit(|c| {
            if c.kind() == kind {
                found = true;
                CXChildVisit_Break
            } else {
                CXChildVisit_Continue
            }
        });

        found
    }

    pub fn is_inlined_function(&self) -> bool {
        unsafe { clang_Cursor_isFunctionInlined(self.c) != 0 }
    }

    pub fn is_defaulted_function(&self) -> bool {
        unsafe { clang_CXXMethod_isDefaulted(self.c) != 0 }
    }

    pub fn is_deleted_function(&self) -> bool {
        self.is_inlined_function() && self.definition().is_none() && !self.is_defaulted_function()
    }

    pub fn is_bit_field(&self) -> bool {
        unsafe { clang_Cursor_isBitField(self.c) != 0 }
    }

    pub fn bit_width_expr(&self) -> Option<Cursor> {
        if !self.is_bit_field() {
            return None;
        }

        let mut result = None;
        self.visit(|cur| {
            if cur.kind() == CXCursor_TypeRef {
                return CXChildVisit_Continue;
            }

            result = Some(cur);

            CXChildVisit_Break
        });

        result
    }

    pub fn bit_width(&self) -> Option<u32> {
        if self.bit_width_expr()?.is_dependent_on_template_parameter() {
            return None;
        }

        unsafe {
            let w = clang_getFieldDeclBitWidth(self.c);
            if w == -1 {
                None
            } else {
                Some(w as u32)
            }
        }
    }

    pub fn enum_type(&self) -> Option<Type> {
        unsafe {
            let t = Type {
                t: clang_getEnumDeclIntegerType(self.c),
            };
            if t.is_valid() {
                Some(t)
            } else {
                None
            }
        }
    }

    pub fn enum_val_boolean(&self) -> Option<bool> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                Some(clang_getEnumConstantDeclValue(self.c) != 0)
            } else {
                None
            }
        }
    }

    pub fn enum_val_signed(&self) -> Option<i64> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                #[allow(clippy::unnecessary_cast)]
                Some(clang_getEnumConstantDeclValue(self.c) as i64)
            } else {
                None
            }
        }
    }

    pub fn enum_val_unsigned(&self) -> Option<u64> {
        unsafe {
            if self.kind() == CXCursor_EnumConstantDecl {
                #[allow(clippy::unnecessary_cast)]
                Some(clang_getEnumConstantDeclUnsignedValue(self.c) as u64)
            } else {
                None
            }
        }
    }

    pub fn has_attrs<const N: usize>(&self, attrs: &[Attribute; N]) -> [bool; N] {
        let mut found_attrs = [false; N];
        let mut found_count = 0;

        self.visit(|cur| {
            let kind = cur.kind();
            for (idx, attr) in attrs.iter().enumerate() {
                let found_attr = &mut found_attrs[idx];
                if !*found_attr {
                    if attr.kind.map_or(false, |k| k == kind)
                        || (kind == CXCursor_UnexposedAttr
                            && cur
                                .tokens()
                                .iter()
                                .any(|t| t.kind == attr.tok_kind && t.spelling() == attr.name))
                    {
                        *found_attr = true;
                        found_count += 1;

                        if found_count == N {
                            return CXChildVisit_Break;
                        }
                    }
                }
            }

            CXChildVisit_Continue
        });

        found_attrs
    }

    pub fn typedef_type(&self) -> Option<Type> {
        let inner = Type {
            t: unsafe { clang_getTypedefDeclUnderlyingType(self.c) },
        };

        if inner.is_valid() {
            Some(inner)
        } else {
            None
        }
    }

    pub fn linkage(&self) -> CXLinkageKind {
        unsafe { clang_getCursorLinkage(self.c) }
    }

    pub fn visibility(&self) -> CXVisibilityKind {
        unsafe { clang_getCursorVisibility(self.c) }
    }

    pub fn args(&self) -> Option<Vec<Cursor>> {
        self.num_args().ok().map(|num| {
            (0..num)
                .map(|i| Cursor {
                    c: unsafe { clang_Cursor_getArgument(self.c, i as c_uint) },
                })
                .collect()
        })
    }

    pub fn num_args(&self) -> Result<u32, ()> {
        unsafe {
            let w = clang_Cursor_getNumArguments(self.c);
            if w == -1 {
                Err(())
            } else {
                Ok(w as u32)
            }
        }
    }

    pub fn access_specifier(&self) -> CX_CXXAccessSpecifier {
        unsafe { clang_getCXXAccessSpecifier(self.c) }
    }

    pub fn public_accessible(&self) -> bool {
        let access = self.access_specifier();
        access == CX_CXXPublic || access == CX_CXXInvalidAccessSpecifier
    }

    pub fn is_mutable_field(&self) -> bool {
        unsafe { clang_CXXField_isMutable(self.c) != 0 }
    }

    pub fn offset_of_field(&self) -> Result<usize, LayoutError> {
        let offset = unsafe { clang_Cursor_getOffsetOfField(self.c) };

        if offset < 0 {
            Err(LayoutError::from(offset as i32))
        } else {
            Ok(offset as usize)
        }
    }

    pub fn method_is_static(&self) -> bool {
        unsafe { clang_CXXMethod_isStatic(self.c) != 0 }
    }

    pub fn method_is_const(&self) -> bool {
        unsafe { clang_CXXMethod_isConst(self.c) != 0 }
    }

    pub fn method_is_virtual(&self) -> bool {
        unsafe { clang_CXXMethod_isVirtual(self.c) != 0 }
    }

    pub fn method_is_pure_virtual(&self) -> bool {
        unsafe { clang_CXXMethod_isPureVirtual(self.c) != 0 }
    }

    pub fn is_virtual_base(&self) -> bool {
        unsafe { clang_isVirtualBase(self.c) != 0 }
    }

    pub fn evaluate(&self) -> Option<EvalResult> {
        EvalResult::new(*self)
    }

    pub fn ret_type(&self) -> Option<Type> {
        let rt = Type {
            t: unsafe { clang_getCursorResultType(self.c) },
        };
        if rt.is_valid() {
            Some(rt)
        } else {
            None
        }
    }

    pub fn tokens(&self) -> RawTokens {
        RawTokens::new(self)
    }

    pub fn cexpr_tokens(self) -> Vec<cexpr::token::Token> {
        self.tokens()
            .iter()
            .filter_map(|token| token.as_cexpr_token())
            .collect()
    }

    pub fn get_included_file_name(&self) -> Option<String> {
        let file = unsafe { clang_lib::clang_getIncludedFile(self.c) };
        if file.is_null() {
            None
        } else {
            Some(unsafe { cxstring_into_string(clang_lib::clang_getFileName(file)) })
        }
    }
}

impl PartialEq for Cursor {
    fn eq(&self, x: &Cursor) -> bool {
        unsafe { clang_equalCursors(self.c, x.c) == 1 }
    }
}

impl Eq for Cursor {}

impl Hash for Cursor {
    fn hash<H: Hasher>(&self, x: &mut H) {
        unsafe { clang_hashCursor(self.c) }.hash(x)
    }
}

impl fmt::Debug for Cursor {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "Cursor({} kind: {}, loc: {}, usr: {:?})",
            self.spelling(),
            kind_to_str(self.kind()),
            self.location(),
            self.usr()
        )
    }
}

pub struct RawTokens<'a> {
    cursor: &'a Cursor,
    tu: CXTranslationUnit,
    tokens: *mut CXToken,
    count: c_uint,
}

impl<'a> RawTokens<'a> {
    fn new(cursor: &'a Cursor) -> Self {
        let mut tokens = ptr::null_mut();
        let mut token_count = 0;
        let range = cursor.extent();
        let tu = unsafe { clang_Cursor_getTranslationUnit(cursor.c) };
        unsafe { clang_tokenize(tu, range, &mut tokens, &mut token_count) };
        Self {
            cursor,
            tu,
            tokens,
            count: token_count,
        }
    }

    fn as_slice(&self) -> &[CXToken] {
        if self.tokens.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.tokens, self.count as usize) }
    }

    pub fn iter(&self) -> TokenIter {
        TokenIter {
            tu: self.tu,
            raw: self.as_slice().iter(),
        }
    }
}

impl<'a> Drop for RawTokens<'a> {
    fn drop(&mut self) {
        if !self.tokens.is_null() {
            unsafe {
                clang_disposeTokens(self.tu, self.tokens, self.count as c_uint);
            }
        }
    }
}

#[derive(Debug)]
pub struct Token {
    spelling: CXString,
    pub extent: CXSourceRange,
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

pub struct TokenIter<'a> {
    tu: CXTranslationUnit,
    raw: slice::Iter<'a, CXToken>,
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = Token;
    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.raw.next()?;
        unsafe {
            let kind = clang_getTokenKind(*raw);
            let spelling = clang_getTokenSpelling(self.tu, *raw);
            let extent = clang_getTokenExtent(self.tu, *raw);
            Some(Token { kind, extent, spelling })
        }
    }
}

pub fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let first_valid = chars.next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false);
    first_valid && chars.all(|c| c.is_alphanumeric() || c == '_')
}

extern "C" fn visit_children<Visitor>(cur: CXCursor, _parent: CXCursor, data: CXClientData) -> CXChildVisitResult
where
    Visitor: FnMut(Cursor) -> CXChildVisitResult,
{
    let func: &mut Visitor = unsafe { &mut *(data as *mut Visitor) };
    let child = Cursor { c: cur };

    (*func)(child)
}

#[derive(Clone, Copy)]
pub struct Type {
    t: CXType,
}

impl Type {
    pub fn kind(&self) -> CXTypeKind {
        self.t.kind
    }
    pub fn declaration(&self) -> Cursor {
        unsafe {
            Cursor {
                c: clang_getTypeDeclaration(self.t),
            }
        }
    }
    pub fn canonical_declaration(&self, loc: Option<&Cursor>) -> Option<CanonicalTypeDeclaration> {
        let mut y = self.declaration();
        if !y.is_valid() {
            if let Some(loc) = loc {
                let mut loc = *loc;
                if let Some(referenced) = loc.referenced() {
                    loc = referenced;
                }
                if loc.is_template_like() {
                    y = loc;
                }
            }
        }
        let y = y.canonical();
        if y.is_valid() && y.kind() != CXCursor_NoDeclFound {
            Some(CanonicalTypeDeclaration(*self, y))
        } else {
            None
        }
    }
    pub fn spelling(&self) -> String {
        let y = unsafe { cxstring_into_string(clang_getTypeSpelling(self.t)) };
        if y.split("::").all(is_valid_identifier) {
            if let Some(y) = y.split("::").last() {
                return y.to_owned();
            }
        }
        y
    }
    pub fn is_const(&self) -> bool {
        unsafe { clang_isConstQualifiedType(self.t) != 0 }
    }
    #[inline]
    fn is_non_deductible_auto_type(&self) -> bool {
        debug_assert_eq!(self.kind(), CXType_Auto);
        self.canonical_type() == *self
    }
    #[inline]
    fn clang_size_of(&self, ctx: &BindgenContext) -> c_longlong {
        match self.kind() {
            CXType_RValueReference | CXType_LValueReference => ctx.target_pointer_size() as c_longlong,
            CXType_Auto if self.is_non_deductible_auto_type() => -6,
            _ => unsafe { clang_Type_getSizeOf(self.t) },
        }
    }
    #[inline]
    fn clang_align_of(&self, ctx: &BindgenContext) -> c_longlong {
        match self.kind() {
            CXType_RValueReference | CXType_LValueReference => ctx.target_pointer_size() as c_longlong,
            CXType_Auto if self.is_non_deductible_auto_type() => -6,
            _ => unsafe { clang_Type_getAlignOf(self.t) },
        }
    }
    pub fn size(&self, ctx: &BindgenContext) -> usize {
        let y = self.clang_size_of(ctx);
        if y < 0 {
            0
        } else {
            y as usize
        }
    }
    pub fn fallible_size(&self, ctx: &BindgenContext) -> Result<usize, LayoutError> {
        let y = self.clang_size_of(ctx);
        if y < 0 {
            Err(LayoutError::from(y as i32))
        } else {
            Ok(y as usize)
        }
    }
    pub fn align(&self, ctx: &BindgenContext) -> usize {
        let y = self.clang_align_of(ctx);
        if y < 0 {
            0
        } else {
            y as usize
        }
    }
    pub fn fallible_align(&self, ctx: &BindgenContext) -> Result<usize, LayoutError> {
        let y = self.clang_align_of(ctx);
        if y < 0 {
            Err(LayoutError::from(y as i32))
        } else {
            Ok(y as usize)
        }
    }
    pub fn fallible_layout(&self, ctx: &BindgenContext) -> Result<crate::ir::layout::Layout, LayoutError> {
        use crate::ir::layout::Layout;
        let size = self.fallible_size(ctx)?;
        let align = self.fallible_align(ctx)?;
        Ok(Layout::new(size, align))
    }
    pub fn num_template_args(&self) -> Option<u32> {
        let y = unsafe { clang_Type_getNumTemplateArguments(self.t) };
        if y >= 0 {
            Some(y as u32)
        } else {
            debug_assert_eq!(y, -1);
            None
        }
    }
    pub fn template_args(&self) -> Option<TypeTemplateArgIterator> {
        self.num_template_args().map(|n| TypeTemplateArgIterator {
            x: self.t,
            length: n,
            index: 0,
        })
    }
    pub fn args(&self) -> Option<Vec<Type>> {
        self.num_args().ok().map(|x| {
            (0..x)
                .map(|i| Type {
                    t: unsafe { clang_getArgType(self.t, i as c_uint) },
                })
                .collect()
        })
    }
    pub fn num_args(&self) -> Result<u32, ()> {
        unsafe {
            let y = clang_getNumArgTypes(self.t);
            if y == -1 {
                Err(())
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
                    t: unsafe { clang_getPointeeType(self.t) },
                };
                debug_assert!(y.is_valid());
                Some(y)
            },
            _ => None,
        }
    }
    pub fn elem_type(&self) -> Option<Type> {
        let y = Type {
            t: unsafe { clang_getElementType(self.t) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn num_elements(&self) -> Option<usize> {
        let y = unsafe { clang_getNumElements(self.t) };
        if y != -1 {
            Some(y as usize)
        } else {
            None
        }
    }
    pub fn canonical_type(&self) -> Type {
        unsafe {
            Type {
                t: clang_getCanonicalType(self.t),
            }
        }
    }
    pub fn is_variadic(&self) -> bool {
        unsafe { clang_isFunctionTypeVariadic(self.t) != 0 }
    }
    pub fn ret_type(&self) -> Option<Type> {
        let y = Type {
            t: unsafe { clang_getResultType(self.t) },
        };
        if y.is_valid() {
            Some(y)
        } else {
            None
        }
    }
    pub fn call_conv(&self) -> CXCallingConv {
        unsafe { clang_getFunctionTypeCallingConv(self.t) }
    }
    pub fn named(&self) -> Type {
        unsafe {
            Type {
                t: clang_Type_getNamedType(self.t),
            }
        }
    }
    pub fn is_valid(&self) -> bool {
        self.kind() != CXType_Invalid
    }

    pub fn is_valid_and_exposed(&self) -> bool {
        self.is_valid() && self.kind() != CXType_Unexposed
    }

    pub fn is_fully_instantiated_template(&self) -> bool {
        self.template_args().map_or(false, |x| x.len() > 0)
            && !matches!(
                self.declaration().kind(),
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
                || hacky_parse_associated_type(self.canonical_type().spelling()))
    }
}

impl PartialEq for Type {
    fn eq(&self, x: &Self) -> bool {
        unsafe { clang_equalTypes(self.t, x.t) != 0 }
    }
}

impl Eq for Type {}

impl fmt::Debug for Type {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "Type({}, kind: {}, cconv: {}, decl: {:?}, canon: {:?})",
            self.spelling(),
            type_to_str(self.kind()),
            self.call_conv(),
            self.declaration(),
            self.declaration().canonical()
        )
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum LayoutError {
    Invalid,
    Incomplete,
    Dependent,
    NotConstantSize,
    InvalidFieldName,
    Unknown,
}

impl ::std::convert::From<i32> for LayoutError {
    fn from(val: i32) -> Self {
        use self::LayoutError::*;

        match val {
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
pub struct CanonicalTypeDeclaration(Type, Cursor);

impl CanonicalTypeDeclaration {
    pub fn ty(&self) -> &Type {
        &self.0
    }

    pub fn cursor(&self) -> &Cursor {
        &self.1
    }
}

pub struct TypeTemplateArgIterator {
    x: CXType,
    length: u32,
    index: u32,
}

impl Iterator for TypeTemplateArgIterator {
    type Item = Type;
    fn next(&mut self) -> Option<Type> {
        if self.index < self.length {
            let idx = self.index as c_uint;
            self.index += 1;
            Some(Type {
                t: unsafe { clang_Type_getTemplateArgumentAsType(self.x, idx) },
            })
        } else {
            None
        }
    }
}

impl ExactSizeIterator for TypeTemplateArgIterator {
    fn len(&self) -> usize {
        assert!(self.index <= self.length);
        (self.length - self.index) as usize
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
            (File { x: file }, line as usize, col as usize, off as usize)
        }
    }
}

impl fmt::Display for SrcLoc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (file, line, col, _) = self.location();
        if let Some(name) = file.name() {
            write!(f, "{}:{}:{}", name, line, col)
        } else {
            "builtin definitions".fmt(f)
        }
    }
}

impl fmt::Debug for SrcLoc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

pub struct Comment {
    c: CXComment,
}

impl Comment {
    pub fn kind(&self) -> CXCommentKind {
        unsafe { clang_Comment_getKind(self.c) }
    }
    pub fn get_children(&self) -> CommentChildrenIterator {
        CommentChildrenIterator {
            parent: self.c,
            length: unsafe { clang_Comment_getNumChildren(self.c) },
            index: 0,
        }
    }
    pub fn get_tag_name(&self) -> String {
        unsafe { cxstring_into_string(clang_HTMLTagComment_getTagName(self.c)) }
    }
    pub fn get_tag_attrs(&self) -> CommentAttributesIterator {
        CommentAttributesIterator {
            x: self.c,
            length: unsafe { clang_HTMLStartTag_getNumAttrs(self.c) },
            index: 0,
        }
    }
}

pub struct CommentChildrenIterator {
    parent: CXComment,
    length: c_uint,
    index: c_uint,
}

impl Iterator for CommentChildrenIterator {
    type Item = Comment;
    fn next(&mut self) -> Option<Comment> {
        if self.index < self.length {
            let idx = self.index;
            self.index += 1;
            Some(Comment {
                c: unsafe { clang_Comment_getChild(self.parent, idx) },
            })
        } else {
            None
        }
    }
}

pub struct CommentAttribute {
    pub name: String,
    pub value: String,
}

pub struct CommentAttributesIterator {
    x: CXComment,
    length: c_uint,
    index: c_uint,
}

impl Iterator for CommentAttributesIterator {
    type Item = CommentAttribute;
    fn next(&mut self) -> Option<CommentAttribute> {
        if self.index < self.length {
            let idx = self.index;
            self.index += 1;
            Some(CommentAttribute {
                name: unsafe { cxstring_into_string(clang_HTMLStartTag_getAttrName(self.x, idx)) },
                value: unsafe { cxstring_into_string(clang_HTMLStartTag_getAttrValue(self.x, idx)) },
            })
        } else {
            None
        }
    }
}

pub struct File {
    x: CXFile,
}

impl File {
    pub fn name(&self) -> Option<String> {
        if self.x.is_null() {
            return None;
        }
        Some(unsafe { cxstring_into_string(clang_getFileName(self.x)) })
    }
}

fn cxstring_to_string_leaky(s: CXString) -> String {
    if s.data.is_null() {
        return "".to_owned();
    }
    let c_str = unsafe { CStr::from_ptr(clang_getCString(s) as *const _) };
    c_str.to_string_lossy().into_owned()
}

fn cxstring_into_string(s: CXString) -> String {
    let ret = cxstring_to_string_leaky(s);
    unsafe { clang_disposeString(s) };
    ret
}

pub struct Index {
    x: CXIndex,
}

impl Index {
    pub fn new(pch: bool, diag: bool) -> Index {
        unsafe {
            Index {
                x: clang_createIndex(pch as c_int, diag as c_int),
            }
        }
    }
}

impl fmt::Debug for Index {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Index {{ }}")
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe {
            clang_disposeIndex(self.x);
        }
    }
}

pub struct TranslationUnit {
    x: CXTranslationUnit,
}

impl fmt::Debug for TranslationUnit {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "TranslationUnit {{ }}")
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
        let mut c_unsaved: Vec<CXUnsavedFile> = unsaved.iter().map(|f| f.x).collect();
        let tu = unsafe {
            clang_parseTranslationUnit(
                ix.x,
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
            Some(TranslationUnit { x: tu })
        }
    }

    pub fn diags(&self) -> Vec<Diagnostic> {
        unsafe {
            let num = clang_getNumDiagnostics(self.x) as usize;
            let mut diags = vec![];
            for i in 0..num {
                diags.push(Diagnostic {
                    x: clang_getDiagnostic(self.x, i as c_uint),
                });
            }
            diags
        }
    }

    pub fn cursor(&self) -> Cursor {
        unsafe {
            Cursor {
                c: clang_getTranslationUnitCursor(self.x),
            }
        }
    }

    pub fn is_null(&self) -> bool {
        self.x.is_null()
    }
}

impl Drop for TranslationUnit {
    fn drop(&mut self) {
        unsafe {
            clang_disposeTranslationUnit(self.x);
        }
    }
}

pub struct Diagnostic {
    x: CXDiagnostic,
}

impl Diagnostic {
    pub fn format(&self) -> String {
        unsafe {
            let opts = clang_defaultDiagnosticDisplayOptions();
            cxstring_into_string(clang_formatDiagnostic(self.x, opts))
        }
    }

    pub fn severity(&self) -> CXDiagnosticSeverity {
        unsafe { clang_getDiagnosticSeverity(self.x) }
    }
}

impl Drop for Diagnostic {
    fn drop(&mut self) {
        unsafe {
            clang_disposeDiagnostic(self.x);
        }
    }
}

pub struct UnsavedFile {
    x: CXUnsavedFile,
    pub name: CString,
    contents: CString,
}

impl UnsavedFile {
    pub fn new(name: String, contents: String) -> UnsavedFile {
        let name = CString::new(name).unwrap();
        let contents = CString::new(contents).unwrap();
        let x = CXUnsavedFile {
            Filename: name.as_ptr(),
            Contents: contents.as_ptr(),
            Length: contents.as_bytes().len() as c_ulong,
        };
        UnsavedFile { x, name, contents }
    }
}

impl fmt::Debug for UnsavedFile {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "UnsavedFile(name: {:?}, contents: {:?})", self.name, self.contents)
    }
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
        print_indent(
            depth,
            format!(" {}is-inlined-function? {}", prefix, c.is_inlined_function()),
        );

        let templ_kind = c.template_kind();
        if templ_kind != CXCursor_NoDeclFound {
            print_indent(depth, format!(" {}template-kind = {}", prefix, kind_to_str(templ_kind)));
        }
        if let Some(usr) = c.usr() {
            print_indent(depth, format!(" {}usr = \"{}\"", prefix, usr));
        }
        if let Ok(num) = c.num_args() {
            print_indent(depth, format!(" {}number-of-args = {}", prefix, num));
        }
        if let Some(num) = c.num_template_args() {
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

        let canonical = c.canonical();
        if canonical != *c {
            println!();
            print_cursor(depth, String::from(prefix) + "canonical.", &canonical);
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
        let num_template_args = unsafe { clang_Type_getNumTemplateArguments(ty.t) };
        if num_template_args >= 0 {
            print_indent(
                depth,
                format!(" {}number-of-template-args = {}", prefix, num_template_args),
            );
        }
        if let Some(num) = ty.num_elements() {
            print_indent(depth, format!(" {}number-of-elements = {}", prefix, num));
        }
        print_indent(depth, format!(" {}is-variadic? {}", prefix, ty.is_variadic()));

        let canonical = ty.canonical_type();
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

    let declaration = ty.declaration();
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

#[derive(Debug)]
pub struct EvalResult {
    x: CXEvalResult,
    ty: Type,
}

impl EvalResult {
    pub fn new(cursor: Cursor) -> Option<Self> {
        {
            let mut found_cant_eval = false;
            cursor.visit(|c| {
                if c.kind() == CXCursor_TypeRef && c.cur_type().canonical_type().kind() == CXType_Unexposed {
                    found_cant_eval = true;
                    return CXChildVisit_Break;
                }

                CXChildVisit_Recurse
            });

            if found_cant_eval {
                return None;
            }
        }
        Some(EvalResult {
            x: unsafe { clang_Cursor_Evaluate(cursor.c) },
            ty: cursor.cur_type().canonical_type(),
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
        Some(value as i64)
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
pub struct TargetInfo {
    pub triple: String,
    pub pointer_width: usize,
}

impl TargetInfo {
    pub fn new(tu: &TranslationUnit) -> Self {
        let triple;
        let pointer_width;
        unsafe {
            let ti = clang_getTranslationUnitTargetInfo(tu.x);
            triple = cxstring_into_string(clang_TargetInfo_getTriple(ti));
            pointer_width = clang_TargetInfo_getPointerWidth(ti);
            clang_TargetInfo_dispose(ti);
        }
        assert!(pointer_width > 0);
        assert_eq!(pointer_width % 8, 0);
        TargetInfo {
            triple,
            pointer_width: pointer_width as usize,
        }
    }
}
