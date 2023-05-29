pub mod analysis;
pub mod annotations {
    use crate::clang;
    use std::str::FromStr;
    #[derive(Copy, PartialEq, Eq, Clone, Debug)]
    pub enum FieldVisibilityKind {
        Private,
        PublicCrate,
        Public,
    }
    impl FromStr for FieldVisibilityKind {
        type Err = String;
        fn from_str(x: &str) -> Result<Self, Self::Err> {
            match x {
                "private" => Ok(Self::Private),
                "crate" => Ok(Self::PublicCrate),
                "public" => Ok(Self::Public),
                _ => Err(format!("Invalid visibility kind: `{}`", x)),
            }
        }
    }
    impl std::fmt::Display for FieldVisibilityKind {
        fn fmt(&self, x: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let y = match self {
                FieldVisibilityKind::Private => "private",
                FieldVisibilityKind::PublicCrate => "crate",
                FieldVisibilityKind::Public => "public",
            };
            y.fmt(x)
        }
    }
    impl Default for FieldVisibilityKind {
        fn default() -> Self {
            FieldVisibilityKind::Public
        }
    }
    #[derive(Copy, PartialEq, Eq, Clone, Debug)]
    pub enum FieldAccessorKind {
        None,
        Regular,
        Unsafe,
        Immutable,
    }
    fn parse_accessor(x: &str) -> FieldAccessorKind {
        match x {
            "false" => FieldAccessorKind::None,
            "unsafe" => FieldAccessorKind::Unsafe,
            "immutable" => FieldAccessorKind::Immutable,
            _ => FieldAccessorKind::Regular,
        }
    }
    #[derive(Default, Clone, PartialEq, Eq, Debug)]
    pub struct Annotations {
        opaque: bool,
        hide: bool,
        use_instead_of: Option<Vec<String>>,
        disallow_copy: bool,
        disallow_debug: bool,
        disallow_default: bool,
        must_use_type: bool,
        visibility_kind: Option<FieldVisibilityKind>,
        accessor_kind: Option<FieldAccessorKind>,
        constify_enum_variant: bool,
        derives: Vec<String>,
    }
    impl Annotations {
        pub fn new(cur: &clang::Cursor) -> Option<Annotations> {
            let mut anno = Annotations::default();
            let mut matched_one = false;
            anno.parse(&cur.comment(), &mut matched_one);
            if matched_one {
                Some(anno)
            } else {
                None
            }
        }
        pub fn hide(&self) -> bool {
            self.hide
        }
        pub fn opaque(&self) -> bool {
            self.opaque
        }
        pub fn use_instead_of(&self) -> Option<&[String]> {
            self.use_instead_of.as_deref()
        }
        pub fn derives(&self) -> &[String] {
            &self.derives
        }
        pub fn disallow_copy(&self) -> bool {
            self.disallow_copy
        }
        pub fn disallow_debug(&self) -> bool {
            self.disallow_debug
        }
        pub fn disallow_default(&self) -> bool {
            self.disallow_default
        }
        pub fn must_use_type(&self) -> bool {
            self.must_use_type
        }
        pub fn visibility_kind(&self) -> Option<FieldVisibilityKind> {
            self.visibility_kind
        }
        pub fn accessor_kind(&self) -> Option<FieldAccessorKind> {
            self.accessor_kind
        }
        fn parse(&mut self, comment: &clang::Comment, matched: &mut bool) {
            use clang_lib::CXComment_HTMLStartTag;
            if comment.kind() == CXComment_HTMLStartTag
                && comment.get_tag_name() == "div"
                && comment
                    .get_tag_attrs()
                    .next()
                    .map_or(false, |x| x.name == "rustbindgen")
            {
                *matched = true;
                for x in comment.get_tag_attrs() {
                    match x.name.as_str() {
                        "opaque" => self.opaque = true,
                        "hide" => self.hide = true,
                        "nocopy" => self.disallow_copy = true,
                        "nodebug" => self.disallow_debug = true,
                        "nodefault" => self.disallow_default = true,
                        "mustusetype" => self.must_use_type = true,
                        "replaces" => self.use_instead_of = Some(x.value.split("::").map(Into::into).collect()),
                        "derive" => self.derives.push(x.value),
                        "private" => {
                            self.visibility_kind = if x.value != "false" {
                                Some(FieldVisibilityKind::Private)
                            } else {
                                Some(FieldVisibilityKind::Public)
                            };
                        },
                        "accessor" => self.accessor_kind = Some(parse_accessor(&x.value)),
                        "constant" => self.constify_enum_variant = true,
                        _ => {},
                    }
                }
            }
            for x in comment.get_children() {
                self.parse(&x, matched);
            }
        }
        pub fn constify_enum_variant(&self) -> bool {
            self.constify_enum_variant
        }
    }
}
pub mod comment {
    #[derive(Debug, PartialEq, Eq)]
    enum Kind {
        SingleLines,
        MultiLine,
    }
    pub fn preproc(x: &str) -> String {
        match self::kind(x) {
            Some(Kind::SingleLines) => preprocess_single_lines(x),
            Some(Kind::MultiLine) => preprocess_multi_line(x),
            None => x.to_owned(),
        }
    }
    fn kind(x: &str) -> Option<Kind> {
        if x.starts_with("/*") {
            Some(Kind::MultiLine)
        } else if x.starts_with("//") {
            Some(Kind::SingleLines)
        } else {
            None
        }
    }
    fn preprocess_single_lines(x: &str) -> String {
        debug_assert!(x.starts_with("//"), "comment is not single line");
        let ys: Vec<_> = x.lines().map(|l| l.trim().trim_start_matches('/')).collect();
        ys.join("\n")
    }
    fn preprocess_multi_line(x: &str) -> String {
        let x = x.trim_start_matches('/').trim_end_matches('/').trim_end_matches('*');
        let mut ys: Vec<_> = x
            .lines()
            .map(|x| x.trim().trim_start_matches('*').trim_start_matches('!'))
            .skip_while(|x| x.trim().is_empty())
            .collect();
        if ys.last().map_or(false, |x| x.trim().is_empty()) {
            ys.pop();
        }
        ys.join("\n")
    }
    #[cfg(test)]
    mod test {
        use super::*;
        #[test]
        fn picks_up_single_and_multi_line_doc_comments() {
            assert_eq!(kind("/// hello"), Some(Kind::SingleLines));
            assert_eq!(kind("/** world */"), Some(Kind::MultiLine));
        }
        #[test]
        fn processes_single_lines_correctly() {
            assert_eq!(preproc("///"), "");
            assert_eq!(preproc("/// hello"), " hello");
            assert_eq!(preproc("// hello"), " hello");
            assert_eq!(preproc("//    hello"), "    hello");
        }
        #[test]
        fn processes_multi_lines_correctly() {
            assert_eq!(preproc("/**/"), "");
            assert_eq!(preproc("/** hello \n * world \n * foo \n */"), " hello\n world\n foo");
            assert_eq!(preproc("/**\nhello\n*world\n*foo\n*/"), "hello\nworld\nfoo");
        }
    }
}
pub mod comp;
pub mod context;
pub mod derive {
    use super::context::BindgenContext;
    use std::cmp;
    use std::ops;
    pub trait CanDeriveDebug {
        fn can_derive_debug(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDeriveCopy {
        fn can_derive_copy(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDeriveDefault {
        fn can_derive_default(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDeriveHash {
        fn can_derive_hash(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDerivePartialEq {
        fn can_derive_partialeq(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDerivePartialOrd {
        fn can_derive_partialord(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDeriveEq {
        fn can_derive_eq(&self, ctx: &BindgenContext) -> bool;
    }
    pub trait CanDeriveOrd {
        fn can_derive_ord(&self, ctx: &BindgenContext) -> bool;
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub enum YDerive {
        Yes,
        Manually,
        No,
    }
    impl Default for YDerive {
        fn default() -> YDerive {
            YDerive::Yes
        }
    }
    impl YDerive {
        pub fn join(self, rhs: Self) -> Self {
            cmp::max(self, rhs)
        }
    }
    impl ops::BitOr for YDerive {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output {
            self.join(rhs)
        }
    }
    impl ops::BitOrAssign for YDerive {
        fn bitor_assign(&mut self, rhs: Self) {
            *self = self.join(rhs)
        }
    }
}
pub mod dot {
    use super::context::{BindgenContext, ItemId};
    use super::traversal::Trace;
    use std::fs::File;
    use std::io::{self, Write};
    use std::path::Path;
    pub trait DotAttrs {
        fn dot_attrs<W>(&self, ctx: &BindgenContext, y: &mut W) -> io::Result<()>
        where
            W: io::Write;
    }
    pub fn write_dot_file<P>(ctx: &BindgenContext, path: P) -> io::Result<()>
    where
        P: AsRef<Path>,
    {
        let mut y = io::BufWriter::new(File::create(path)?);
        writeln!(&mut y, "digraph {{")?;
        let mut err: Option<io::Result<_>> = None;
        for (id, it) in ctx.items() {
            let is_allowed = ctx.allowed_items().contains(&id);
            writeln!(
                &mut y,
                r#"{} [fontname="courier", color={}, label=< <table border="0" align="left">"#,
                id.as_usize(),
                if is_allowed { "black" } else { "gray" }
            )?;
            it.dot_attrs(ctx, &mut y)?;
            writeln!(&mut y, r#"</table> >];"#)?;
            it.trace(
                ctx,
                &mut |id2: ItemId, kind| {
                    if err.is_some() {
                        return;
                    }
                    match writeln!(
                        &mut y,
                        "{} -> {} [label={:?}, color={}];",
                        id.as_usize(),
                        id2.as_usize(),
                        kind,
                        if is_allowed { "black" } else { "gray" }
                    ) {
                        Ok(_) => {},
                        Err(x) => err = Some(Err(x)),
                    }
                },
                &(),
            );
            if let Some(x) = err {
                return x;
            }
            if let Some(x) = it.as_module() {
                for x in x.children() {
                    writeln!(
                        &mut y,
                        "{} -> {} [style=dotted, color=gray]",
                        it.id().as_usize(),
                        x.as_usize()
                    )?;
                }
            }
        }
        writeln!(&mut y, "}}")?;
        Ok(())
    }
}
pub mod enum_ty {
    use super::super::codegen::EnumVariation;
    use super::context::{BindgenContext, TypeId};
    use super::item::Item;
    use super::ty::{TyKind, Type};
    use crate::clang;
    use crate::ir::annotations::Annotations;
    use crate::parse;
    use crate::regex_set::RegexSet;
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum EnumVariantCustomBehavior {
        ModuleConstify,
        Constify,
        Hide,
    }
    #[derive(Debug)]
    pub struct Enum {
        repr: Option<TypeId>,
        variants: Vec<EnumVariant>,
    }
    impl Enum {
        pub fn new(repr: Option<TypeId>, variants: Vec<EnumVariant>) -> Self {
            Enum { repr, variants }
        }
        pub fn repr(&self) -> Option<TypeId> {
            self.repr
        }
        pub fn variants(&self) -> &[EnumVariant] {
            &self.variants
        }
        pub fn from_ty(ty: &clang::Type, ctx: &mut BindgenContext) -> Result<Self, parse::Error> {
            use clang_lib::*;
            debug!("Enum::from_ty {:?}", ty);
            if ty.kind() != CXType_Enum {
                return Err(parse::Error::Continue);
            }
            let declaration = ty.declaration().canonical();
            let repr = declaration
                .enum_type()
                .and_then(|et| Item::from_ty(&et, declaration, None, ctx).ok());
            let mut variants = vec![];
            let variant_ty = repr.and_then(|r| ctx.resolve_type(r).safe_canonical_type(ctx));
            let is_bool = variant_ty.map_or(false, Type::is_bool);
            let is_signed = variant_ty.map_or(true, |ty| match *ty.kind() {
                TyKind::Int(ref int_kind) => int_kind.is_signed(),
                ref other => {
                    panic!("Since when enums can be non-integers? {:?}", other)
                },
            });
            let type_name = ty.spelling();
            let type_name = if type_name.is_empty() { None } else { Some(type_name) };
            let type_name = type_name.as_deref();
            let definition = declaration.definition().unwrap_or(declaration);
            definition.visit(|cur| {
                if cur.kind() == CXCursor_EnumConstantDecl {
                    let value = if is_bool {
                        cur.enum_val_boolean().map(EnumVariantValue::Boolean)
                    } else if is_signed {
                        cur.enum_val_signed().map(EnumVariantValue::Signed)
                    } else {
                        cur.enum_val_unsigned().map(EnumVariantValue::Unsigned)
                    };
                    if let Some(val) = value {
                        let name = cur.spelling();
                        let annotations = Annotations::new(&cur);
                        let custom_behavior = ctx
                            .opts()
                            .last_callback(|callbacks| callbacks.enum_variant_behavior(type_name, &name, val))
                            .or_else(|| {
                                let annotations = annotations.as_ref()?;
                                if annotations.hide() {
                                    Some(EnumVariantCustomBehavior::Hide)
                                } else if annotations.constify_enum_variant() {
                                    Some(EnumVariantCustomBehavior::Constify)
                                } else {
                                    None
                                }
                            });
                        let new_name = ctx
                            .opts()
                            .last_callback(|callbacks| callbacks.enum_variant_name(type_name, &name, val))
                            .or_else(|| annotations.as_ref()?.use_instead_of()?.last().cloned())
                            .unwrap_or_else(|| name.clone());
                        let comment = cur.raw_comment();
                        variants.push(EnumVariant::new(new_name, name, comment, val, custom_behavior));
                    }
                }
                CXChildVisit_Continue
            });
            Ok(Enum::new(repr, variants))
        }
        fn is_matching_enum(&self, ctx: &BindgenContext, enums: &RegexSet, item: &Item) -> bool {
            let path = item.path_for_allowlisting(ctx);
            let enum_ty = item.expect_type();
            if enums.matches(path[1..].join("::")) {
                return true;
            }
            if enum_ty.name().is_some() {
                return false;
            }
            self.variants().iter().any(|v| enums.matches(v.name()))
        }
        pub fn computed_enum_variation(&self, ctx: &BindgenContext, item: &Item) -> EnumVariation {
            if self.is_matching_enum(ctx, &ctx.opts().constified_enum_modules, item) {
                EnumVariation::ModuleConsts
            } else if self.is_matching_enum(ctx, &ctx.opts().bitfield_enums, item) {
                EnumVariation::NewType {
                    is_bitfield: true,
                    is_global: false,
                }
            } else if self.is_matching_enum(ctx, &ctx.opts().newtype_enums, item) {
                EnumVariation::NewType {
                    is_bitfield: false,
                    is_global: false,
                }
            } else if self.is_matching_enum(ctx, &ctx.opts().newtype_global_enums, item) {
                EnumVariation::NewType {
                    is_bitfield: false,
                    is_global: true,
                }
            } else if self.is_matching_enum(ctx, &ctx.opts().rustified_enums, item) {
                EnumVariation::Rust { non_exhaustive: false }
            } else if self.is_matching_enum(ctx, &ctx.opts().rustified_non_exhaustive_enums, item) {
                EnumVariation::Rust { non_exhaustive: true }
            } else if self.is_matching_enum(ctx, &ctx.opts().constified_enums, item) {
                EnumVariation::Consts
            } else {
                ctx.opts().default_enum_style
            }
        }
    }
    #[derive(Debug)]
    pub struct EnumVariant {
        name: String,
        name_for_allowlisting: String,
        comment: Option<String>,
        val: EnumVariantValue,
        custom_behavior: Option<EnumVariantCustomBehavior>,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum EnumVariantValue {
        Boolean(bool),
        Signed(i64),
        Unsigned(u64),
    }
    impl EnumVariant {
        pub fn new(
            name: String,
            name_for_allowlisting: String,
            comment: Option<String>,
            val: EnumVariantValue,
            custom_behavior: Option<EnumVariantCustomBehavior>,
        ) -> Self {
            EnumVariant {
                name,
                name_for_allowlisting,
                comment,
                val,
                custom_behavior,
            }
        }
        pub fn name(&self) -> &str {
            &self.name
        }
        pub fn name_for_allowlisting(&self) -> &str {
            &self.name_for_allowlisting
        }
        pub fn val(&self) -> EnumVariantValue {
            self.val
        }
        pub fn comment(&self) -> Option<&str> {
            self.comment.as_deref()
        }
        pub fn force_constification(&self) -> bool {
            self.custom_behavior
                .map_or(false, |b| b == EnumVariantCustomBehavior::Constify)
        }
        pub fn hidden(&self) -> bool {
            self.custom_behavior
                .map_or(false, |b| b == EnumVariantCustomBehavior::Hide)
        }
    }
}
pub mod function;
pub mod int {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum IntKind {
        Bool,
        SChar,
        UChar,
        WChar,
        Char { is_signed: bool },
        Short,
        UShort,
        Int,
        UInt,
        Long,
        ULong,
        LongLong,
        ULongLong,
        I8,
        U8,
        I16,
        U16,
        I32,
        U32,
        I64,
        U64,
        I128,
        U128,
        Custom { name: &'static str, is_signed: bool },
    }
    impl IntKind {
        pub fn is_signed(&self) -> bool {
            use self::IntKind::*;
            match *self {
                Bool | UChar | UShort | UInt | ULong | ULongLong | U8 | U16 | WChar | U32 | U64 | U128 => false,
                SChar | Short | Int | Long | LongLong | I8 | I16 | I32 | I64 | I128 => true,
                Char { is_signed } => is_signed,
                Custom { is_signed, .. } => is_signed,
            }
        }
        pub fn known_size(&self) -> Option<usize> {
            use self::IntKind::*;
            Some(match *self {
                Bool | UChar | SChar | U8 | I8 | Char { .. } => 1,
                U16 | I16 => 2,
                U32 | I32 => 4,
                U64 | I64 => 8,
                I128 | U128 => 16,
                _ => return None,
            })
        }
        pub fn signedness_matches(&self, x: i64) -> bool {
            x >= 0 || self.is_signed()
        }
    }
}
pub mod item;
pub mod item_kind {
    use super::context::BindgenContext;
    use super::dot::DotAttrs;
    use super::function::Function;
    use super::module::Module;
    use super::ty::Type;
    use super::var::Var;
    use std::io;
    #[derive(Debug)]
    pub enum ItemKind {
        Module(Module),
        Type(Type),
        Function(Function),
        Var(Var),
    }
    impl ItemKind {
        pub fn as_module(&self) -> Option<&Module> {
            match *self {
                ItemKind::Module(ref x) => Some(x),
                _ => None,
            }
        }
        pub fn kind_name(&self) -> &'static str {
            match *self {
                ItemKind::Module(..) => "Module",
                ItemKind::Type(..) => "Type",
                ItemKind::Function(..) => "Function",
                ItemKind::Var(..) => "Var",
            }
        }
        pub fn is_module(&self) -> bool {
            self.as_module().is_some()
        }
        pub fn as_function(&self) -> Option<&Function> {
            match *self {
                ItemKind::Function(ref x) => Some(x),
                _ => None,
            }
        }
        pub fn is_function(&self) -> bool {
            self.as_function().is_some()
        }
        pub fn expect_function(&self) -> &Function {
            self.as_function().expect("Not a function")
        }
        pub fn as_type(&self) -> Option<&Type> {
            match *self {
                ItemKind::Type(ref x) => Some(x),
                _ => None,
            }
        }
        pub fn as_type_mut(&mut self) -> Option<&mut Type> {
            match *self {
                ItemKind::Type(ref mut x) => Some(x),
                _ => None,
            }
        }
        pub fn is_type(&self) -> bool {
            self.as_type().is_some()
        }
        pub fn expect_type(&self) -> &Type {
            self.as_type().expect("Not a type")
        }
        pub fn as_var(&self) -> Option<&Var> {
            match *self {
                ItemKind::Var(ref v) => Some(v),
                _ => None,
            }
        }
        pub fn is_var(&self) -> bool {
            self.as_var().is_some()
        }
    }
    impl DotAttrs for ItemKind {
        fn dot_attrs<W>(&self, ctx: &BindgenContext, y: &mut W) -> io::Result<()>
        where
            W: io::Write,
        {
            writeln!(y, "<tr><td>kind</td><td>{}</td></tr>", self.kind_name())?;
            match *self {
                ItemKind::Module(ref x) => x.dot_attrs(ctx, y),
                ItemKind::Type(ref x) => x.dot_attrs(ctx, y),
                ItemKind::Function(ref x) => x.dot_attrs(ctx, y),
                ItemKind::Var(ref x) => x.dot_attrs(ctx, y),
            }
        }
    }
}
pub mod layout {
    use super::derive::YDerive;
    use super::ty::{TyKind, Type, RUST_DERIVE_IN_ARRAY_LIMIT};
    use crate::clang;
    use crate::ir::context::BindgenContext;
    use std::cmp;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Layout {
        pub size: usize,
        pub align: usize,
        pub packed: bool,
    }
    #[test]
    fn test_layout_for_size() {
        use std::mem;
        let ptr_size = mem::size_of::<*mut ()>();
        assert_eq!(
            Layout::for_size_internal(ptr_size, ptr_size),
            Layout::new(ptr_size, ptr_size)
        );
        assert_eq!(
            Layout::for_size_internal(ptr_size, 3 * ptr_size),
            Layout::new(3 * ptr_size, ptr_size)
        );
    }
    impl Layout {
        pub fn known_type_for_size(ctx: &BindgenContext, size: usize) -> Option<&'static str> {
            Some(match size {
                16 => "u128",
                8 => "u64",
                4 => "u32",
                2 => "u16",
                1 => "u8",
                _ => return None,
            })
        }
        pub fn new(size: usize, align: usize) -> Self {
            Layout {
                size,
                align,
                packed: false,
            }
        }
        fn for_size_internal(ptr_size: usize, size: usize) -> Self {
            let mut align = 2;
            while size % align == 0 && align <= ptr_size {
                align *= 2;
            }
            Layout {
                size,
                align: align / 2,
                packed: false,
            }
        }
        pub fn for_size(ctx: &BindgenContext, size: usize) -> Self {
            Self::for_size_internal(ctx.target_pointer_size(), size)
        }
        pub fn opaque(&self) -> Opaque {
            Opaque(*self)
        }
    }
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Opaque(pub Layout);
    impl Opaque {
        pub fn from_clang_ty(ty: &clang::Type, ctx: &BindgenContext) -> Type {
            let layout = Layout::new(ty.size(ctx), ty.align(ctx));
            let ty_kind = TyKind::Opaque;
            let is_const = ty.is_const();
            Type::new(None, Some(layout), ty_kind, is_const)
        }
        pub fn known_rust_type_for_array(&self, ctx: &BindgenContext) -> Option<&'static str> {
            Layout::known_type_for_size(ctx, self.0.align)
        }
        pub fn array_size(&self, ctx: &BindgenContext) -> Option<usize> {
            if self.known_rust_type_for_array(ctx).is_some() {
                Some(self.0.size / cmp::max(self.0.align, 1))
            } else {
                None
            }
        }
        pub fn array_size_within_derive_limit(&self, ctx: &BindgenContext) -> YDerive {
            if self
                .array_size(ctx)
                .map_or(false, |size| size <= RUST_DERIVE_IN_ARRAY_LIMIT)
            {
                YDerive::Yes
            } else {
                YDerive::Manually
            }
        }
    }
}
pub mod module {
    use super::context::BindgenContext;
    use super::dot::DotAttrs;
    use super::item::ItemSet;
    use crate::clang;
    use crate::parse;
    use crate::parse_one;
    use std::io;
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum ModKind {
        Normal,
        Inline,
    }
    #[derive(Clone, Debug)]
    pub struct Module {
        name: Option<String>,
        kind: ModKind,
        children: ItemSet,
    }
    impl Module {
        pub fn new(name: Option<String>, kind: ModKind) -> Self {
            Module {
                name,
                kind,
                children: ItemSet::new(),
            }
        }
        pub fn name(&self) -> Option<&str> {
            self.name.as_deref()
        }
        pub fn children_mut(&mut self) -> &mut ItemSet {
            &mut self.children
        }
        pub fn children(&self) -> &ItemSet {
            &self.children
        }
        pub fn is_inline(&self) -> bool {
            self.kind == ModKind::Inline
        }
    }
    impl DotAttrs for Module {
        fn dot_attrs<W>(&self, _: &BindgenContext, y: &mut W) -> io::Result<()>
        where
            W: io::Write,
        {
            writeln!(y, "<tr><td>ModuleKind</td><td>{:?}</td></tr>", self.kind)
        }
    }
    impl parse::SubItem for Module {
        fn parse(cur: clang::Cursor, ctx: &mut BindgenContext) -> Result<parse::Result<Self>, parse::Error> {
            match cur.kind() {
                clang_lib::CXCursor_Namespace => {
                    let id = ctx.module(cur);
                    ctx.with_module(id, |ctx2| cur.visit(|cur2| parse_one(ctx2, cur2, Some(id.into()))));
                    Ok(parse::Result::AlreadyResolved(id.into()))
                },
                _ => Err(parse::Error::Continue),
            }
        }
    }
}
pub mod template {
    use super::context::{BindgenContext, ItemId, TypeId};
    use super::item::{Ancestors, IsOpaque, Item};
    use super::traversal::{EdgeKind, Trace, Tracer};
    use crate::clang;
    pub trait TemplParams: Sized {
        fn self_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>;
        fn num_self_template_params(&self, ctx: &BindgenContext) -> usize {
            self.self_template_params(ctx).len()
        }
        fn all_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>
        where
            Self: Ancestors,
        {
            let mut ys: Vec<_> = self.ancestors(ctx).collect();
            ys.reverse();
            ys.into_iter()
                .flat_map(|id| id.self_template_params(ctx).into_iter())
                .collect()
        }
        fn used_template_params(&self, ctx: &BindgenContext) -> Vec<TypeId>
        where
            Self: AsRef<ItemId>,
        {
            assert!(
                ctx.in_codegen_phase(),
                "template parameter usage is not computed until codegen"
            );
            let id = *self.as_ref();
            ctx.resolve_item(id)
                .all_template_params(ctx)
                .into_iter()
                .filter(|p| ctx.uses_template_parameter(id, *p))
                .collect()
        }
    }
    pub trait AsTemplParam {
        type Extra;
        fn as_template_param(&self, ctx: &BindgenContext, extra: &Self::Extra) -> Option<TypeId>;
        fn is_template_param(&self, ctx: &BindgenContext, extra: &Self::Extra) -> bool {
            self.as_template_param(ctx, extra).is_some()
        }
    }
    #[derive(Clone, Debug)]
    pub struct TemplInstantiation {
        def: TypeId,
        args: Vec<TypeId>,
    }
    impl TemplInstantiation {
        pub fn new<I>(def: TypeId, args: I) -> TemplInstantiation
        where
            I: IntoIterator<Item = TypeId>,
        {
            TemplInstantiation {
                def,
                args: args.into_iter().collect(),
            }
        }
        pub fn template_definition(&self) -> TypeId {
            self.def
        }
        pub fn template_arguments(&self) -> &[TypeId] {
            &self.args[..]
        }
        pub fn from_ty(ty: &clang::Type, ctx: &mut BindgenContext) -> Option<TemplInstantiation> {
            use clang_lib::*;
            let template_args = ty
                .template_args()
                .map_or(vec![], |args| match ty.canonical_type().template_args() {
                    Some(canonical_args) => {
                        let arg_count = args.len();
                        args.chain(canonical_args.skip(arg_count))
                            .filter(|t| t.kind() != CXType_Invalid)
                            .map(|t| Item::from_ty_or_ref(t, t.declaration(), None, ctx))
                            .collect()
                    },
                    None => args
                        .filter(|t| t.kind() != CXType_Invalid)
                        .map(|t| Item::from_ty_or_ref(t, t.declaration(), None, ctx))
                        .collect(),
                });
            let decl = ty.declaration();
            let def = if decl.kind() == CXCursor_TypeAliasTemplateDecl {
                Some(decl)
            } else {
                decl.specialized().or_else(|| {
                    let mut template_ref = None;
                    ty.declaration().visit(|child| {
                        if child.kind() == CXCursor_TemplateRef {
                            template_ref = Some(child);
                            return CXVisit_Break;
                        }
                        CXChildVisit_Recurse
                    });
                    template_ref.and_then(|cur| cur.referenced())
                })
            };
            let def = match def {
                Some(x) => x,
                None => {
                    if !ty.declaration().is_builtin() {
                        warn!(
                            "Could not find template definition for template \
                         instantiation"
                        );
                    }
                    return None;
                },
            };
            let template_definition = Item::from_ty_or_ref(def.cur_type(), def, None, ctx);
            Some(TemplInstantiation::new(template_definition, template_args))
        }
    }
    impl IsOpaque for TemplInstantiation {
        type Extra = Item;
        fn is_opaque(&self, ctx: &BindgenContext, it: &Item) -> bool {
            if self.template_definition().is_opaque(ctx, &()) {
                return true;
            }
            let mut path = it.path_for_allowlisting(ctx).clone();
            let args: Vec<_> = self
                .template_arguments()
                .iter()
                .map(|arg| {
                    let arg_path = ctx.resolve_item(*arg).path_for_allowlisting(ctx);
                    arg_path[1..].join("::")
                })
                .collect();
            {
                let last = path.last_mut().unwrap();
                last.push('<');
                last.push_str(&args.join(", "));
                last.push('>');
            }
            ctx.opaque_by_name(&path)
        }
    }
    impl Trace for TemplInstantiation {
        type Extra = ();
        fn trace<T>(&self, _ctx: &BindgenContext, tracer: &mut T, _: &())
        where
            T: Tracer,
        {
            tracer.visit_kind(self.def.into(), EdgeKind::TemplateDeclaration);
            for arg in self.template_arguments() {
                tracer.visit_kind(arg.into(), EdgeKind::TemplateArgument);
            }
        }
    }
}
pub mod traversal;
pub mod ty;
pub mod var;
