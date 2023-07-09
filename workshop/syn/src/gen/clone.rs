#![allow(clippy::clone_on_copy, clippy::expl_impl_clone_on_copy)]
use crate::*;
impl Clone for Abi {
    fn clone(&self) -> Self {
        Abi {
            extern_: self.extern_.clone(),
            name: self.name.clone(),
        }
    }
}
impl Clone for AngledArgs {
    fn clone(&self) -> Self {
        AngledArgs {
            colon2: self.colon2.clone(),
            lt: self.lt.clone(),
            args: self.args.clone(),
            gt: self.gt.clone(),
        }
    }
}
impl Clone for Arm {
    fn clone(&self) -> Self {
        Arm {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            guard: self.guard.clone(),
            fat_arrow: self.fat_arrow.clone(),
            body: self.body.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Clone for AssocConst {
    fn clone(&self) -> Self {
        AssocConst {
            ident: self.ident.clone(),
            gnrs: self.gnrs.clone(),
            eq: self.eq.clone(),
            val: self.val.clone(),
        }
    }
}
impl Clone for AssocType {
    fn clone(&self) -> Self {
        AssocType {
            ident: self.ident.clone(),
            gnrs: self.gnrs.clone(),
            eq: self.eq.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Copy for AttrStyle {}
impl Clone for AttrStyle {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for Attribute {
    fn clone(&self) -> Self {
        Attribute {
            pound: self.pound.clone(),
            style: self.style.clone(),
            bracket: self.bracket.clone(),
            meta: self.meta.clone(),
        }
    }
}
impl Clone for ty::BareFnArg {
    fn clone(&self) -> Self {
        ty::BareFnArg {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Clone for ty::BareVari {
    fn clone(&self) -> Self {
        ty::BareVari {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Copy for BinOp {}
impl Clone for BinOp {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for Block {
    fn clone(&self) -> Self {
        Block {
            brace: self.brace.clone(),
            stmts: self.stmts.clone(),
        }
    }
}
impl Clone for BoundLifetimes {
    fn clone(&self) -> Self {
        BoundLifetimes {
            for_: self.for_.clone(),
            lt: self.lt.clone(),
            lifes: self.lifes.clone(),
            gt: self.gt.clone(),
        }
    }
}
impl Clone for ConstParam {
    fn clone(&self) -> Self {
        ConstParam {
            attrs: self.attrs.clone(),
            const_: self.const_.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            default: self.default.clone(),
        }
    }
}
impl Clone for Constraint {
    fn clone(&self) -> Self {
        Constraint {
            ident: self.ident.clone(),
            gnrs: self.gnrs.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for Data {
    fn clone(&self) -> Self {
        match self {
            Data::Struct(v0) => Data::Struct(v0.clone()),
            Data::Enum(v0) => Data::Enum(v0.clone()),
            Data::Union(v0) => Data::Union(v0.clone()),
        }
    }
}
impl Clone for DataEnum {
    fn clone(&self) -> Self {
        DataEnum {
            enum_: self.enum_.clone(),
            brace: self.brace.clone(),
            variants: self.variants.clone(),
        }
    }
}
impl Clone for DataStruct {
    fn clone(&self) -> Self {
        DataStruct {
            struct_: self.struct_.clone(),
            fields: self.fields.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for DataUnion {
    fn clone(&self) -> Self {
        DataUnion {
            union_: self.union_.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for DeriveInput {
    fn clone(&self) -> Self {
        DeriveInput {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            data: self.data.clone(),
        }
    }
}
impl Clone for Expr {
    fn clone(&self) -> Self {
        match self {
            Expr::Array(v0) => Expr::Array(v0.clone()),
            Expr::Assign(v0) => Expr::Assign(v0.clone()),
            Expr::Async(v0) => Expr::Async(v0.clone()),
            Expr::Await(v0) => Expr::Await(v0.clone()),
            Expr::Binary(v0) => Expr::Binary(v0.clone()),
            Expr::Block(v0) => Expr::Block(v0.clone()),
            Expr::Break(v0) => Expr::Break(v0.clone()),
            Expr::Call(v0) => Expr::Call(v0.clone()),
            Expr::Cast(v0) => Expr::Cast(v0.clone()),
            Expr::Closure(v0) => Expr::Closure(v0.clone()),
            Expr::Const(v0) => Expr::Const(v0.clone()),
            Expr::Continue(v0) => Expr::Continue(v0.clone()),
            Expr::Field(v0) => Expr::Field(v0.clone()),
            Expr::ForLoop(v0) => Expr::ForLoop(v0.clone()),
            Expr::Group(v0) => Expr::Group(v0.clone()),
            Expr::If(v0) => Expr::If(v0.clone()),
            Expr::Index(v0) => Expr::Index(v0.clone()),
            Expr::Infer(v0) => Expr::Infer(v0.clone()),
            Expr::Let(v0) => Expr::Let(v0.clone()),
            Expr::Lit(v0) => Expr::Lit(v0.clone()),
            Expr::Loop(v0) => Expr::Loop(v0.clone()),
            Expr::Macro(v0) => Expr::Macro(v0.clone()),
            Expr::Match(v0) => Expr::Match(v0.clone()),
            Expr::MethodCall(v0) => Expr::MethodCall(v0.clone()),
            Expr::Paren(v0) => Expr::Paren(v0.clone()),
            Expr::Path(v0) => Expr::Path(v0.clone()),
            Expr::Range(v0) => Expr::Range(v0.clone()),
            Expr::Reference(v0) => Expr::Reference(v0.clone()),
            Expr::Repeat(v0) => Expr::Repeat(v0.clone()),
            Expr::Return(v0) => Expr::Return(v0.clone()),
            Expr::Struct(v0) => Expr::Struct(v0.clone()),
            Expr::Try(v0) => Expr::Try(v0.clone()),
            Expr::TryBlock(v0) => Expr::TryBlock(v0.clone()),
            Expr::Tuple(v0) => Expr::Tuple(v0.clone()),
            Expr::Unary(v0) => Expr::Unary(v0.clone()),
            Expr::Unsafe(v0) => Expr::Unsafe(v0.clone()),
            Expr::Verbatim(v0) => Expr::Verbatim(v0.clone()),
            Expr::While(v0) => Expr::While(v0.clone()),
            Expr::Yield(v0) => Expr::Yield(v0.clone()),
            #[cfg(not(feature = "full"))]
            _ => unreachable!(),
        }
    }
}
impl Clone for expr::Array {
    fn clone(&self) -> Self {
        expr::Array {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for expr::Assign {
    fn clone(&self) -> Self {
        expr::Assign {
            attrs: self.attrs.clone(),
            left: self.left.clone(),
            eq: self.eq.clone(),
            right: self.right.clone(),
        }
    }
}
impl Clone for expr::Async {
    fn clone(&self) -> Self {
        expr::Async {
            attrs: self.attrs.clone(),
            async_: self.async_.clone(),
            move_: self.move_.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for expr::Await {
    fn clone(&self) -> Self {
        expr::Await {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            dot: self.dot.clone(),
            await_: self.await_.clone(),
        }
    }
}
impl Clone for expr::Binary {
    fn clone(&self) -> Self {
        expr::Binary {
            attrs: self.attrs.clone(),
            left: self.left.clone(),
            op: self.op.clone(),
            right: self.right.clone(),
        }
    }
}
impl Clone for expr::Block {
    fn clone(&self) -> Self {
        expr::Block {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for expr::Break {
    fn clone(&self) -> Self {
        expr::Break {
            attrs: self.attrs.clone(),
            break_: self.break_.clone(),
            label: self.label.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Call {
    fn clone(&self) -> Self {
        expr::Call {
            attrs: self.attrs.clone(),
            func: self.func.clone(),
            paren: self.paren.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for expr::Cast {
    fn clone(&self) -> Self {
        expr::Cast {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            as_: self.as_.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for expr::Closure {
    fn clone(&self) -> Self {
        expr::Closure {
            attrs: self.attrs.clone(),
            lifes: self.lifes.clone(),
            const_: self.const_.clone(),
            static_: self.static_.clone(),
            async_: self.async_.clone(),
            move_: self.move_.clone(),
            or1: self.or1.clone(),
            inputs: self.inputs.clone(),
            or2: self.or2.clone(),
            ret: self.ret.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for expr::Const {
    fn clone(&self) -> Self {
        expr::Const {
            attrs: self.attrs.clone(),
            const_: self.const_.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for expr::Continue {
    fn clone(&self) -> Self {
        expr::Continue {
            attrs: self.attrs.clone(),
            continue_: self.continue_.clone(),
            label: self.label.clone(),
        }
    }
}
impl Clone for expr::Field {
    fn clone(&self) -> Self {
        expr::Field {
            attrs: self.attrs.clone(),
            base: self.base.clone(),
            dot: self.dot.clone(),
            memb: self.memb.clone(),
        }
    }
}
impl Clone for expr::ForLoop {
    fn clone(&self) -> Self {
        expr::ForLoop {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            for_: self.for_.clone(),
            pat: self.pat.clone(),
            in_: self.in_.clone(),
            expr: self.expr.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for expr::Group {
    fn clone(&self) -> Self {
        expr::Group {
            attrs: self.attrs.clone(),
            group: self.group.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::If {
    fn clone(&self) -> Self {
        expr::If {
            attrs: self.attrs.clone(),
            if_: self.if_.clone(),
            cond: self.cond.clone(),
            then_branch: self.then_branch.clone(),
            else_branch: self.else_branch.clone(),
        }
    }
}
impl Clone for expr::Index {
    fn clone(&self) -> Self {
        expr::Index {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            bracket: self.bracket.clone(),
            index: self.index.clone(),
        }
    }
}
impl Clone for expr::Infer {
    fn clone(&self) -> Self {
        expr::Infer {
            attrs: self.attrs.clone(),
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for expr::Let {
    fn clone(&self) -> Self {
        expr::Let {
            attrs: self.attrs.clone(),
            let_: self.let_.clone(),
            pat: self.pat.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Lit {
    fn clone(&self) -> Self {
        expr::Lit {
            attrs: self.attrs.clone(),
            lit: self.lit.clone(),
        }
    }
}
impl Clone for expr::Loop {
    fn clone(&self) -> Self {
        expr::Loop {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            loop_: self.loop_.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for expr::Mac {
    fn clone(&self) -> Self {
        expr::Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
        }
    }
}
impl Clone for expr::Match {
    fn clone(&self) -> Self {
        expr::Match {
            attrs: self.attrs.clone(),
            match_: self.match_.clone(),
            expr: self.expr.clone(),
            brace: self.brace.clone(),
            arms: self.arms.clone(),
        }
    }
}
impl Clone for expr::MethodCall {
    fn clone(&self) -> Self {
        expr::MethodCall {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            dot: self.dot.clone(),
            method: self.method.clone(),
            turbofish: self.turbofish.clone(),
            paren: self.paren.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for expr::Paren {
    fn clone(&self) -> Self {
        expr::Paren {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Path {
    fn clone(&self) -> Self {
        expr::Path {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for expr::Range {
    fn clone(&self) -> Self {
        expr::Range {
            attrs: self.attrs.clone(),
            beg: self.beg.clone(),
            limits: self.limits.clone(),
            end: self.end.clone(),
        }
    }
}
impl Clone for expr::Ref {
    fn clone(&self) -> Self {
        expr::Ref {
            attrs: self.attrs.clone(),
            and: self.and.clone(),
            mut_: self.mut_.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Repeat {
    fn clone(&self) -> Self {
        expr::Repeat {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
            len: self.len.clone(),
        }
    }
}
impl Clone for expr::Return {
    fn clone(&self) -> Self {
        expr::Return {
            attrs: self.attrs.clone(),
            return_: self.return_.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Struct {
    fn clone(&self) -> Self {
        expr::Struct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace: self.brace.clone(),
            fields: self.fields.clone(),
            dot2: self.dot2.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Clone for expr::Try {
    fn clone(&self) -> Self {
        expr::Try {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            question: self.question.clone(),
        }
    }
}
impl Clone for expr::TryBlock {
    fn clone(&self) -> Self {
        expr::TryBlock {
            attrs: self.attrs.clone(),
            try_: self.try_.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for expr::Tuple {
    fn clone(&self) -> Self {
        expr::Tuple {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for expr::Unary {
    fn clone(&self) -> Self {
        expr::Unary {
            attrs: self.attrs.clone(),
            op: self.op.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for expr::Unsafe {
    fn clone(&self) -> Self {
        expr::Unsafe {
            attrs: self.attrs.clone(),
            unsafe_: self.unsafe_.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for expr::While {
    fn clone(&self) -> Self {
        expr::While {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            while_: self.while_.clone(),
            cond: self.cond.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for expr::Yield {
    fn clone(&self) -> Self {
        expr::Yield {
            attrs: self.attrs.clone(),
            yield_: self.yield_.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for Field {
    fn clone(&self) -> Self {
        Field {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            mutability: self.mutability.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for FieldMut {
    fn clone(&self) -> Self {
        match self {
            FieldMut::None => FieldMut::None,
        }
    }
}
impl Clone for patt::Field {
    fn clone(&self) -> Self {
        patt::Field {
            attrs: self.attrs.clone(),
            member: self.member.clone(),
            colon: self.colon.clone(),
            patt: self.patt.clone(),
        }
    }
}
impl Clone for FieldValue {
    fn clone(&self) -> Self {
        FieldValue {
            attrs: self.attrs.clone(),
            member: self.member.clone(),
            colon: self.colon.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for Fields {
    fn clone(&self) -> Self {
        match self {
            Fields::Named(v0) => Fields::Named(v0.clone()),
            Fields::Unnamed(v0) => Fields::Unnamed(v0.clone()),
            Fields::Unit => Fields::Unit,
        }
    }
}
impl Clone for FieldsNamed {
    fn clone(&self) -> Self {
        FieldsNamed {
            brace: self.brace.clone(),
            named: self.named.clone(),
        }
    }
}
impl Clone for FieldsUnnamed {
    fn clone(&self) -> Self {
        FieldsUnnamed {
            paren: self.paren.clone(),
            unnamed: self.unnamed.clone(),
        }
    }
}
impl Clone for File {
    fn clone(&self) -> Self {
        File {
            shebang: self.shebang.clone(),
            attrs: self.attrs.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for FnArg {
    fn clone(&self) -> Self {
        match self {
            FnArg::Receiver(v0) => FnArg::Receiver(v0.clone()),
            FnArg::Typed(v0) => FnArg::Typed(v0.clone()),
        }
    }
}
impl Clone for ForeignItem {
    fn clone(&self) -> Self {
        match self {
            ForeignItem::Fn(v0) => ForeignItem::Fn(v0.clone()),
            ForeignItem::Static(v0) => ForeignItem::Static(v0.clone()),
            ForeignItem::Type(v0) => ForeignItem::Type(v0.clone()),
            ForeignItem::Macro(v0) => ForeignItem::Macro(v0.clone()),
            ForeignItem::Verbatim(v0) => ForeignItem::Verbatim(v0.clone()),
        }
    }
}
impl Clone for ForeignItemFn {
    fn clone(&self) -> Self {
        ForeignItemFn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: self.sig.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ForeignItemMacro {
    fn clone(&self) -> Self {
        ForeignItemMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ForeignItemStatic {
    fn clone(&self) -> Self {
        ForeignItemStatic {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            static_: self.static_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ForeignItemType {
    fn clone(&self) -> Self {
        ForeignItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type: self.type.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for Arg {
    fn clone(&self) -> Self {
        match self {
            Arg::Lifetime(v0) => Arg::Lifetime(v0.clone()),
            Arg::Type(v0) => Arg::Type(v0.clone()),
            Arg::Const(v0) => Arg::Const(v0.clone()),
            Arg::AssocType(v0) => Arg::AssocType(v0.clone()),
            Arg::AssocConst(v0) => Arg::AssocConst(v0.clone()),
            Arg::Constraint(v0) => Arg::Constraint(v0.clone()),
        }
    }
}
impl Clone for GenericParam {
    fn clone(&self) -> Self {
        match self {
            GenericParam::Lifetime(v0) => GenericParam::Lifetime(v0.clone()),
            GenericParam::Type(v0) => GenericParam::Type(v0.clone()),
            GenericParam::Const(v0) => GenericParam::Const(v0.clone()),
        }
    }
}
impl Clone for Generics {
    fn clone(&self) -> Self {
        Generics {
            lt: self.lt.clone(),
            params: self.params.clone(),
            gt: self.gt.clone(),
            where_: self.where_.clone(),
        }
    }
}
impl Clone for ImplItem {
    fn clone(&self) -> Self {
        match self {
            ImplItem::Const(v0) => ImplItem::Const(v0.clone()),
            ImplItem::Fn(v0) => ImplItem::Fn(v0.clone()),
            ImplItem::Type(v0) => ImplItem::Type(v0.clone()),
            ImplItem::Macro(v0) => ImplItem::Macro(v0.clone()),
            ImplItem::Verbatim(v0) => ImplItem::Verbatim(v0.clone()),
        }
    }
}
impl Clone for ImplItemConst {
    fn clone(&self) -> Self {
        ImplItemConst {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default_: self.default_.clone(),
            const_: self.const_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ImplItemFn {
    fn clone(&self) -> Self {
        ImplItemFn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default_: self.default_.clone(),
            sig: self.sig.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ImplItemMacro {
    fn clone(&self) -> Self {
        ImplItemMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ImplItemType {
    fn clone(&self) -> Self {
        ImplItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default_: self.default_.clone(),
            type: self.type.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ImplRestriction {
    fn clone(&self) -> Self {
        match *self {}
    }
}
impl Clone for Index {
    fn clone(&self) -> Self {
        Index {
            index: self.index.clone(),
            span: self.span.clone(),
        }
    }
}
impl Clone for Item {
    fn clone(&self) -> Self {
        match self {
            Item::Const(v0) => Item::Const(v0.clone()),
            Item::Enum(v0) => Item::Enum(v0.clone()),
            Item::ExternCrate(v0) => Item::ExternCrate(v0.clone()),
            Item::Fn(v0) => Item::Fn(v0.clone()),
            Item::ForeignMod(v0) => Item::ForeignMod(v0.clone()),
            Item::Impl(v0) => Item::Impl(v0.clone()),
            Item::Macro(v0) => Item::Macro(v0.clone()),
            Item::Mod(v0) => Item::Mod(v0.clone()),
            Item::Static(v0) => Item::Static(v0.clone()),
            Item::Struct(v0) => Item::Struct(v0.clone()),
            Item::Trait(v0) => Item::Trait(v0.clone()),
            Item::TraitAlias(v0) => Item::TraitAlias(v0.clone()),
            Item::Type(v0) => Item::Type(v0.clone()),
            Item::Union(v0) => Item::Union(v0.clone()),
            Item::Use(v0) => Item::Use(v0.clone()),
            Item::Verbatim(v0) => Item::Verbatim(v0.clone()),
        }
    }
}
impl Clone for ItemConst {
    fn clone(&self) -> Self {
        ItemConst {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            const_: self.const_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemEnum {
    fn clone(&self) -> Self {
        ItemEnum {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            enum_: self.enum_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            brace: self.brace.clone(),
            variants: self.variants.clone(),
        }
    }
}
impl Clone for ItemExternCrate {
    fn clone(&self) -> Self {
        ItemExternCrate {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            extern_: self.extern_.clone(),
            crate_: self.crate_.clone(),
            ident: self.ident.clone(),
            rename: self.rename.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemFn {
    fn clone(&self) -> Self {
        ItemFn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: self.sig.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ItemForeignMod {
    fn clone(&self) -> Self {
        ItemForeignMod {
            attrs: self.attrs.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for ItemImpl {
    fn clone(&self) -> Self {
        ItemImpl {
            attrs: self.attrs.clone(),
            default_: self.default_.clone(),
            unsafe_: self.unsafe_.clone(),
            impl_: self.impl_.clone(),
            gens: self.gens.clone(),
            trait_: self.trait_.clone(),
            typ: self.typ.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for ItemMacro {
    fn clone(&self) -> Self {
        ItemMacro {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemMod {
    fn clone(&self) -> Self {
        ItemMod {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            mod_: self.mod_.clone(),
            ident: self.ident.clone(),
            gist: self.gist.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemStatic {
    fn clone(&self) -> Self {
        ItemStatic {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            static_: self.static_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemStruct {
    fn clone(&self) -> Self {
        ItemStruct {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            struct_: self.struct_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            fields: self.fields.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemTrait {
    fn clone(&self) -> Self {
        ItemTrait {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            auto_: self.auto_.clone(),
            restriction: self.restriction.clone(),
            trait_: self.trait_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            supertraits: self.supertraits.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for ItemTraitAlias {
    fn clone(&self) -> Self {
        ItemTraitAlias {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            trait_: self.trait_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            bounds: self.bounds.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemType {
    fn clone(&self) -> Self {
        ItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type: self.type.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemUnion {
    fn clone(&self) -> Self {
        ItemUnion {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            union_: self.union_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for ItemUse {
    fn clone(&self) -> Self {
        ItemUse {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            use_: self.use_.clone(),
            leading_colon: self.leading_colon.clone(),
            tree: self.tree.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for Label {
    fn clone(&self) -> Self {
        Label {
            name: self.name.clone(),
            colon: self.colon.clone(),
        }
    }
}
impl Clone for LifetimeParam {
    fn clone(&self) -> Self {
        LifetimeParam {
            attrs: self.attrs.clone(),
            life: self.life.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for Lit {
    fn clone(&self) -> Self {
        match self {
            Lit::Str(v0) => Lit::Str(v0.clone()),
            Lit::ByteStr(v0) => Lit::ByteStr(v0.clone()),
            Lit::Byte(v0) => Lit::Byte(v0.clone()),
            Lit::Char(v0) => Lit::Char(v0.clone()),
            Lit::Int(v0) => Lit::Int(v0.clone()),
            Lit::Float(v0) => Lit::Float(v0.clone()),
            Lit::Bool(v0) => Lit::Bool(v0.clone()),
            Lit::Verbatim(v0) => Lit::Verbatim(v0.clone()),
        }
    }
}
impl Clone for lit::Bool {
    fn clone(&self) -> Self {
        lit::Bool {
            val: self.val.clone(),
            span: self.span.clone(),
        }
    }
}
impl Clone for stmt::Local {
    fn clone(&self) -> Self {
        stmt::Local {
            attrs: self.attrs.clone(),
            let_: self.let_.clone(),
            pat: self.pat.clone(),
            init: self.init.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for stmt::LocalInit {
    fn clone(&self) -> Self {
        stmt::LocalInit {
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            diverge: self.diverge.clone(),
        }
    }
}
impl Clone for Macro {
    fn clone(&self) -> Self {
        Macro {
            path: self.path.clone(),
            bang: self.bang.clone(),
            delim: self.delim.clone(),
            toks: self.toks.clone(),
        }
    }
}
impl Clone for MacroDelim {
    fn clone(&self) -> Self {
        match self {
            MacroDelim::Paren(v0) => MacroDelim::Paren(v0.clone()),
            MacroDelim::Brace(v0) => MacroDelim::Brace(v0.clone()),
            MacroDelim::Bracket(v0) => MacroDelim::Bracket(v0.clone()),
        }
    }
}
impl Clone for Member {
    fn clone(&self) -> Self {
        match self {
            Member::Named(v0) => Member::Named(v0.clone()),
            Member::Unnamed(v0) => Member::Unnamed(v0.clone()),
        }
    }
}
impl Clone for Meta {
    fn clone(&self) -> Self {
        match self {
            Meta::Path(v0) => Meta::Path(v0.clone()),
            Meta::List(v0) => Meta::List(v0.clone()),
            Meta::NameValue(v0) => Meta::NameValue(v0.clone()),
        }
    }
}
impl Clone for MetaList {
    fn clone(&self) -> Self {
        MetaList {
            path: self.path.clone(),
            delim: self.delim.clone(),
            toks: self.toks.clone(),
        }
    }
}
impl Clone for MetaNameValue {
    fn clone(&self) -> Self {
        MetaNameValue {
            path: self.path.clone(),
            eq: self.eq.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ParenthesizedArgs {
    fn clone(&self) -> Self {
        ParenthesizedArgs {
            paren: self.paren.clone(),
            ins: self.ins.clone(),
            out: self.out.clone(),
        }
    }
}
impl Clone for patt::Patt {
    fn clone(&self) -> Self {
        match self {
            patt::Patt::Const(v0) => patt::Patt::Const(v0.clone()),
            patt::Patt::Ident(v0) => patt::Patt::Ident(v0.clone()),
            patt::Patt::Lit(v0) => patt::Patt::Lit(v0.clone()),
            patt::Patt::Mac(v0) => patt::Patt::Mac(v0.clone()),
            patt::Patt::Or(v0) => patt::Patt::Or(v0.clone()),
            patt::Patt::Paren(v0) => patt::Patt::Paren(v0.clone()),
            patt::Patt::Path(v0) => patt::Patt::Path(v0.clone()),
            patt::Patt::Range(v0) => patt::Patt::Range(v0.clone()),
            patt::Patt::Ref(v0) => patt::Patt::Ref(v0.clone()),
            patt::Patt::Rest(v0) => patt::Patt::Rest(v0.clone()),
            patt::Patt::Slice(v0) => patt::Patt::Slice(v0.clone()),
            patt::Patt::Struct(v0) => patt::Patt::Struct(v0.clone()),
            patt::Patt::Tuple(v0) => patt::Patt::Tuple(v0.clone()),
            patt::Patt::TupleStruct(v0) => patt::Patt::TupleStruct(v0.clone()),
            patt::Patt::Type(v0) => patt::Patt::Type(v0.clone()),
            patt::Patt::Verbatim(v0) => patt::Patt::Verbatim(v0.clone()),
            patt::Patt::Wild(v0) => patt::Patt::Wild(v0.clone()),
        }
    }
}
impl Clone for patt::Ident {
    fn clone(&self) -> Self {
        patt::Ident {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            sub: self.sub.clone(),
        }
    }
}
impl Clone for patt::Or {
    fn clone(&self) -> Self {
        patt::Or {
            attrs: self.attrs.clone(),
            vert: self.vert.clone(),
            cases: self.cases.clone(),
        }
    }
}
impl Clone for patt::Paren {
    fn clone(&self) -> Self {
        patt::Paren {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            patt: self.patt.clone(),
        }
    }
}
impl Clone for patt::Ref {
    fn clone(&self) -> Self {
        patt::Ref {
            attrs: self.attrs.clone(),
            and: self.and.clone(),
            mut_: self.mut_.clone(),
            patt: self.patt.clone(),
        }
    }
}
impl Clone for patt::Rest {
    fn clone(&self) -> Self {
        patt::Rest {
            attrs: self.attrs.clone(),
            dot2: self.dot2.clone(),
        }
    }
}
impl Clone for patt::Slice {
    fn clone(&self) -> Self {
        patt::Slice {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for patt::Struct {
    fn clone(&self) -> Self {
        patt::Struct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace: self.brace.clone(),
            fields: self.fields.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Clone for patt::Tuple {
    fn clone(&self) -> Self {
        patt::Tuple {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for patt::TupleStruct {
    fn clone(&self) -> Self {
        patt::TupleStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for patt::Type {
    fn clone(&self) -> Self {
        patt::Type {
            attrs: self.attrs.clone(),
            patt: self.patt.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for patt::Wild {
    fn clone(&self) -> Self {
        patt::Wild {
            attrs: self.attrs.clone(),
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for Path {
    fn clone(&self) -> Self {
        Path {
            colon: self.colon.clone(),
            segs: self.segs.clone(),
        }
    }
}
impl Clone for Args {
    fn clone(&self) -> Self {
        match self {
            Args::None => Args::None,
            Args::Angled(v0) => Args::Angled(v0.clone()),
            Args::Parenthesized(v0) => Args::Parenthesized(v0.clone()),
        }
    }
}
impl Clone for Segment {
    fn clone(&self) -> Self {
        Segment {
            ident: self.ident.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for PredLifetime {
    fn clone(&self) -> Self {
        PredLifetime {
            life: self.life.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for PredType {
    fn clone(&self) -> Self {
        PredType {
            lifes: self.lifes.clone(),
            bounded: self.bounded.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for QSelf {
    fn clone(&self) -> Self {
        QSelf {
            lt: self.lt.clone(),
            ty: self.ty.clone(),
            pos: self.pos.clone(),
            as_: self.as_.clone(),
            gt_: self.gt_.clone(),
        }
    }
}
impl Copy for RangeLimits {}
impl Clone for RangeLimits {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for Receiver {
    fn clone(&self) -> Self {
        Receiver {
            attrs: self.attrs.clone(),
            reference: self.reference.clone(),
            mut_: self.mut_.clone(),
            self_: self.self_.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for ty::Ret {
    fn clone(&self) -> Self {
        match self {
            ty::Ret::Default => ty::Ret::Default,
            ty::Ret::Type(v0, v1) => ty::Ret::Type(v0.clone(), v1.clone()),
        }
    }
}
impl Clone for Signature {
    fn clone(&self) -> Self {
        Signature {
            constness: self.constness.clone(),
            async_: self.async_.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            paren: self.paren.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Clone for StaticMut {
    fn clone(&self) -> Self {
        match self {
            StaticMut::Mut(v0) => StaticMut::Mut(v0.clone()),
            StaticMut::None => StaticMut::None,
        }
    }
}
impl Clone for stmt::Stmt {
    fn clone(&self) -> Self {
        match self {
            stmt::Stmt::stmt::Local(v0) => stmt::Stmt::stmt::Local(v0.clone()),
            stmt::Stmt::Item(v0) => stmt::Stmt::Item(v0.clone()),
            stmt::Stmt::Expr(v0, v1) => stmt::Stmt::Expr(v0.clone(), v1.clone()),
            stmt::Stmt::Mac(v0) => stmt::Stmt::Mac(v0.clone()),
        }
    }
}
impl Clone for stmt::Mac {
    fn clone(&self) -> Self {
        stmt::Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for TraitBound {
    fn clone(&self) -> Self {
        TraitBound {
            paren: self.paren.clone(),
            modifier: self.modifier.clone(),
            lifes: self.lifes.clone(),
            path: self.path.clone(),
        }
    }
}
impl Copy for TraitBoundModifier {}
impl Clone for TraitBoundModifier {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for TraitItem {
    fn clone(&self) -> Self {
        match self {
            TraitItem::Const(v0) => TraitItem::Const(v0.clone()),
            TraitItem::Fn(v0) => TraitItem::Fn(v0.clone()),
            TraitItem::Type(v0) => TraitItem::Type(v0.clone()),
            TraitItem::Macro(v0) => TraitItem::Macro(v0.clone()),
            TraitItem::Verbatim(v0) => TraitItem::Verbatim(v0.clone()),
        }
    }
}
impl Clone for TraitItemConst {
    fn clone(&self) -> Self {
        TraitItemConst {
            attrs: self.attrs.clone(),
            const_: self.const_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
            default: self.default.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for TraitItemFn {
    fn clone(&self) -> Self {
        TraitItemFn {
            attrs: self.attrs.clone(),
            sig: self.sig.clone(),
            default: self.default.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for TraitItemMacro {
    fn clone(&self) -> Self {
        TraitItemMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for TraitItemType {
    fn clone(&self) -> Self {
        TraitItemType {
            attrs: self.attrs.clone(),
            type: self.type.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
            default: self.default.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ty::Type {
    fn clone(&self) -> Self {
        match self {
            ty::Type::Array(v0) => ty::Type::Array(v0.clone()),
            ty::Type::BareFn(v0) => ty::Type::BareFn(v0.clone()),
            ty::Type::Group(v0) => ty::Type::Group(v0.clone()),
            ty::Type::Impl(v0) => ty::Type::Impl(v0.clone()),
            ty::Type::Infer(v0) => ty::Type::Infer(v0.clone()),
            ty::Type::Mac(v0) => ty::Type::Mac(v0.clone()),
            ty::Type::Never(v0) => ty::Type::Never(v0.clone()),
            ty::Type::Paren(v0) => ty::Type::Paren(v0.clone()),
            ty::Type::Path(v0) => ty::Type::Path(v0.clone()),
            ty::Type::Ptr(v0) => ty::Type::Ptr(v0.clone()),
            ty::Type::Ref(v0) => ty::Type::Ref(v0.clone()),
            ty::Type::Slice(v0) => ty::Type::Slice(v0.clone()),
            ty::Type::TraitObj(v0) => ty::Type::TraitObj(v0.clone()),
            ty::Type::Tuple(v0) => ty::Type::Tuple(v0.clone()),
            ty::Type::Verbatim(v0) => ty::Type::Verbatim(v0.clone()),
        }
    }
}
impl Clone for ty::Array {
    fn clone(&self) -> Self {
        ty::Array {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
            semi: self.semi.clone(),
            len: self.len.clone(),
        }
    }
}
impl Clone for ty::BareFn {
    fn clone(&self) -> Self {
        ty::BareFn {
            lifes: self.lifes.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            paren: self.paren.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Clone for ty::Group {
    fn clone(&self) -> Self {
        ty::Group {
            group: self.group.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for ty::Impl {
    fn clone(&self) -> Self {
        ty::Impl {
            impl_: self.impl_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for ty::Infer {
    fn clone(&self) -> Self {
        ty::Infer {
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for ty::Mac {
    fn clone(&self) -> Self {
        ty::Mac { mac: self.mac.clone() }
    }
}
impl Clone for ty::Never {
    fn clone(&self) -> Self {
        ty::Never {
            bang: self.bang.clone(),
        }
    }
}
impl Clone for TypeParam {
    fn clone(&self) -> Self {
        TypeParam {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
            eq: self.eq.clone(),
            default: self.default.clone(),
        }
    }
}
impl Clone for TypeParamBound {
    fn clone(&self) -> Self {
        match self {
            TypeParamBound::Trait(v0) => TypeParamBound::Trait(v0.clone()),
            TypeParamBound::Lifetime(v0) => TypeParamBound::Lifetime(v0.clone()),
            TypeParamBound::Verbatim(v0) => TypeParamBound::Verbatim(v0.clone()),
        }
    }
}
impl Clone for ty::Paren {
    fn clone(&self) -> Self {
        ty::Paren {
            paren: self.paren.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for ty::Path {
    fn clone(&self) -> Self {
        ty::Path {
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for ty::Ptr {
    fn clone(&self) -> Self {
        ty::Ptr {
            star: self.star.clone(),
            const_: self.const_.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for ty::Ref {
    fn clone(&self) -> Self {
        ty::Ref {
            and: self.and.clone(),
            life: self.life.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for ty::Slice {
    fn clone(&self) -> Self {
        ty::Slice {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for ty::TraitObj {
    fn clone(&self) -> Self {
        ty::TraitObj {
            dyn_: self.dyn_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for ty::Tuple {
    fn clone(&self) -> Self {
        ty::Tuple {
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Copy for UnOp {}
impl Clone for UnOp {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for UseGlob {
    fn clone(&self) -> Self {
        UseGlob {
            star: self.star.clone(),
        }
    }
}
impl Clone for UseGroup {
    fn clone(&self) -> Self {
        UseGroup {
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for UseName {
    fn clone(&self) -> Self {
        UseName {
            ident: self.ident.clone(),
        }
    }
}
impl Clone for UsePath {
    fn clone(&self) -> Self {
        UsePath {
            ident: self.ident.clone(),
            colon2: self.colon2.clone(),
            tree: self.tree.clone(),
        }
    }
}
impl Clone for UseRename {
    fn clone(&self) -> Self {
        UseRename {
            ident: self.ident.clone(),
            as_: self.as_.clone(),
            rename: self.rename.clone(),
        }
    }
}
impl Clone for UseTree {
    fn clone(&self) -> Self {
        match self {
            UseTree::Path(v0) => UseTree::Path(v0.clone()),
            UseTree::Name(v0) => UseTree::Name(v0.clone()),
            UseTree::Rename(v0) => UseTree::Rename(v0.clone()),
            UseTree::Glob(v0) => UseTree::Glob(v0.clone()),
            UseTree::Group(v0) => UseTree::Group(v0.clone()),
        }
    }
}
impl Clone for Variadic {
    fn clone(&self) -> Self {
        Variadic {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Clone for Variant {
    fn clone(&self) -> Self {
        Variant {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            fields: self.fields.clone(),
            discriminant: self.discriminant.clone(),
        }
    }
}
impl Clone for VisRestricted {
    fn clone(&self) -> Self {
        VisRestricted {
            pub_: self.pub_.clone(),
            paren: self.paren.clone(),
            in_: self.in_.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for Visibility {
    fn clone(&self) -> Self {
        match self {
            Visibility::Public(v0) => Visibility::Public(v0.clone()),
            Visibility::Restricted(v0) => Visibility::Restricted(v0.clone()),
            Visibility::Inherited => Visibility::Inherited,
        }
    }
}
impl Clone for WhereClause {
    fn clone(&self) -> Self {
        WhereClause {
            where_: self.where_.clone(),
            preds: self.preds.clone(),
        }
    }
}
impl Clone for WherePred {
    fn clone(&self) -> Self {
        match self {
            WherePred::Lifetime(v0) => WherePred::Lifetime(v0.clone()),
            WherePred::Type(v0) => WherePred::Type(v0.clone()),
        }
    }
}
