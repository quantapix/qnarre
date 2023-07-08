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
impl Clone for AngleBracketedGenericArguments {
    fn clone(&self) -> Self {
        AngleBracketedGenericArguments {
            colon2_token: self.colon2_token.clone(),
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
            fat_arrow_token: self.fat_arrow_token.clone(),
            body: self.body.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Clone for AssocConst {
    fn clone(&self) -> Self {
        AssocConst {
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            eq: self.eq.clone(),
            value: self.value.clone(),
        }
    }
}
impl Clone for AssocType {
    fn clone(&self) -> Self {
        AssocType {
            ident: self.ident.clone(),
            generics: self.generics.clone(),
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
impl Clone for BareFnArg {
    fn clone(&self) -> Self {
        BareFnArg {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Clone for BareVariadic {
    fn clone(&self) -> Self {
        BareVariadic {
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
            lifetimes: self.lifetimes.clone(),
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
            ty: self.ty.clone(),
            eq: self.eq.clone(),
            default: self.default.clone(),
        }
    }
}
impl Clone for Constraint {
    fn clone(&self) -> Self {
        Constraint {
            ident: self.ident.clone(),
            generics: self.generics.clone(),
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
            generics: self.generics.clone(),
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
impl Clone for ExprArray {
    fn clone(&self) -> Self {
        ExprArray {
            attrs: self.attrs.clone(),
            bracket_token: self.bracket_token.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for ExprAssign {
    fn clone(&self) -> Self {
        ExprAssign {
            attrs: self.attrs.clone(),
            left: self.left.clone(),
            eq_token: self.eq_token.clone(),
            right: self.right.clone(),
        }
    }
}
impl Clone for ExprAsync {
    fn clone(&self) -> Self {
        ExprAsync {
            attrs: self.attrs.clone(),
            async_token: self.async_token.clone(),
            capture: self.capture.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ExprAwait {
    fn clone(&self) -> Self {
        ExprAwait {
            attrs: self.attrs.clone(),
            base: self.base.clone(),
            dot_token: self.dot_token.clone(),
            await_token: self.await_token.clone(),
        }
    }
}
impl Clone for ExprBinary {
    fn clone(&self) -> Self {
        ExprBinary {
            attrs: self.attrs.clone(),
            left: self.left.clone(),
            op: self.op.clone(),
            right: self.right.clone(),
        }
    }
}
impl Clone for ExprBlock {
    fn clone(&self) -> Self {
        ExprBlock {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ExprBreak {
    fn clone(&self) -> Self {
        ExprBreak {
            attrs: self.attrs.clone(),
            break_token: self.break_token.clone(),
            label: self.label.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprCall {
    fn clone(&self) -> Self {
        ExprCall {
            attrs: self.attrs.clone(),
            func: self.func.clone(),
            paren_token: self.paren_token.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for ExprCast {
    fn clone(&self) -> Self {
        ExprCast {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            as_token: self.as_token.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Clone for ExprClosure {
    fn clone(&self) -> Self {
        ExprClosure {
            attrs: self.attrs.clone(),
            lifetimes: self.lifetimes.clone(),
            constness: self.constness.clone(),
            movability: self.movability.clone(),
            asyncness: self.asyncness.clone(),
            capture: self.capture.clone(),
            or1_token: self.or1_token.clone(),
            inputs: self.inputs.clone(),
            or2_token: self.or2_token.clone(),
            output: self.output.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for ExprConst {
    fn clone(&self) -> Self {
        ExprConst {
            attrs: self.attrs.clone(),
            const_token: self.const_token.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ExprContinue {
    fn clone(&self) -> Self {
        ExprContinue {
            attrs: self.attrs.clone(),
            continue_token: self.continue_token.clone(),
            label: self.label.clone(),
        }
    }
}
impl Clone for ExprField {
    fn clone(&self) -> Self {
        ExprField {
            attrs: self.attrs.clone(),
            base: self.base.clone(),
            dot_token: self.dot_token.clone(),
            member: self.member.clone(),
        }
    }
}
impl Clone for ExprForLoop {
    fn clone(&self) -> Self {
        ExprForLoop {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            for_token: self.for_token.clone(),
            pat: self.pat.clone(),
            in_token: self.in_token.clone(),
            expr: self.expr.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for ExprGroup {
    fn clone(&self) -> Self {
        ExprGroup {
            attrs: self.attrs.clone(),
            group_token: self.group_token.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprIf {
    fn clone(&self) -> Self {
        ExprIf {
            attrs: self.attrs.clone(),
            if_token: self.if_token.clone(),
            cond: self.cond.clone(),
            then_branch: self.then_branch.clone(),
            else_branch: self.else_branch.clone(),
        }
    }
}
impl Clone for ExprIndex {
    fn clone(&self) -> Self {
        ExprIndex {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            bracket_token: self.bracket_token.clone(),
            index: self.index.clone(),
        }
    }
}
impl Clone for ExprInfer {
    fn clone(&self) -> Self {
        ExprInfer {
            attrs: self.attrs.clone(),
            underscore_token: self.underscore_token.clone(),
        }
    }
}
impl Clone for ExprLet {
    fn clone(&self) -> Self {
        ExprLet {
            attrs: self.attrs.clone(),
            let_token: self.let_token.clone(),
            pat: self.pat.clone(),
            eq_token: self.eq_token.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprLit {
    fn clone(&self) -> Self {
        ExprLit {
            attrs: self.attrs.clone(),
            lit: self.lit.clone(),
        }
    }
}
impl Clone for ExprLoop {
    fn clone(&self) -> Self {
        ExprLoop {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            loop_token: self.loop_token.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for ExprMacro {
    fn clone(&self) -> Self {
        ExprMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
        }
    }
}
impl Clone for ExprMatch {
    fn clone(&self) -> Self {
        ExprMatch {
            attrs: self.attrs.clone(),
            match_token: self.match_token.clone(),
            expr: self.expr.clone(),
            brace_token: self.brace_token.clone(),
            arms: self.arms.clone(),
        }
    }
}
impl Clone for ExprMethodCall {
    fn clone(&self) -> Self {
        ExprMethodCall {
            attrs: self.attrs.clone(),
            receiver: self.receiver.clone(),
            dot_token: self.dot_token.clone(),
            method: self.method.clone(),
            turbofish: self.turbofish.clone(),
            paren_token: self.paren_token.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for ExprParen {
    fn clone(&self) -> Self {
        ExprParen {
            attrs: self.attrs.clone(),
            paren_token: self.paren_token.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprPath {
    fn clone(&self) -> Self {
        ExprPath {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for ExprRange {
    fn clone(&self) -> Self {
        ExprRange {
            attrs: self.attrs.clone(),
            start: self.start.clone(),
            limits: self.limits.clone(),
            end: self.end.clone(),
        }
    }
}
impl Clone for ExprReference {
    fn clone(&self) -> Self {
        ExprReference {
            attrs: self.attrs.clone(),
            and_token: self.and_token.clone(),
            mutability: self.mutability.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprRepeat {
    fn clone(&self) -> Self {
        ExprRepeat {
            attrs: self.attrs.clone(),
            bracket_token: self.bracket_token.clone(),
            expr: self.expr.clone(),
            semi_token: self.semi_token.clone(),
            len: self.len.clone(),
        }
    }
}
impl Clone for ExprReturn {
    fn clone(&self) -> Self {
        ExprReturn {
            attrs: self.attrs.clone(),
            return_token: self.return_token.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprStruct {
    fn clone(&self) -> Self {
        ExprStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace_token: self.brace_token.clone(),
            fields: self.fields.clone(),
            dot2_token: self.dot2_token.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Clone for ExprTry {
    fn clone(&self) -> Self {
        ExprTry {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            question_token: self.question_token.clone(),
        }
    }
}
impl Clone for ExprTryBlock {
    fn clone(&self) -> Self {
        ExprTryBlock {
            attrs: self.attrs.clone(),
            try_token: self.try_token.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ExprTuple {
    fn clone(&self) -> Self {
        ExprTuple {
            attrs: self.attrs.clone(),
            paren_token: self.paren_token.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for ExprUnary {
    fn clone(&self) -> Self {
        ExprUnary {
            attrs: self.attrs.clone(),
            op: self.op.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for ExprUnsafe {
    fn clone(&self) -> Self {
        ExprUnsafe {
            attrs: self.attrs.clone(),
            unsafe_token: self.unsafe_token.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for ExprWhile {
    fn clone(&self) -> Self {
        ExprWhile {
            attrs: self.attrs.clone(),
            label: self.label.clone(),
            while_token: self.while_token.clone(),
            cond: self.cond.clone(),
            body: self.body.clone(),
        }
    }
}
impl Clone for ExprYield {
    fn clone(&self) -> Self {
        ExprYield {
            attrs: self.attrs.clone(),
            yield_token: self.yield_token.clone(),
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
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
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
impl Clone for FieldPat {
    fn clone(&self) -> Self {
        FieldPat {
            attrs: self.attrs.clone(),
            member: self.member.clone(),
            colon: self.colon.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for FieldValue {
    fn clone(&self) -> Self {
        FieldValue {
            attrs: self.attrs.clone(),
            member: self.member.clone(),
            colon_token: self.colon_token.clone(),
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
            brace_token: self.brace_token.clone(),
            named: self.named.clone(),
        }
    }
}
impl Clone for FieldsUnnamed {
    fn clone(&self) -> Self {
        FieldsUnnamed {
            paren_token: self.paren_token.clone(),
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
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ForeignItemMacro {
    fn clone(&self) -> Self {
        ForeignItemMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ForeignItemStatic {
    fn clone(&self) -> Self {
        ForeignItemStatic {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            static_token: self.static_token.clone(),
            mutability: self.mutability.clone(),
            ident: self.ident.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ForeignItemType {
    fn clone(&self) -> Self {
        ForeignItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type_token: self.type_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for GenericArgument {
    fn clone(&self) -> Self {
        match self {
            GenericArgument::Lifetime(v0) => GenericArgument::Lifetime(v0.clone()),
            GenericArgument::Type(v0) => GenericArgument::Type(v0.clone()),
            GenericArgument::Const(v0) => GenericArgument::Const(v0.clone()),
            GenericArgument::AssocType(v0) => GenericArgument::AssocType(v0.clone()),
            GenericArgument::AssocConst(v0) => GenericArgument::AssocConst(v0.clone()),
            GenericArgument::Constraint(v0) => GenericArgument::Constraint(v0.clone()),
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
            clause: self.clause.clone(),
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
            defaultness: self.defaultness.clone(),
            const_token: self.const_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
            eq_token: self.eq_token.clone(),
            expr: self.expr.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ImplItemFn {
    fn clone(&self) -> Self {
        ImplItemFn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            defaultness: self.defaultness.clone(),
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
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ImplItemType {
    fn clone(&self) -> Self {
        ImplItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            defaultness: self.defaultness.clone(),
            type_token: self.type_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            eq_token: self.eq_token.clone(),
            ty: self.ty.clone(),
            semi_token: self.semi_token.clone(),
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
            const_token: self.const_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
            eq_token: self.eq_token.clone(),
            expr: self.expr.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemEnum {
    fn clone(&self) -> Self {
        ItemEnum {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            enum_token: self.enum_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            brace_token: self.brace_token.clone(),
            variants: self.variants.clone(),
        }
    }
}
impl Clone for ItemExternCrate {
    fn clone(&self) -> Self {
        ItemExternCrate {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            extern_token: self.extern_token.clone(),
            crate_token: self.crate_token.clone(),
            ident: self.ident.clone(),
            rename: self.rename.clone(),
            semi_token: self.semi_token.clone(),
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
            unsafety: self.unsafety.clone(),
            abi: self.abi.clone(),
            brace_token: self.brace_token.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for ItemImpl {
    fn clone(&self) -> Self {
        ItemImpl {
            attrs: self.attrs.clone(),
            defaultness: self.defaultness.clone(),
            unsafety: self.unsafety.clone(),
            impl_token: self.impl_token.clone(),
            generics: self.generics.clone(),
            trait_: self.trait_.clone(),
            self_ty: self.self_ty.clone(),
            brace_token: self.brace_token.clone(),
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
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemMod {
    fn clone(&self) -> Self {
        ItemMod {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafety: self.unsafety.clone(),
            mod_token: self.mod_token.clone(),
            ident: self.ident.clone(),
            content: self.content.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for ItemStatic {
    fn clone(&self) -> Self {
        ItemStatic {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            static_token: self.static_token.clone(),
            mutability: self.mutability.clone(),
            ident: self.ident.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
            eq_token: self.eq_token.clone(),
            expr: self.expr.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemStruct {
    fn clone(&self) -> Self {
        ItemStruct {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            struct_token: self.struct_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            fields: self.fields.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemTrait {
    fn clone(&self) -> Self {
        ItemTrait {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafety: self.unsafety.clone(),
            auto_token: self.auto_token.clone(),
            restriction: self.restriction.clone(),
            trait_token: self.trait_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            colon_token: self.colon_token.clone(),
            supertraits: self.supertraits.clone(),
            brace_token: self.brace_token.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for ItemTraitAlias {
    fn clone(&self) -> Self {
        ItemTraitAlias {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            trait_token: self.trait_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            eq_token: self.eq_token.clone(),
            bounds: self.bounds.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemType {
    fn clone(&self) -> Self {
        ItemType {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type_token: self.type_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            eq_token: self.eq_token.clone(),
            ty: self.ty.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for ItemUnion {
    fn clone(&self) -> Self {
        ItemUnion {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            union_token: self.union_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for ItemUse {
    fn clone(&self) -> Self {
        ItemUse {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            use_token: self.use_token.clone(),
            leading_colon: self.leading_colon.clone(),
            tree: self.tree.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for Label {
    fn clone(&self) -> Self {
        Label {
            name: self.name.clone(),
            colon_token: self.colon_token.clone(),
        }
    }
}
impl Clone for LifetimeParam {
    fn clone(&self) -> Self {
        LifetimeParam {
            attrs: self.attrs.clone(),
            lifetime: self.lifetime.clone(),
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
impl Clone for Local {
    fn clone(&self) -> Self {
        Local {
            attrs: self.attrs.clone(),
            let_: self.let_.clone(),
            pat: self.pat.clone(),
            init: self.init.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for LocalInit {
    fn clone(&self) -> Self {
        LocalInit {
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
impl Clone for MacroDelimiter {
    fn clone(&self) -> Self {
        match self {
            MacroDelimiter::Paren(v0) => MacroDelimiter::Paren(v0.clone()),
            MacroDelimiter::Brace(v0) => MacroDelimiter::Brace(v0.clone()),
            MacroDelimiter::Bracket(v0) => MacroDelimiter::Bracket(v0.clone()),
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
            val: self.val.clone(),
        }
    }
}
impl Clone for ParenthesizedGenericArguments {
    fn clone(&self) -> Self {
        ParenthesizedGenericArguments {
            paren: self.paren.clone(),
            inputs: self.inputs.clone(),
            output: self.output.clone(),
        }
    }
}
impl Clone for Pat {
    fn clone(&self) -> Self {
        match self {
            Pat::Const(v0) => Pat::Const(v0.clone()),
            Pat::Ident(v0) => Pat::Ident(v0.clone()),
            Pat::Lit(v0) => Pat::Lit(v0.clone()),
            Pat::Macro(v0) => Pat::Macro(v0.clone()),
            Pat::Or(v0) => Pat::Or(v0.clone()),
            Pat::Paren(v0) => Pat::Paren(v0.clone()),
            Pat::Path(v0) => Pat::Path(v0.clone()),
            Pat::Range(v0) => Pat::Range(v0.clone()),
            Pat::Reference(v0) => Pat::Reference(v0.clone()),
            Pat::Rest(v0) => Pat::Rest(v0.clone()),
            Pat::Slice(v0) => Pat::Slice(v0.clone()),
            Pat::Struct(v0) => Pat::Struct(v0.clone()),
            Pat::Tuple(v0) => Pat::Tuple(v0.clone()),
            Pat::TupleStruct(v0) => Pat::TupleStruct(v0.clone()),
            Pat::Type(v0) => Pat::Type(v0.clone()),
            Pat::Verbatim(v0) => Pat::Verbatim(v0.clone()),
            Pat::Wild(v0) => Pat::Wild(v0.clone()),
        }
    }
}
impl Clone for PatIdent {
    fn clone(&self) -> Self {
        PatIdent {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            subpat: self.subpat.clone(),
        }
    }
}
impl Clone for PatOr {
    fn clone(&self) -> Self {
        PatOr {
            attrs: self.attrs.clone(),
            leading_vert: self.leading_vert.clone(),
            cases: self.cases.clone(),
        }
    }
}
impl Clone for PatParen {
    fn clone(&self) -> Self {
        PatParen {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for PatReference {
    fn clone(&self) -> Self {
        PatReference {
            attrs: self.attrs.clone(),
            and_: self.and_.clone(),
            mutability: self.mutability.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for PatRest {
    fn clone(&self) -> Self {
        PatRest {
            attrs: self.attrs.clone(),
            dot2: self.dot2.clone(),
        }
    }
}
impl Clone for PatSlice {
    fn clone(&self) -> Self {
        PatSlice {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for PatStruct {
    fn clone(&self) -> Self {
        PatStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace: self.brace.clone(),
            fields: self.fields.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Clone for PatTuple {
    fn clone(&self) -> Self {
        PatTuple {
            attrs: self.attrs.clone(),
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for PatTupleStruct {
    fn clone(&self) -> Self {
        PatTupleStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            paren: self.paren.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Clone for PatType {
    fn clone(&self) -> Self {
        PatType {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            colon: self.colon.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Clone for PatWild {
    fn clone(&self) -> Self {
        PatWild {
            attrs: self.attrs.clone(),
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for Path {
    fn clone(&self) -> Self {
        Path {
            leading_colon: self.leading_colon.clone(),
            segments: self.segments.clone(),
        }
    }
}
impl Clone for PathArguments {
    fn clone(&self) -> Self {
        match self {
            PathArguments::None => PathArguments::None,
            PathArguments::AngleBracketed(v0) => PathArguments::AngleBracketed(v0.clone()),
            PathArguments::Parenthesized(v0) => PathArguments::Parenthesized(v0.clone()),
        }
    }
}
impl Clone for PathSegment {
    fn clone(&self) -> Self {
        PathSegment {
            ident: self.ident.clone(),
            arguments: self.arguments.clone(),
        }
    }
}
impl Clone for PredLifetime {
    fn clone(&self) -> Self {
        PredLifetime {
            lifetime: self.lifetime.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for PredType {
    fn clone(&self) -> Self {
        PredType {
            lifetimes: self.lifetimes.clone(),
            bounded_ty: self.bounded_ty.clone(),
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
            position: self.position.clone(),
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
            mutability: self.mutability.clone(),
            self_token: self.self_token.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
        }
    }
}
impl Clone for ReturnType {
    fn clone(&self) -> Self {
        match self {
            ReturnType::Default => ReturnType::Default,
            ReturnType::Type(v0, v1) => ReturnType::Type(v0.clone(), v1.clone()),
        }
    }
}
impl Clone for Signature {
    fn clone(&self) -> Self {
        Signature {
            constness: self.constness.clone(),
            asyncness: self.asyncness.clone(),
            unsafety: self.unsafety.clone(),
            abi: self.abi.clone(),
            fn_token: self.fn_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            paren_token: self.paren_token.clone(),
            inputs: self.inputs.clone(),
            variadic: self.variadic.clone(),
            output: self.output.clone(),
        }
    }
}
impl Clone for StaticMutability {
    fn clone(&self) -> Self {
        match self {
            StaticMutability::Mut(v0) => StaticMutability::Mut(v0.clone()),
            StaticMutability::None => StaticMutability::None,
        }
    }
}
impl Clone for Stmt {
    fn clone(&self) -> Self {
        match self {
            Stmt::Local(v0) => Stmt::Local(v0.clone()),
            Stmt::Item(v0) => Stmt::Item(v0.clone()),
            Stmt::Expr(v0, v1) => Stmt::Expr(v0.clone(), v1.clone()),
            Stmt::Macro(v0) => Stmt::Macro(v0.clone()),
        }
    }
}
impl Clone for StmtMacro {
    fn clone(&self) -> Self {
        StmtMacro {
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
            lifetimes: self.lifetimes.clone(),
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
            const_token: self.const_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            colon_token: self.colon_token.clone(),
            ty: self.ty.clone(),
            default: self.default.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for TraitItemFn {
    fn clone(&self) -> Self {
        TraitItemFn {
            attrs: self.attrs.clone(),
            sig: self.sig.clone(),
            default: self.default.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for TraitItemMacro {
    fn clone(&self) -> Self {
        TraitItemMacro {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for TraitItemType {
    fn clone(&self) -> Self {
        TraitItemType {
            attrs: self.attrs.clone(),
            type_token: self.type_token.clone(),
            ident: self.ident.clone(),
            generics: self.generics.clone(),
            colon_token: self.colon_token.clone(),
            bounds: self.bounds.clone(),
            default: self.default.clone(),
            semi_token: self.semi_token.clone(),
        }
    }
}
impl Clone for Type {
    fn clone(&self) -> Self {
        match self {
            Type::Array(v0) => Type::Array(v0.clone()),
            Type::BareFn(v0) => Type::BareFn(v0.clone()),
            Type::Group(v0) => Type::Group(v0.clone()),
            Type::ImplTrait(v0) => Type::ImplTrait(v0.clone()),
            Type::Infer(v0) => Type::Infer(v0.clone()),
            Type::Macro(v0) => Type::Macro(v0.clone()),
            Type::Never(v0) => Type::Never(v0.clone()),
            Type::Paren(v0) => Type::Paren(v0.clone()),
            Type::Path(v0) => Type::Path(v0.clone()),
            Type::Ptr(v0) => Type::Ptr(v0.clone()),
            Type::Reference(v0) => Type::Reference(v0.clone()),
            Type::Slice(v0) => Type::Slice(v0.clone()),
            Type::TraitObject(v0) => Type::TraitObject(v0.clone()),
            Type::Tuple(v0) => Type::Tuple(v0.clone()),
            Type::Verbatim(v0) => Type::Verbatim(v0.clone()),
        }
    }
}
impl Clone for TypeArray {
    fn clone(&self) -> Self {
        TypeArray {
            bracket_token: self.bracket_token.clone(),
            elem: self.elem.clone(),
            semi: self.semi.clone(),
            len: self.len.clone(),
        }
    }
}
impl Clone for TypeBareFn {
    fn clone(&self) -> Self {
        TypeBareFn {
            lifetimes: self.lifetimes.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            paren: self.paren.clone(),
            inputs: self.inputs.clone(),
            variadic: self.variadic.clone(),
            output: self.output.clone(),
        }
    }
}
impl Clone for TypeGroup {
    fn clone(&self) -> Self {
        TypeGroup {
            group: self.group.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for TypeImplTrait {
    fn clone(&self) -> Self {
        TypeImplTrait {
            impl_: self.impl_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for TypeInfer {
    fn clone(&self) -> Self {
        TypeInfer {
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for TypeMacro {
    fn clone(&self) -> Self {
        TypeMacro { mac: self.mac.clone() }
    }
}
impl Clone for TypeNever {
    fn clone(&self) -> Self {
        TypeNever {
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
impl Clone for TypeParen {
    fn clone(&self) -> Self {
        TypeParen {
            paren_token: self.paren_token.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for TypePath {
    fn clone(&self) -> Self {
        TypePath {
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for TypePtr {
    fn clone(&self) -> Self {
        TypePtr {
            star: self.star.clone(),
            const_: self.const_.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for TypeReference {
    fn clone(&self) -> Self {
        TypeReference {
            and_: self.and_.clone(),
            lifetime: self.lifetime.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for TypeSlice {
    fn clone(&self) -> Self {
        TypeSlice {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for TypeTraitObject {
    fn clone(&self) -> Self {
        TypeTraitObject {
            dyn_: self.dyn_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for TypeTuple {
    fn clone(&self) -> Self {
        TypeTuple {
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
            star_token: self.star_token.clone(),
        }
    }
}
impl Clone for UseGroup {
    fn clone(&self) -> Self {
        UseGroup {
            brace_token: self.brace_token.clone(),
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
            colon2_token: self.colon2_token.clone(),
            tree: self.tree.clone(),
        }
    }
}
impl Clone for UseRename {
    fn clone(&self) -> Self {
        UseRename {
            ident: self.ident.clone(),
            as_token: self.as_token.clone(),
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
