#![allow(clippy::clone_on_copy, clippy::expl_impl_clone_on_copy)]
use crate::*;

impl Clone for typ::Abi {
    fn clone(&self) -> Self {
        typ::Abi {
            extern_: self.extern_.clone(),
            name: self.name.clone(),
        }
    }
}
impl Clone for path::Angled {
    fn clone(&self) -> Self {
        path::Angled {
            colon2: self.colon2.clone(),
            lt: self.lt.clone(),
            args: self.args.clone(),
            gt: self.gt.clone(),
        }
    }
}
impl Clone for expr::Arm {
    fn clone(&self) -> Self {
        expr::Arm {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            guard: self.guard.clone(),
            fat_arrow: self.fat_arrow.clone(),
            body: self.body.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Clone for path::AssocConst {
    fn clone(&self) -> Self {
        path::AssocConst {
            ident: self.ident.clone(),
            args: self.args.clone(),
            eq: self.eq.clone(),
            val: self.val.clone(),
        }
    }
}
impl Clone for path::AssocType {
    fn clone(&self) -> Self {
        path::AssocType {
            ident: self.ident.clone(),
            args: self.args.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Copy for attr::Style {}
impl Clone for attr::Style {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for attr::Attr {
    fn clone(&self) -> Self {
        attr::Attr {
            pound: self.pound.clone(),
            style: self.style.clone(),
            bracket: self.bracket.clone(),
            meta: self.meta.clone(),
        }
    }
}
impl Clone for typ::FnArg {
    fn clone(&self) -> Self {
        typ::FnArg {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for typ::Variadic {
    fn clone(&self) -> Self {
        typ::Variadic {
            attrs: self.attrs.clone(),
            name: self.name.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Copy for expr::BinOp {}
impl Clone for expr::BinOp {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for stmt::Block {
    fn clone(&self) -> Self {
        stmt::Block {
            brace: self.brace.clone(),
            stmts: self.stmts.clone(),
        }
    }
}
impl Clone for gen::bound::Lifes {
    fn clone(&self) -> Self {
        gen::bound::Lifes {
            for_: self.for_.clone(),
            lt: self.lt.clone(),
            lifes: self.lifes.clone(),
            gt: self.gt.clone(),
        }
    }
}
impl Clone for gen::param::Const {
    fn clone(&self) -> Self {
        gen::param::Const {
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
impl Clone for path::Constraint {
    fn clone(&self) -> Self {
        path::Constraint {
            ident: self.ident.clone(),
            args: self.args.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for data::Data {
    fn clone(&self) -> Self {
        use data::Data::*;
        match self {
            Struct(x) => Struct(x.clone()),
            Enum(x) => Enum(x.clone()),
            Union(x) => Union(x.clone()),
        }
    }
}
impl Clone for data::Enum {
    fn clone(&self) -> Self {
        data::Enum {
            enum_: self.enum_.clone(),
            brace: self.brace.clone(),
            variants: self.variants.clone(),
        }
    }
}
impl Clone for data::Struct {
    fn clone(&self) -> Self {
        data::Struct {
            struct_: self.struct_.clone(),
            fields: self.fields.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for data::Union {
    fn clone(&self) -> Self {
        data::Union {
            union_: self.union_.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for Input {
    fn clone(&self) -> Self {
        Input {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            data: self.data.clone(),
        }
    }
}
impl Clone for expr::Expr {
    fn clone(&self) -> Self {
        use expr::Expr::*;
        match self {
            Array(x) => Array(x.clone()),
            Assign(x) => Assign(x.clone()),
            Async(x) => Async(x.clone()),
            Await(x) => Await(x.clone()),
            Binary(x) => Binary(x.clone()),
            Block(x) => Block(x.clone()),
            Break(x) => Break(x.clone()),
            Call(x) => Call(x.clone()),
            Cast(x) => Cast(x.clone()),
            Closure(x) => Closure(x.clone()),
            Const(x) => Const(x.clone()),
            Continue(x) => Continue(x.clone()),
            Field(x) => Field(x.clone()),
            For(x) => For(x.clone()),
            Group(x) => Group(x.clone()),
            If(x) => If(x.clone()),
            Index(x) => Index(x.clone()),
            Infer(x) => Infer(x.clone()),
            Let(x) => Let(x.clone()),
            Lit(x) => Lit(x.clone()),
            Loop(x) => Loop(x.clone()),
            Mac(x) => Mac(x.clone()),
            Match(x) => Match(x.clone()),
            Method(x) => Method(x.clone()),
            Parenth(x) => Parenth(x.clone()),
            Path(x) => Path(x.clone()),
            Range(x) => Range(x.clone()),
            Ref(x) => Ref(x.clone()),
            Repeat(x) => Repeat(x.clone()),
            Return(x) => Return(x.clone()),
            Struct(x) => Struct(x.clone()),
            Try(x) => Try(x.clone()),
            TryBlock(x) => TryBlock(x.clone()),
            Tuple(x) => Tuple(x.clone()),
            Unary(x) => Unary(x.clone()),
            Unsafe(x) => Unsafe(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
            While(x) => While(x.clone()),
            Yield(x) => Yield(x.clone()),
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
            life: self.life.clone(),
            val: self.val.clone(),
        }
    }
}
impl Clone for expr::Call {
    fn clone(&self) -> Self {
        expr::Call {
            attrs: self.attrs.clone(),
            func: self.func.clone(),
            parenth: self.parenth.clone(),
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
            ins: self.inputs.clone(),
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
            life: self.life.clone(),
        }
    }
}
impl Clone for expr::Field {
    fn clone(&self) -> Self {
        expr::Field {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            dot: self.dot.clone(),
            memb: self.memb.clone(),
        }
    }
}
impl Clone for expr::For {
    fn clone(&self) -> Self {
        expr::For {
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
            then_: self.then_branch.clone(),
            else_: self.else_branch.clone(),
        }
    }
}
impl Clone for expr::Index {
    fn clone(&self) -> Self {
        expr::Index {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            bracket: self.bracket.clone(),
            idx: self.idx.clone(),
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
impl Clone for expr::Method {
    fn clone(&self) -> Self {
        expr::Method {
            attrs: self.attrs.clone(),
            expr: self.expr.clone(),
            dot: self.dot.clone(),
            method: self.method.clone(),
            turbofish: self.turbofish.clone(),
            parenth: self.parenth.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for expr::Parenth {
    fn clone(&self) -> Self {
        expr::Parenth {
            attrs: self.attrs.clone(),
            parenth: self.parenth.clone(),
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
            parenth: self.parenth.clone(),
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
            block: self.block.clone(),
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
impl Clone for data::Field {
    fn clone(&self) -> Self {
        data::Field {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for data::Mut {
    fn clone(&self) -> Self {
        match self {
            data::Mut::None => data::Mut::None,
        }
    }
}
impl Clone for pat::Field {
    fn clone(&self) -> Self {
        pat::Field {
            attrs: self.attrs.clone(),
            memb: self.memb.clone(),
            colon: self.colon.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for expr::FieldValue {
    fn clone(&self) -> Self {
        expr::FieldValue {
            attrs: self.attrs.clone(),
            memb: self.memb.clone(),
            colon: self.colon.clone(),
            expr: self.expr.clone(),
        }
    }
}
impl Clone for data::Fields {
    fn clone(&self) -> Self {
        match self {
            data::Fields::Named(x) => data::Fields::Named(x.clone()),
            data::Fields::Unnamed(x) => data::Fields::Unnamed(x.clone()),
            data::Fields::Unit => data::Fields::Unit,
        }
    }
}
impl Clone for data::Named {
    fn clone(&self) -> Self {
        data::Named {
            brace: self.brace.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for data::Unnamed {
    fn clone(&self) -> Self {
        data::Unnamed {
            parenth: self.parenth.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for item::File {
    fn clone(&self) -> Self {
        item::File {
            shebang: self.shebang.clone(),
            attrs: self.attrs.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for item::FnArg {
    fn clone(&self) -> Self {
        match self {
            item::FnArg::Receiver(x) => item::FnArg::Receiver(x.clone()),
            item::FnArg::Type(x) => item::FnArg::Type(x.clone()),
        }
    }
}
impl Clone for item::foreign::Item {
    fn clone(&self) -> Self {
        match self {
            item::foreign::Item::Fn(x) => item::foreign::Item::Fn(x.clone()),
            item::foreign::Item::Static(x) => item::foreign::Item::Static(x.clone()),
            item::foreign::Item::Type(x) => item::foreign::Item::Type(x.clone()),
            item::foreign::Item::Macro(x) => item::foreign::Item::Macro(x.clone()),
            item::foreign::Item::Verbatim(x) => item::foreign::Item::Verbatim(x.clone()),
        }
    }
}
impl Clone for item::foreign::Fn {
    fn clone(&self) -> Self {
        item::foreign::Fn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: self.sig.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::foreign::Mac {
    fn clone(&self) -> Self {
        item::foreign::Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::foreign::Static {
    fn clone(&self) -> Self {
        item::foreign::Static {
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
impl Clone for item::foreign::Type {
    fn clone(&self) -> Self {
        item::foreign::Type {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type_: self.type_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for path::Arg {
    fn clone(&self) -> Self {
        use path::Arg::*;
        match self {
            AssocConst(x) => AssocConst(x.clone()),
            AssocType(x) => AssocType(x.clone()),
            Const(x) => Const(x.clone()),
            Constraint(x) => Constraint(x.clone()),
            Life(x) => Life(x.clone()),
            Type(x) => Type(x.clone()),
        }
    }
}
impl Clone for gen::Param {
    fn clone(&self) -> Self {
        use gen::Param::*;
        match self {
            Const(x) => Const(x.clone()),
            Life(x) => Life(x.clone()),
            Type(x) => Type(x.clone()),
        }
    }
}
impl Clone for gen::Gens {
    fn clone(&self) -> Self {
        gen::Gens {
            lt: self.lt.clone(),
            params: self.params.clone(),
            gt: self.gt.clone(),
            where_: self.where_.clone(),
        }
    }
}
impl Clone for item::impl_::Item {
    fn clone(&self) -> Self {
        use item::impl_::Item::*;
        match self {
            Const(x) => Const(x.clone()),
            Fn(x) => Fn(x.clone()),
            Mac(x) => Mac(x.clone()),
            Type(x) => Type(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Clone for item::impl_::Const {
    fn clone(&self) -> Self {
        item::impl_::Const {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default: self.default.clone(),
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
impl Clone for item::impl_::Fn {
    fn clone(&self) -> Self {
        item::impl_::Fn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default: self.default.clone(),
            sig: self.sig.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for item::impl_::Mac {
    fn clone(&self) -> Self {
        item::impl_::Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::impl_::Type {
    fn clone(&self) -> Self {
        item::impl_::Type {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            default: self.default.clone(),
            type_: self.type_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::impl_::Restriction {
    fn clone(&self) -> Self {
        match *self {}
    }
}
impl Clone for expr::Idx {
    fn clone(&self) -> Self {
        expr::Idx {
            idx: self.idx.clone(),
            span: self.span.clone(),
        }
    }
}
impl Clone for item::Item {
    fn clone(&self) -> Self {
        use item::Item::*;
        match self {
            Const(x) => Const(x.clone()),
            Enum(x) => Enum(x.clone()),
            Extern(x) => Extern(x.clone()),
            Fn(x) => Fn(x.clone()),
            Foreign(x) => Foreign(x.clone()),
            Impl(x) => Impl(x.clone()),
            Mac(x) => Mac(x.clone()),
            Mod(x) => Mod(x.clone()),
            Static(x) => Static(x.clone()),
            Struct(x) => Struct(x.clone()),
            Trait(x) => Trait(x.clone()),
            TraitAlias(x) => TraitAlias(x.clone()),
            Type(x) => Type(x.clone()),
            Union(x) => Union(x.clone()),
            Use(x) => Use(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Clone for item::Const {
    fn clone(&self) -> Self {
        item::Const {
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
impl Clone for item::Enum {
    fn clone(&self) -> Self {
        item::Enum {
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
impl Clone for item::Extern {
    fn clone(&self) -> Self {
        item::Extern {
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
impl Clone for item::Fn {
    fn clone(&self) -> Self {
        item::Fn {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            sig: self.sig.clone(),
            block: self.block.clone(),
        }
    }
}
impl Clone for item::Foreign {
    fn clone(&self) -> Self {
        item::Foreign {
            attrs: self.attrs.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for item::Impl {
    fn clone(&self) -> Self {
        item::Impl {
            attrs: self.attrs.clone(),
            default: self.default.clone(),
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
impl Clone for item::Mac {
    fn clone(&self) -> Self {
        item::Mac {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::Mod {
    fn clone(&self) -> Self {
        item::Mod {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            mod_: self.mod_.clone(),
            ident: self.ident.clone(),
            items: self.items.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::Static {
    fn clone(&self) -> Self {
        item::Static {
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
impl Clone for item::Struct {
    fn clone(&self) -> Self {
        item::Struct {
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
impl Clone for item::Trait {
    fn clone(&self) -> Self {
        item::Trait {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            unsafe_: self.unsafe_.clone(),
            auto: self.auto.clone(),
            restriction: self.restriction.clone(),
            trait_: self.trait_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            supers: self.supers.clone(),
            brace: self.brace.clone(),
            items: self.items.clone(),
        }
    }
}
impl Clone for item::TraitAlias {
    fn clone(&self) -> Self {
        item::TraitAlias {
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
impl Clone for item::Type {
    fn clone(&self) -> Self {
        item::Type {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            type_: self.type_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            eq: self.eq.clone(),
            typ: self.typ.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::Union {
    fn clone(&self) -> Self {
        item::Union {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            union_: self.union_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            fields: self.fields.clone(),
        }
    }
}
impl Clone for item::Use {
    fn clone(&self) -> Self {
        item::Use {
            attrs: self.attrs.clone(),
            vis: self.vis.clone(),
            use_: self.use_.clone(),
            colon: self.colon.clone(),
            tree: self.tree.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for expr::Label {
    fn clone(&self) -> Self {
        expr::Label {
            name: self.name.clone(),
            colon: self.colon.clone(),
        }
    }
}
impl Clone for gen::param::Life {
    fn clone(&self) -> Self {
        gen::param::Life {
            attrs: self.attrs.clone(),
            life: self.life.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for lit::Lit {
    fn clone(&self) -> Self {
        use lit::Lit::*;
        match self {
            Bool(x) => Bool(x.clone()),
            Byte(x) => Byte(x.clone()),
            ByteStr(x) => ByteStr(x.clone()),
            Char(x) => Char(x.clone()),
            Float(x) => Float(x.clone()),
            Int(x) => Int(x.clone()),
            Str(x) => Str(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
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
impl Clone for stmt::Init {
    fn clone(&self) -> Self {
        stmt::Init {
            eq: self.eq.clone(),
            expr: self.expr.clone(),
            diverge: self.diverge.clone(),
        }
    }
}
impl Clone for mac::Mac {
    fn clone(&self) -> Self {
        mac::Mac {
            path: self.path.clone(),
            bang: self.bang.clone(),
            delim: self.delim.clone(),
            toks: self.toks.clone(),
        }
    }
}
impl Clone for tok::Delim {
    fn clone(&self) -> Self {
        use tok::Delim::*;
        match self {
            Brace(x) => Brace(x.clone()),
            Bracket(x) => Bracket(x.clone()),
            Parenth(x) => Parenth(x.clone()),
        }
    }
}
impl Clone for expr::Member {
    fn clone(&self) -> Self {
        use expr::Member::*;
        match self {
            Named(x) => Named(x.clone()),
            Unnamed(x) => Unnamed(x.clone()),
        }
    }
}
impl Clone for attr::Meta {
    fn clone(&self) -> Self {
        use attr::Meta::*;
        match self {
            List(x) => List(x.clone()),
            NameValue(x) => NameValue(x.clone()),
            Path(x) => Path(x.clone()),
        }
    }
}
impl Clone for attr::List {
    fn clone(&self) -> Self {
        attr::List {
            path: self.path.clone(),
            delim: self.delim.clone(),
            toks: self.toks.clone(),
        }
    }
}
impl Clone for attr::NameValue {
    fn clone(&self) -> Self {
        attr::NameValue {
            name: self.name.clone(),
            eq: self.eq.clone(),
            val: self.val.clone(),
        }
    }
}
impl Clone for path::Parenthed {
    fn clone(&self) -> Self {
        path::Parenthed {
            parenth: self.parenth.clone(),
            args: self.args.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Clone for pat::Pat {
    fn clone(&self) -> Self {
        use pat::Pat::*;
        match self {
            Const(x) => Const(x.clone()),
            Ident(x) => Ident(x.clone()),
            Lit(x) => Lit(x.clone()),
            Mac(x) => Mac(x.clone()),
            Or(x) => Or(x.clone()),
            Parenth(x) => Parenth(x.clone()),
            Path(x) => Path(x.clone()),
            Range(x) => Range(x.clone()),
            Ref(x) => Ref(x.clone()),
            Rest(x) => Rest(x.clone()),
            Slice(x) => Slice(x.clone()),
            Struct(x) => Struct(x.clone()),
            Tuple(x) => Tuple(x.clone()),
            TupleStruct(x) => TupleStruct(x.clone()),
            Type(x) => Type(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
            Wild(x) => Wild(x.clone()),
        }
    }
}
impl Clone for pat::Ident {
    fn clone(&self) -> Self {
        pat::Ident {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            ident: self.ident.clone(),
            sub: self.sub.clone(),
        }
    }
}
impl Clone for pat::Or {
    fn clone(&self) -> Self {
        pat::Or {
            attrs: self.attrs.clone(),
            vert: self.vert.clone(),
            cases: self.cases.clone(),
        }
    }
}
impl Clone for pat::Parenth {
    fn clone(&self) -> Self {
        pat::Parenth {
            attrs: self.attrs.clone(),
            parenth: self.parenth.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for pat::Ref {
    fn clone(&self) -> Self {
        pat::Ref {
            attrs: self.attrs.clone(),
            and: self.and.clone(),
            mut_: self.mut_.clone(),
            pat: self.pat.clone(),
        }
    }
}
impl Clone for pat::Rest {
    fn clone(&self) -> Self {
        pat::Rest {
            attrs: self.attrs.clone(),
            dot2: self.dot2.clone(),
        }
    }
}
impl Clone for pat::Slice {
    fn clone(&self) -> Self {
        pat::Slice {
            attrs: self.attrs.clone(),
            bracket: self.bracket.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Clone for pat::Struct {
    fn clone(&self) -> Self {
        pat::Struct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            brace: self.brace.clone(),
            fields: self.fields.clone(),
            rest: self.rest.clone(),
        }
    }
}
impl Clone for pat::Tuple {
    fn clone(&self) -> Self {
        pat::Tuple {
            attrs: self.attrs.clone(),
            parenth: self.parenth.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Clone for pat::TupleStruct {
    fn clone(&self) -> Self {
        pat::TupleStruct {
            attrs: self.attrs.clone(),
            qself: self.qself.clone(),
            path: self.path.clone(),
            parenth: self.parenth.clone(),
            pats: self.pats.clone(),
        }
    }
}
impl Clone for pat::Type {
    fn clone(&self) -> Self {
        pat::Type {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for pat::Wild {
    fn clone(&self) -> Self {
        pat::Wild {
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
impl Clone for path::Args {
    fn clone(&self) -> Self {
        use path::Args::*;
        match self {
            None => None,
            Angled(x) => Angled(x.clone()),
            Parenthed(x) => Parenthed(x.clone()),
        }
    }
}
impl Clone for path::Segment {
    fn clone(&self) -> Self {
        path::Segment {
            ident: self.ident.clone(),
            args: self.args.clone(),
        }
    }
}
impl Clone for gen::where_::Life {
    fn clone(&self) -> Self {
        gen::where_::Life {
            life: self.life.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for gen::where_::Type {
    fn clone(&self) -> Self {
        gen::where_::Type {
            lifes: self.lifes.clone(),
            typ: self.typ.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for path::QSelf {
    fn clone(&self) -> Self {
        path::QSelf {
            lt: self.lt.clone(),
            typ: self.typ.clone(),
            pos: self.pos.clone(),
            as_: self.as_.clone(),
            gt: self.gt.clone(),
        }
    }
}
impl Copy for expr::Limits {}
impl Clone for expr::Limits {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for item::Receiver {
    fn clone(&self) -> Self {
        item::Receiver {
            attrs: self.attrs.clone(),
            ref_: self.ref_.clone(),
            mut_: self.mut_.clone(),
            self_: self.self_.clone(),
            colon: self.colon.clone(),
            typ: self.typ.clone(),
        }
    }
}
impl Clone for typ::Ret {
    fn clone(&self) -> Self {
        use typ::Ret::*;
        match self {
            Default => Default,
            Type(x, v1) => Type(x.clone(), v1.clone()),
        }
    }
}
impl Clone for item::Sig {
    fn clone(&self) -> Self {
        item::Sig {
            const_: self.const_.clone(),
            async_: self.async_.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            parenth: self.parenth.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Clone for item::StaticMut {
    fn clone(&self) -> Self {
        use item::StaticMut::*;
        match self {
            Mut(x) => Mut(x.clone()),
            None => None,
        }
    }
}
impl Clone for stmt::Stmt {
    fn clone(&self) -> Self {
        use stmt::Stmt::*;
        match self {
            Expr(x, v1) => Expr(x.clone(), v1.clone()),
            Item(x) => Item(x.clone()),
            Local(x) => Local(x.clone()),
            Mac(x) => Mac(x.clone()),
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
impl Clone for gen::bound::Trait {
    fn clone(&self) -> Self {
        gen::bound::Trait {
            parenth: self.parenth.clone(),
            modif: self.modif.clone(),
            lifes: self.lifes.clone(),
            path: self.path.clone(),
        }
    }
}
impl Copy for gen::bound::Modifier {}
impl Clone for gen::bound::Modifier {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for item::trait_::Item {
    fn clone(&self) -> Self {
        use item::trait_::Item::*;
        match self {
            Const(x) => Const(x.clone()),
            Fn(x) => Fn(x.clone()),
            Mac(x) => Mac(x.clone()),
            Type(x) => Type(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Clone for item::trait_::Const {
    fn clone(&self) -> Self {
        item::trait_::Const {
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
impl Clone for item::trait_::Fn {
    fn clone(&self) -> Self {
        item::trait_::Fn {
            attrs: self.attrs.clone(),
            sig: self.sig.clone(),
            default: self.default.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::trait_::Mac {
    fn clone(&self) -> Self {
        item::trait_::Mac {
            attrs: self.attrs.clone(),
            mac: self.mac.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for item::trait_::Type {
    fn clone(&self) -> Self {
        item::trait_::Type {
            attrs: self.attrs.clone(),
            type_: self.type_.clone(),
            ident: self.ident.clone(),
            gens: self.gens.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
            default: self.default.clone(),
            semi: self.semi.clone(),
        }
    }
}
impl Clone for typ::Type {
    fn clone(&self) -> Self {
        use typ::Type::*;
        match self {
            Array(x) => Array(x.clone()),
            Fn(x) => Fn(x.clone()),
            Group(x) => Group(x.clone()),
            Impl(x) => Impl(x.clone()),
            Infer(x) => Infer(x.clone()),
            Mac(x) => Mac(x.clone()),
            Never(x) => Never(x.clone()),
            Parenth(x) => Parenth(x.clone()),
            Path(x) => Path(x.clone()),
            Ptr(x) => Ptr(x.clone()),
            Ref(x) => Ref(x.clone()),
            Slice(x) => Slice(x.clone()),
            Trait(x) => Trait(x.clone()),
            Tuple(x) => Tuple(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Clone for typ::Array {
    fn clone(&self) -> Self {
        typ::Array {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
            semi: self.semi.clone(),
            len: self.len.clone(),
        }
    }
}
impl Clone for typ::Fn {
    fn clone(&self) -> Self {
        typ::Fn {
            lifes: self.lifes.clone(),
            unsafe_: self.unsafe_.clone(),
            abi: self.abi.clone(),
            fn_: self.fn_.clone(),
            parenth: self.parenth.clone(),
            args: self.args.clone(),
            vari: self.vari.clone(),
            ret: self.ret.clone(),
        }
    }
}
impl Clone for typ::Group {
    fn clone(&self) -> Self {
        typ::Group {
            group: self.group.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for typ::Impl {
    fn clone(&self) -> Self {
        typ::Impl {
            impl_: self.impl_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for typ::Infer {
    fn clone(&self) -> Self {
        typ::Infer {
            underscore: self.underscore.clone(),
        }
    }
}
impl Clone for typ::Mac {
    fn clone(&self) -> Self {
        typ::Mac { mac: self.mac.clone() }
    }
}
impl Clone for typ::Never {
    fn clone(&self) -> Self {
        typ::Never {
            bang: self.bang.clone(),
        }
    }
}
impl Clone for gen::param::Type {
    fn clone(&self) -> Self {
        gen::param::Type {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            colon: self.colon.clone(),
            bounds: self.bounds.clone(),
            eq: self.eq.clone(),
            default: self.default.clone(),
        }
    }
}
impl Clone for gen::bound::Type {
    fn clone(&self) -> Self {
        use gen::bound::Type::*;
        match self {
            Life(x) => Life(x.clone()),
            Trait(x) => Trait(x.clone()),
            Verbatim(x) => Verbatim(x.clone()),
        }
    }
}
impl Clone for typ::Parenth {
    fn clone(&self) -> Self {
        typ::Parenth {
            parenth: self.parenth.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for typ::Path {
    fn clone(&self) -> Self {
        typ::Path {
            qself: self.qself.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for typ::Ptr {
    fn clone(&self) -> Self {
        typ::Ptr {
            star: self.star.clone(),
            const_: self.const_.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for typ::Ref {
    fn clone(&self) -> Self {
        typ::Ref {
            and: self.and.clone(),
            life: self.life.clone(),
            mut_: self.mut_.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for typ::Slice {
    fn clone(&self) -> Self {
        typ::Slice {
            bracket: self.bracket.clone(),
            elem: self.elem.clone(),
        }
    }
}
impl Clone for typ::Trait {
    fn clone(&self) -> Self {
        typ::Trait {
            dyn_: self.dyn_.clone(),
            bounds: self.bounds.clone(),
        }
    }
}
impl Clone for typ::Tuple {
    fn clone(&self) -> Self {
        typ::Tuple {
            parenth: self.parenth.clone(),
            elems: self.elems.clone(),
        }
    }
}
impl Copy for expr::UnOp {}
impl Clone for expr::UnOp {
    fn clone(&self) -> Self {
        *self
    }
}
impl Clone for item::use_::Glob {
    fn clone(&self) -> Self {
        item::use_::Glob {
            star: self.star.clone(),
        }
    }
}
impl Clone for item::use_::Group {
    fn clone(&self) -> Self {
        item::use_::Group {
            brace: self.brace.clone(),
            trees: self.trees.clone(),
        }
    }
}
impl Clone for item::use_::Name {
    fn clone(&self) -> Self {
        item::use_::Name {
            ident: self.ident.clone(),
        }
    }
}
impl Clone for item::use_::Path {
    fn clone(&self) -> Self {
        item::use_::Path {
            ident: self.ident.clone(),
            colon2: self.colon2.clone(),
            tree: self.tree.clone(),
        }
    }
}
impl Clone for item::use_::Rename {
    fn clone(&self) -> Self {
        item::use_::Rename {
            ident: self.ident.clone(),
            as_: self.as_.clone(),
            rename: self.rename.clone(),
        }
    }
}
impl Clone for item::use_::Tree {
    fn clone(&self) -> Self {
        use item::use_::Tree::*;
        match self {
            Glob(x) => Glob(x.clone()),
            Group(x) => Group(x.clone()),
            Name(x) => Name(x.clone()),
            Path(x) => Path(x.clone()),
            Rename(x) => Rename(x.clone()),
        }
    }
}
impl Clone for item::Variadic {
    fn clone(&self) -> Self {
        item::Variadic {
            attrs: self.attrs.clone(),
            pat: self.pat.clone(),
            dots: self.dots.clone(),
            comma: self.comma.clone(),
        }
    }
}
impl Clone for data::Variant {
    fn clone(&self) -> Self {
        data::Variant {
            attrs: self.attrs.clone(),
            ident: self.ident.clone(),
            fields: self.fields.clone(),
            discrim: self.discrim.clone(),
        }
    }
}
impl Clone for data::Restricted {
    fn clone(&self) -> Self {
        data::Restricted {
            pub_: self.pub_.clone(),
            parenth: self.parenth.clone(),
            in_: self.in_.clone(),
            path: self.path.clone(),
        }
    }
}
impl Clone for data::Visibility {
    fn clone(&self) -> Self {
        use data::Visibility::*;
        match self {
            Inherited => Inherited,
            Public(x) => Public(x.clone()),
            Restricted(x) => Restricted(x.clone()),
        }
    }
}
impl Clone for gen::Where {
    fn clone(&self) -> Self {
        gen::Where {
            where_: self.where_.clone(),
            preds: self.preds.clone(),
        }
    }
}
impl Clone for gen::where_::Pred {
    fn clone(&self) -> Self {
        use gen::where_::Pred::*;
        match self {
            Life(x) => Life(x.clone()),
            Type(x) => Type(x.clone()),
        }
    }
}
