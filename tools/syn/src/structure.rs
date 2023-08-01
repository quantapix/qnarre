use super::{
    parse::{parse::Stream, Parser},
    visit::{self, Visit},
    Error, Field, Fields, FieldsNamed, FieldsUnnamed, GenericParam, Ident, PredicateType, Result, Token, TraitBound,
    Type, TypeMacro, TypeParamBound, TypePath, WhereClause, WherePredicate, *,
};
use std::collections::HashSet;
use unicode_xid::UnicodeXID;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AddBounds {
    Both,
    Fields,
    Gens,
    None,
    __Nonexhaustive,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BindStyle {
    Move,
    MoveMut,
    Ref,
    RefMut,
}
impl Lower for BindStyle {
    fn lower(&self, s: &mut pm2::Stream) {
        match self {
            BindStyle::Move => {},
            BindStyle::MoveMut => quote_spanned!(pm2::Span::call_site() => mut).lower(s),
            BindStyle::Ref => quote_spanned!(pm2::Span::call_site() => ref).lower(s),
            BindStyle::RefMut => quote_spanned!(pm2::Span::call_site() => ref mut).lower(s),
        }
    }
}
fn generics_fuse(res: &mut Vec<bool>, new: &[bool]) {
    for (i, &flag) in new.iter().enumerate() {
        if i == res.len() {
            res.push(false);
        }
        if flag {
            res[i] = true;
        }
    }
}
fn fetch_generics<'a>(set: &[bool], generics: &'a gen::Gens) -> Vec<&'a Ident> {
    let mut tys = vec![];
    for (&seen, param) in set.iter().zip(generics.params.iter()) {
        if seen {
            if let GenericParam::Type(tparam) = param {
                tys.push(&tparam.ident);
            }
        }
    }
    tys
}
fn sanitize_ident(s: &str) -> Ident {
    let mut res = String::with_capacity(s.len());
    for mut c in s.chars() {
        if !UnicodeXID::is_xid_continue(c) {
            c = '_';
        }
        if res.ends_with('_') && c == '_' {
            continue;
        }
        res.push(c);
    }
    Ident::new(&res, pm2::Span::call_site())
}
fn merge_generics(into: &mut gen::Gens, from: &gen::Gens) -> Result<()> {
    for p in &from.params {
        for op in &into.params {
            match (op, p) {
                (GenericParam::Type(otp), GenericParam::Type(tp)) => {
                    if otp.ident == tp.ident {
                        return Err(Error::new_spanned(
                            p,
                            format!(
                                "Attempted to merge conflicting generic parameters: {} and {}",
                                quote!(#op),
                                quote!(#p)
                            ),
                        ));
                    }
                },
                (GenericParam::Lifetime(olp), GenericParam::Lifetime(lp)) => {
                    if olp.lifetime == lp.lifetime {
                        return Err(Error::new_spanned(
                            p,
                            format!(
                                "Attempted to merge conflicting generic parameters: {} and {}",
                                quote!(#op),
                                quote!(#p)
                            ),
                        ));
                    }
                },
                _ => (),
            }
        }
        into.params.push(p.clone());
    }
    if let Some(from_clause) = &from.where_clause {
        into.make_where_clause()
            .predicates
            .extend(from_clause.predicates.iter().cloned());
    }
    Ok(())
}
fn get_or_insert_with<T, F>(opt: &mut Option<T>, f: F) -> &mut T
where
    F: FnOnce() -> T,
{
    if opt.is_none() {
        *opt = Some(f());
    }
    match opt {
        Some(v) => v,
        None => unreachable!(),
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingInfo<'a> {
    pub binding: Ident,
    pub style: BindStyle,
    field: &'a Field,
    gens: &'a gen::Gens,
    seen_generics: Vec<bool>,
    index: usize,
}
impl<'a> Lower for BindingInfo<'a> {
    fn lower(&self, s: &mut pm2::Stream) {
        self.binding.lower(s);
    }
}
impl<'a> BindingInfo<'a> {
    pub fn ast(&self) -> &'a Field {
        self.field
    }
    pub fn pat(&self) -> pm2::Stream {
        let BindingInfo { binding, style, .. } = self;
        quote!(#style #binding)
    }
    pub fn referenced_ty_params(&self) -> Vec<&'a Ident> {
        fetch_generics(&self.seen_generics, self.gens)
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VariantAst<'a> {
    pub attrs: &'a [attr::Attr],
    pub ident: &'a Ident,
    pub fields: &'a Fields,
    pub discriminant: &'a Option<(token::Eq, expr::Expr)>,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantInfo<'a> {
    pub prefix: Option<&'a Ident>,
    bindings: Vec<BindingInfo<'a>>,
    ast: VariantAst<'a>,
    gens: &'a gen::Gens,
    original_length: usize,
}
fn get_ty_params(field: &Field, generics: &gen::Gens) -> Vec<bool> {
    struct BoundTypeLocator<'a> {
        result: Vec<bool>,
        gens: &'a gen::Gens,
    }
    impl<'a> Visit<'a> for BoundTypeLocator<'a> {
        fn visit_ident(&mut self, id: &Ident) {
            for (idx, i) in self.gens.params.iter().enumerate() {
                if let GenericParam::Type(tparam) = i {
                    if tparam.ident == *id {
                        self.result[idx] = true;
                    }
                }
            }
        }
        fn visit_type_macro(&mut self, x: &'a TypeMacro) {
            for r in &mut self.result {
                *r = true;
            }
            visit::visit_type_macro(self, x);
        }
    }
    let mut btl = BoundTypeLocator {
        result: vec![false; generics.params.len()],
        gens: generics,
    };
    btl.visit_type(&field.ty);
    btl.result
}
impl<'a> VariantInfo<'a> {
    fn new(ast: VariantAst<'a>, prefix: Option<&'a Ident>, generics: &'a gen::Gens) -> Self {
        let bindings = match ast.fields {
            Fields::Unit => vec![],
            Fields::Unnamed(FieldsUnnamed { unnamed: fields, .. })
            | Fields::Named(FieldsNamed { named: fields, .. }) => fields
                .into_iter()
                .enumerate()
                .map(|(i, field)| BindingInfo {
                    binding: format_ident!("__binding_{}", i),
                    style: BindStyle::Ref,
                    field,
                    gens: generics,
                    seen_generics: get_ty_params(field, generics),
                    index: i,
                })
                .collect::<Vec<_>>(),
        };
        let original_length = bindings.len();
        VariantInfo {
            prefix,
            bindings,
            ast,
            gens: generics,
            original_length,
        }
    }
    pub fn bindings(&self) -> &[BindingInfo<'a>] {
        &self.bindings
    }
    pub fn bindings_mut(&mut self) -> &mut [BindingInfo<'a>] {
        &mut self.bindings
    }
    pub fn ast(&self) -> VariantAst<'a> {
        self.ast
    }
    pub fn omitted_bindings(&self) -> bool {
        self.original_length != self.bindings.len()
    }
    pub fn pat(&self) -> pm2::Stream {
        let mut t = pm2::Stream::new();
        if let Some(prefix) = self.prefix {
            prefix.lower(&mut t);
            quote!(::).lower(&mut t);
        }
        self.ast.ident.lower(&mut t);
        match self.ast.fields {
            Fields::Unit => {
                assert!(self.bindings.is_empty());
            },
            Fields::Unnamed(..) => tok::Paren(pm2::Span::call_site()).surround(&mut t, |t| {
                let mut expected_index = 0;
                for binding in &self.bindings {
                    while expected_index < binding.index {
                        quote!(_,).lower(t);
                        expected_index += 1;
                    }
                    binding.pat().lower(t);
                    quote!(,).lower(t);
                    expected_index += 1;
                }
                if expected_index != self.original_length {
                    quote!(..).lower(t);
                }
            }),
            Fields::Named(..) => tok::Brace(pm2::Span::call_site()).surround(&mut t, |t| {
                for binding in &self.bindings {
                    binding.field.ident.lower(t);
                    quote!(:).lower(t);
                    binding.pat().lower(t);
                    quote!(,).lower(t);
                }
                if self.omitted_bindings() {
                    quote!(..).lower(t);
                }
            }),
        }
        t
    }
    pub fn construct<F, T>(&self, mut func: F) -> pm2::Stream
    where
        F: FnMut(&Field, usize) -> T,
        T: Lower,
    {
        let mut t = pm2::Stream::new();
        if let Some(prefix) = self.prefix {
            quote!(#prefix ::).lower(&mut t);
        }
        self.ast.ident.lower(&mut t);
        match &self.ast.fields {
            Fields::Unit => (),
            Fields::Unnamed(FieldsUnnamed { unnamed, .. }) => {
                tok::Paren::default().surround(&mut t, |t| {
                    for (i, field) in unnamed.into_iter().enumerate() {
                        func(field, i).lower(t);
                        quote!(,).lower(t);
                    }
                });
            },
            Fields::Named(FieldsNamed { named, .. }) => {
                tok::Brace::default().surround(&mut t, |t| {
                    for (i, field) in named.into_iter().enumerate() {
                        field.ident.lower(t);
                        quote!(:).lower(t);
                        func(field, i).lower(t);
                        quote!(,).lower(t);
                    }
                });
            },
        }
        t
    }
    pub fn each<F, R>(&self, mut f: F) -> pm2::Stream
    where
        F: FnMut(&BindingInfo<'_>) -> R,
        R: Lower,
    {
        let pat = self.pat();
        let mut body = pm2::Stream::new();
        for binding in &self.bindings {
            tok::Brace::default().surround(&mut body, |body| {
                f(binding).lower(body);
            });
        }
        quote!(#pat => { #body })
    }
    pub fn fold<F, I, R>(&self, init: I, mut f: F) -> pm2::Stream
    where
        F: FnMut(pm2::Stream, &BindingInfo<'_>) -> R,
        I: Lower,
        R: Lower,
    {
        let pat = self.pat();
        let body = self.bindings.iter().fold(quote!(#init), |i, bi| {
            let r = f(i, bi);
            quote!(#r)
        });
        quote!(#pat => { #body })
    }
    pub fn filter<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&BindingInfo<'_>) -> bool,
    {
        self.bindings.retain(f);
        self
    }
    #[allow(clippy::return_self_not_must_use)]
    pub fn drain_filter<F>(&mut self, mut f: F) -> Self
    where
        F: FnMut(&BindingInfo<'_>) -> bool,
    {
        let mut other = VariantInfo {
            prefix: self.prefix,
            bindings: vec![],
            ast: self.ast,
            gens: self.gens,
            original_length: self.original_length,
        };
        let (other_bindings, self_bindings) = self.bindings.drain(..).partition(&mut f);
        other.bindings = other_bindings;
        self.bindings = self_bindings;
        other
    }
    pub fn remove_binding(&mut self, idx: usize) -> &mut Self {
        self.bindings.remove(idx);
        self
    }
    pub fn bind_with<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&BindingInfo<'_>) -> BindStyle,
    {
        for binding in &mut self.bindings {
            binding.style = f(binding);
        }
        self
    }
    pub fn binding_name<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&Field, usize) -> Ident,
    {
        for (it, binding) in self.bindings.iter_mut().enumerate() {
            binding.binding = f(binding.field, it);
        }
        self
    }
    pub fn referenced_ty_params(&self) -> Vec<&'a Ident> {
        let mut flags = Vec::new();
        for binding in &self.bindings {
            generics_fuse(&mut flags, &binding.seen_generics);
        }
        fetch_generics(&flags, self.gens)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Structure<'a> {
    variants: Vec<VariantInfo<'a>>,
    omitted_variants: bool,
    underscore_const: bool,
    ast: &'a data::Input,
    extra_impl: Vec<GenericParam>,
    extra_predicates: Vec<WherePredicate>,
    add_bounds: AddBounds,
}
impl<'a> Structure<'a> {
    pub fn new(ast: &'a data::Input) -> Self {
        Self::try_new(ast).expect("Unable to create synstructure::Structure")
    }
    pub fn try_new(ast: &'a data::Input) -> Result<Self> {
        let variants = match &ast.data {
            data::Data::Enum(data) => (&data.variants)
                .into_iter()
                .map(|v| {
                    VariantInfo::new(
                        VariantAst {
                            attrs: &v.attrs,
                            ident: &v.ident,
                            fields: &v.fields,
                            discriminant: &v.discriminant,
                        },
                        Some(&ast.ident),
                        &ast.generics,
                    )
                })
                .collect::<Vec<_>>(),
            data::Data::Struct(data) => {
                vec![VariantInfo::new(
                    VariantAst {
                        attrs: &ast.attrs,
                        ident: &ast.ident,
                        fields: &data.fields,
                        discriminant: &None,
                    },
                    None,
                    &ast.generics,
                )]
            },
            data::Data::Union(_) => {
                return Err(Error::new_spanned(ast, "unexpected unsupported untagged union"));
            },
        };
        Ok(Structure {
            variants,
            omitted_variants: false,
            underscore_const: false,
            ast,
            extra_impl: vec![],
            extra_predicates: vec![],
            add_bounds: AddBounds::Both,
        })
    }
    pub fn variants(&self) -> &[VariantInfo<'a>] {
        &self.variants
    }
    pub fn variants_mut(&mut self) -> &mut [VariantInfo<'a>] {
        &mut self.variants
    }
    pub fn ast(&self) -> &'a data::Input {
        self.ast
    }
    pub fn omitted_variants(&self) -> bool {
        self.omitted_variants
    }
    pub fn each<F, R>(&self, mut f: F) -> pm2::Stream
    where
        F: FnMut(&BindingInfo<'_>) -> R,
        R: Lower,
    {
        let mut t = pm2::Stream::new();
        for variant in &self.variants {
            variant.each(&mut f).lower(&mut t);
        }
        if self.omitted_variants {
            quote!(_ => {}).lower(&mut t);
        }
        t
    }
    pub fn fold<F, I, R>(&self, init: I, mut f: F) -> pm2::Stream
    where
        F: FnMut(pm2::Stream, &BindingInfo<'_>) -> R,
        I: Lower,
        R: Lower,
    {
        let mut t = pm2::Stream::new();
        for variant in &self.variants {
            variant.fold(&init, &mut f).lower(&mut t);
        }
        if self.omitted_variants {
            quote!(_ => { #init }).lower(&mut t);
        }
        t
    }
    pub fn each_variant<F, R>(&self, mut f: F) -> pm2::Stream
    where
        F: FnMut(&VariantInfo<'_>) -> R,
        R: Lower,
    {
        let mut t = pm2::Stream::new();
        for variant in &self.variants {
            let pat = variant.pat();
            let body = f(variant);
            quote!(#pat => { #body }).lower(&mut t);
        }
        if self.omitted_variants {
            quote!(_ => {}).lower(&mut t);
        }
        t
    }
    pub fn filter<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&BindingInfo<'_>) -> bool,
    {
        for variant in &mut self.variants {
            variant.filter(&mut f);
        }
        self
    }
    #[allow(clippy::return_self_not_must_use)]
    pub fn drain_filter<F>(&mut self, mut f: F) -> Self
    where
        F: FnMut(&BindingInfo<'_>) -> bool,
    {
        Self {
            variants: self
                .variants
                .iter_mut()
                .map(|variant| variant.drain_filter(&mut f))
                .collect(),
            omitted_variants: self.omitted_variants,
            underscore_const: self.underscore_const,
            ast: self.ast,
            extra_impl: self.extra_impl.clone(),
            extra_predicates: self.extra_predicates.clone(),
            add_bounds: self.add_bounds,
        }
    }
    pub fn add_where_predicate(&mut self, pred: WherePredicate) -> &mut Self {
        self.extra_predicates.push(pred);
        self
    }
    pub fn add_bounds(&mut self, mode: AddBounds) -> &mut Self {
        self.add_bounds = mode;
        self
    }
    pub fn filter_variants<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&VariantInfo<'_>) -> bool,
    {
        let before_len = self.variants.len();
        self.variants.retain(f);
        if self.variants.len() != before_len {
            self.omitted_variants = true;
        }
        self
    }
    #[allow(clippy::return_self_not_must_use)]
    pub fn drain_filter_variants<F>(&mut self, mut f: F) -> Self
    where
        F: FnMut(&VariantInfo<'_>) -> bool,
    {
        let mut other = Self {
            variants: vec![],
            omitted_variants: self.omitted_variants,
            underscore_const: self.underscore_const,
            ast: self.ast,
            extra_impl: self.extra_impl.clone(),
            extra_predicates: self.extra_predicates.clone(),
            add_bounds: self.add_bounds,
        };
        let (other_variants, self_variants) = self.variants.drain(..).partition(&mut f);
        other.variants = other_variants;
        self.variants = self_variants;
        other
    }
    pub fn remove_variant(&mut self, idx: usize) -> &mut Self {
        self.variants.remove(idx);
        self.omitted_variants = true;
        self
    }
    pub fn bind_with<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&BindingInfo<'_>) -> BindStyle,
    {
        for variant in &mut self.variants {
            variant.bind_with(&mut f);
        }
        self
    }
    pub fn binding_name<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&Field, usize) -> Ident,
    {
        for variant in &mut self.variants {
            variant.binding_name(&mut f);
        }
        self
    }
    pub fn referenced_ty_params(&self) -> Vec<&'a Ident> {
        let mut flags = Vec::new();
        for variant in &self.variants {
            for binding in &variant.bindings {
                generics_fuse(&mut flags, &binding.seen_generics);
            }
        }
        fetch_generics(&flags, &self.ast.generics)
    }
    pub fn add_impl_generic(&mut self, param: GenericParam) -> &mut Self {
        self.extra_impl.push(param);
        self
    }
    pub fn add_trait_bounds(&self, bound: &TraitBound, where_clause: &mut Option<WhereClause>, mode: AddBounds) {
        if !self.extra_predicates.is_empty() {
            let clause = get_or_insert_with(&mut *where_clause, || WhereClause {
                where_token: Default::default(),
                predicates: punctuated::Punctuated::new(),
            });
            clause.predicates.extend(self.extra_predicates.iter().cloned());
        }
        let mut seen = HashSet::new();
        let mut pred = |ty: Type| {
            if !seen.contains(&ty) {
                seen.insert(ty.clone());
                let clause = get_or_insert_with(&mut *where_clause, || WhereClause {
                    where_token: Default::default(),
                    predicates: punctuated::Punctuated::new(),
                });
                clause.predicates.push(WherePredicate::Type(PredicateType {
                    lifetimes: None,
                    bounded_ty: ty,
                    colon_token: Default::default(),
                    bounds: Some(punctuated::Pair::End(TypeParamBound::Trait(bound.clone())))
                        .into_iter()
                        .collect(),
                }));
            }
        };
        for variant in &self.variants {
            for binding in &variant.bindings {
                match mode {
                    AddBounds::Both | AddBounds::Fields => {
                        for &seen in &binding.seen_generics {
                            if seen {
                                pred(binding.ast().ty.clone());
                                break;
                            }
                        }
                    },
                    _ => {},
                }
                match mode {
                    AddBounds::Both | AddBounds::Gens => {
                        for param in binding.referenced_ty_params() {
                            pred(Type::Path(TypePath {
                                qself: None,
                                path: (*param).clone().into(),
                            }));
                        }
                    },
                    _ => {},
                }
            }
        }
    }
    pub fn underscore_const(&mut self, enabled: bool) -> &mut Self {
        self.underscore_const = enabled;
        self
    }
    pub fn bound_impl<P: Lower, B: Lower>(&self, path: P, body: B) -> pm2::Stream {
        self.impl_internal(path.into_token_stream(), body.into_token_stream(), quote!(), None)
    }
    pub fn unsafe_bound_impl<P: Lower, B: Lower>(&self, path: P, body: B) -> pm2::Stream {
        self.impl_internal(path.into_token_stream(), body.into_token_stream(), quote!(unsafe), None)
    }
    pub fn unbound_impl<P: Lower, B: Lower>(&self, path: P, body: B) -> pm2::Stream {
        self.impl_internal(
            path.into_token_stream(),
            body.into_token_stream(),
            quote!(),
            Some(AddBounds::None),
        )
    }
    #[deprecated]
    pub fn unsafe_unbound_impl<P: Lower, B: Lower>(&self, path: P, body: B) -> pm2::Stream {
        self.impl_internal(
            path.into_token_stream(),
            body.into_token_stream(),
            quote!(unsafe),
            Some(AddBounds::None),
        )
    }
    fn impl_internal(
        &self,
        path: pm2::Stream,
        body: pm2::Stream,
        safety: pm2::Stream,
        mode: Option<AddBounds>,
    ) -> pm2::Stream {
        let mode = mode.unwrap_or(self.add_bounds);
        let name = &self.ast.ident;
        let mut gen_clone = self.ast.generics.clone();
        gen_clone.params.extend(self.extra_impl.clone().into_iter());
        let (impl_generics, _, _) = gen_clone.split_for_impl();
        let (_, ty_generics, where_clause) = self.ast.generics.split_for_impl();
        let bound = syn::parse2::<TraitBound>(path).expect("`path` argument must be a valid rust trait bound");
        let mut where_clause = where_clause.cloned();
        self.add_trait_bounds(&bound, &mut where_clause, mode);
        let mut extern_crate = quote!();
        if bound.path.leading_colon.is_none() {
            if let Some(seg) = bound.path.segments.first() {
                let seg = &seg.ident;
                extern_crate = quote! { extern crate #seg; };
            }
        }
        let generated = quote! {
            #extern_crate
            #safety impl #impl_generics #bound for #name #ty_generics #where_clause {
                #body
            }
        };
        if self.underscore_const {
            quote! {
                const _: () = { #generated };
            }
        } else {
            let dummy_const: Ident = sanitize_ident(&format!(
                "_DERIVE_{}_FOR_{}",
                (&bound).into_token_stream(),
                name.into_token_stream(),
            ));
            quote! {
                #[allow(non_upper_case_globals)]
                const #dummy_const: () = {
                    #generated
                };
            }
        }
    }
    pub fn gen_impl(&self, cfg: pm2::Stream) -> pm2::Stream {
        Parser::parse2(
            |input: parse::Stream<'_>| -> Result<pm2::Stream> { self.gen_impl_parse(input, true) },
            cfg,
        )
        .expect("Failed to parse gen_impl")
    }
    fn gen_impl_parse(&self, input: parse::Stream<'_>, wrap: bool) -> Result<pm2::Stream> {
        fn parse_prefix(s: parse::Stream<'_>) -> Result<Option<Token![unsafe]>> {
            if s.parse::<Ident>()? != "gen" {
                return Err(s.error("Expected keyword `gen`"));
            }
            let safety = s.parse::<Option<Token![unsafe]>>()?;
            let _ = s.parse::<Token![impl]>()?;
            Ok(safety)
        }
        let mut before = vec![];
        loop {
            if parse_prefix(&input.fork()).is_ok() {
                break;
            }
            before.push(input.parse::<pm2::Tree>()?);
        }
        let safety = parse_prefix(input)?;
        let mut generics = input.parse::<gen::Gens>()?;
        let bound = input.parse::<TraitBound>()?;
        let _ = input.parse::<Token![for]>()?;
        let _ = input.parse::<Token![@]>()?;
        let _ = input.parse::<Token![Self]>()?;
        generics.where_clause = input.parse()?;
        let body;
        braced!(body in input);
        let body = body.parse::<pm2::Stream>()?;
        let maybe_next_impl = self.gen_impl_parse(&input.fork(), false);
        let mut after = input.parse::<pm2::Stream>()?;
        if let Ok(stream) = maybe_next_impl {
            after = stream;
        }
        assert!(input.is_empty(), "Should've consumed the rest of our input");
        /* Codegen Logic */
        let name = &self.ast.ident;
        if let Err(err) = merge_generics(&mut generics, &self.ast.generics) {
            return Ok(err.to_compile_error());
        }
        self.add_trait_bounds(&bound, &mut generics.where_clause, self.add_bounds);
        let (impl_generics, _, where_clause) = generics.split_for_impl();
        let (_, ty_generics, _) = self.ast.generics.split_for_impl();
        let generated = quote! {
            #(#before)*
            #safety impl #impl_generics #bound for #name #ty_generics #where_clause {
                #body
            }
            #after
        };
        if wrap {
            if self.underscore_const {
                Ok(quote! {
                    const _: () = { #generated };
                })
            } else {
                let dummy_const: Ident = sanitize_ident(&format!(
                    "_DERIVE_{}_FOR_{}",
                    (&bound).into_token_stream(),
                    name.into_token_stream(),
                ));
                Ok(quote! {
                    #[allow(non_upper_case_globals)]
                    const #dummy_const: () = {
                        #generated
                    };
                })
            }
        } else {
            Ok(generated)
        }
    }
}
pub fn unpretty_print<T: std::fmt::Display>(ts: T) -> String {
    let mut res = String::new();
    let raw_s = ts.to_string();
    let mut s = &raw_s[..];
    let mut indent = 0;
    while let Some(i) = s.find(&['(', '{', '[', ')', '}', ']', ';'][..]) {
        match &s[i..=i] {
            "(" | "{" | "[" => indent += 1,
            ")" | "}" | "]" => indent -= 1,
            _ => {},
        }
        res.push_str(&s[..=i]);
        res.push('\n');
        for _ in 0..indent {
            res.push_str("    ");
        }
        s = trim_start_matches(&s[i + 1..], ' ');
    }
    res.push_str(s);
    res
}
#[allow(deprecated)]
fn trim_start_matches(s: &str, c: char) -> &str {
    s.trim_left_matches(c)
}
pub trait MacroResult {
    fn into_result(self) -> Result<pm2::Stream>;
    fn into_stream(self) -> pm2::Stream
    where
        Self: Sized,
    {
        match self.into_result() {
            Ok(ts) => ts.into(),
            Err(err) => err.to_compile_error().into(),
        }
    }
}
impl MacroResult for pm2::Stream {
    fn into_result(self) -> Result<pm2::Stream> {
        Ok(self.into())
    }
    fn into_stream(self) -> pm2::Stream {
        self
    }
}
impl MacroResult for pm2::Stream {
    fn into_result(self) -> Result<pm2::Stream> {
        Ok(self)
    }
}
impl<T: MacroResult> MacroResult for Result<T> {
    fn into_result(self) -> Result<pm2::Stream> {
        match self {
            Ok(v) => v.into_result(),
            Err(err) => Err(err),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_each_enum() {
        let di: syn::data::Input = syn::parse_quote! {
         enum A {
             Foo(usize, bool),
             Bar(bool, usize),
             Baz(usize, bool, usize),
             Quux(bool, usize, bool)
         }
        };
        let mut s = Structure::new(&di);
        s.filter(|bi| bi.ast().ty.to_token_stream().to_string() == "bool");
        assert_eq!(
            s.each(|bi| quote!(do_something(#bi))).to_string(),
            quote! {
                A::Foo(_, ref __binding_1,) => { { do_something(__binding_1) } }
                A::Bar(ref __binding_0, ..) => { { do_something(__binding_0) } }
                A::Baz(_, ref __binding_1, ..) => { { do_something(__binding_1) } }
                A::Quux(ref __binding_0, _, ref __binding_2,) => {
                    {
                        do_something(__binding_0)
                    }
                    {
                        do_something(__binding_2)
                    }
                }
            }
            .to_string()
        );
    }
}
