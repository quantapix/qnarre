use super::{
    err::{Err, Result},
    ext::IdentExt,
    parse::{discouraged::Speculative, Parse, ParseBuffer, ParseStream, Result},
    tok::Token,
    *,
};
use proc_macro2::{Ident, Punct, Spacing, Span, Span, TokenStream};
use std::{
    cmp::Ordering,
    fmt::{self, Display},
};

pub(crate) fn parse_inner(x: ParseStream, ys: &mut Vec<Attribute>) -> Result<()> {
    while x.peek(Token![#]) && x.peek2(Token![!]) {
        ys.push(x.call(single_parse_inner)?);
    }
    Ok(())
}
pub(crate) fn single_parse_inner(x: ParseStream) -> Result<Attribute> {
    let content;
    Ok(Attribute {
        pound: x.parse()?,
        style: AttrStyle::Inner(x.parse()?),
        bracket: bracketed!(content in x),
        meta: content.parse()?,
    })
}
pub(crate) fn single_parse_outer(x: ParseStream) -> Result<Attribute> {
    let content;
    Ok(Attribute {
        pound: x.parse()?,
        style: AttrStyle::Outer,
        bracket: bracketed!(content in x),
        meta: content.parse()?,
    })
}

impl Parse for Meta {
    fn parse(x: ParseStream) -> Result<Self> {
        let path = x.call(Path::parse_mod_style)?;
        parse_meta_after_path(path, x)
    }
}
impl Parse for MetaList {
    fn parse(x: ParseStream) -> Result<Self> {
        let path = x.call(Path::parse_mod_style)?;
        parse_meta_list_after_path(path, x)
    }
}
impl Parse for MetaNameValue {
    fn parse(x: ParseStream) -> Result<Self> {
        let path = x.call(Path::parse_mod_style)?;
        parse_meta_name_value_after_path(path, x)
    }
}

pub(crate) fn parse_meta_after_path(path: Path, x: ParseStream) -> Result<Meta> {
    if x.peek(tok::Paren) || x.peek(tok::Bracket) || x.peek(tok::Brace) {
        parse_meta_list_after_path(path, x).map(Meta::List)
    } else if x.peek(Token![=]) {
        parse_meta_name_value_after_path(path, x).map(Meta::NameValue)
    } else {
        Ok(Meta::Path(path))
    }
}
fn parse_meta_list_after_path(path: Path, x: ParseStream) -> Result<MetaList> {
    let (delimiter, tokens) = mac_parse_delimiter(x)?;
    Ok(MetaList {
        path,
        delim: delimiter,
        toks: tokens,
    })
}
fn parse_meta_name_value_after_path(path: Path, x: ParseStream) -> Result<MetaNameValue> {
    let eq_token: Token![=] = x.parse()?;
    let ahead = x.fork();
    let lit: Option<Lit> = ahead.parse()?;
    let value = if let (Some(lit), true) = (lit, ahead.is_empty()) {
        x.advance_to(&ahead);
        Expr::Lit(ExprLit { attrs: Vec::new(), lit })
    } else if x.peek(Token![#]) && x.peek2(tok::Bracket) {
        return Err(x.error("unexpected attribute inside of attribute"));
    } else {
        x.parse()?
    };
    Ok(MetaNameValue {
        path,
        equal: eq_token,
        val: value,
    })
}

pub(super) struct DisplayAttrStyle<'a>(pub &'a AttrStyle);
impl<'a> Display for DisplayAttrStyle<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self.0 {
            AttrStyle::Outer => "#",
            AttrStyle::Inner(_) => "#!",
        })
    }
}

pub(super) struct DisplayPath<'a>(pub &'a Path);
impl<'a> Display for DisplayPath<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, x) in self.0.segments.iter().enumerate() {
            if i > 0 || self.0.leading_colon.is_some() {
                f.write_str("::")?;
            }
            write!(f, "{}", x.ident)?;
        }
        Ok(())
    }
}

impl Parse for Generics {
    fn parse(input: ParseStream) -> Result<Self> {
        if !input.peek(Token![<]) {
            return Ok(Generics::default());
        }
        let lt_token: Token![<] = input.parse()?;
        let mut params = Punctuated::new();
        loop {
            if input.peek(Token![>]) {
                break;
            }
            let attrs = input.call(Attribute::parse_outer)?;
            let lookahead = input.lookahead1();
            if lookahead.peek(Lifetime) {
                params.push_value(GenericParam::Lifetime(LifetimeParam {
                    attrs,
                    ..input.parse()?
                }));
            } else if lookahead.peek(Ident) {
                params.push_value(GenericParam::Type(TypeParam {
                    attrs,
                    ..input.parse()?
                }));
            } else if lookahead.peek(Token![const]) {
                params.push_value(GenericParam::Const(ConstParam {
                    attrs,
                    ..input.parse()?
                }));
            } else if input.peek(Token![_]) {
                params.push_value(GenericParam::Type(TypeParam {
                    attrs,
                    ident: input.call(Ident::parse_any)?,
                    colon: None,
                    bounds: Punctuated::new(),
                    equal: None,
                    default: None,
                }));
            } else {
                return Err(lookahead.error());
            }
            if input.peek(Token![>]) {
                break;
            }
            let punct = input.parse()?;
            params.push_punct(punct);
        }
        let gt_token: Token![>] = input.parse()?;
        Ok(Generics {
            lt: Some(lt_token),
            params,
            gt: Some(gt_token),
            clause: None,
        })
    }
}
impl Parse for GenericParam {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let lookahead = input.lookahead1();
        if lookahead.peek(Ident) {
            Ok(GenericParam::Type(TypeParam {
                attrs,
                ..input.parse()?
            }))
        } else if lookahead.peek(Lifetime) {
            Ok(GenericParam::Lifetime(LifetimeParam {
                attrs,
                ..input.parse()?
            }))
        } else if lookahead.peek(Token![const]) {
            Ok(GenericParam::Const(ConstParam {
                attrs,
                ..input.parse()?
            }))
        } else {
            Err(lookahead.error())
        }
    }
}
impl Parse for LifetimeParam {
    fn parse(input: ParseStream) -> Result<Self> {
        let has_colon;
        Ok(LifetimeParam {
            attrs: input.call(Attribute::parse_outer)?,
            lifetime: input.parse()?,
            colon: {
                if input.peek(Token![:]) {
                    has_colon = true;
                    Some(input.parse()?)
                } else {
                    has_colon = false;
                    None
                }
            },
            bounds: {
                let mut bounds = Punctuated::new();
                if has_colon {
                    loop {
                        if input.peek(Token![,]) || input.peek(Token![>]) {
                            break;
                        }
                        let value = input.parse()?;
                        bounds.push_value(value);
                        if !input.peek(Token![+]) {
                            break;
                        }
                        let punct = input.parse()?;
                        bounds.push_punct(punct);
                    }
                }
                bounds
            },
        })
    }
}
impl Parse for BoundLifetimes {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(BoundLifetimes {
            for_: input.parse()?,
            lt: input.parse()?,
            lifetimes: {
                let mut lifetimes = Punctuated::new();
                while !input.peek(Token![>]) {
                    let attrs = input.call(Attribute::parse_outer)?;
                    let lifetime: Lifetime = input.parse()?;
                    lifetimes.push_value(GenericParam::Lifetime(LifetimeParam {
                        attrs,
                        lifetime,
                        colon: None,
                        bounds: Punctuated::new(),
                    }));
                    if input.peek(Token![>]) {
                        break;
                    }
                    lifetimes.push_punct(input.parse()?);
                }
                lifetimes
            },
            gt: input.parse()?,
        })
    }
}
impl Parse for Option<BoundLifetimes> {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![for]) {
            input.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Parse for TypeParam {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let ident: Ident = input.parse()?;
        let colon_token: Option<Token![:]> = input.parse()?;
        let mut bounds = Punctuated::new();
        if colon_token.is_some() {
            loop {
                if input.peek(Token![,]) || input.peek(Token![>]) || input.peek(Token![=]) {
                    break;
                }
                let value: TypeParamBound = input.parse()?;
                bounds.push_value(value);
                if !input.peek(Token![+]) {
                    break;
                }
                let punct: Token![+] = input.parse()?;
                bounds.push_punct(punct);
            }
        }
        let eq_token: Option<Token![=]> = input.parse()?;
        let default = if eq_token.is_some() {
            Some(input.parse::<Type>()?)
        } else {
            None
        };
        Ok(TypeParam {
            attrs,
            ident,
            colon: colon_token,
            bounds,
            equal: eq_token,
            default,
        })
    }
}
impl Parse for TypeParamBound {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Lifetime) {
            return input.parse().map(TypeParamBound::Lifetime);
        }
        let begin = input.fork();
        let content;
        let (paren_token, content) = if input.peek(tok::Paren) {
            (Some(parenthesized!(content in input)), &content)
        } else {
            (None, input)
        };
        let is_tilde_const = cfg!(feature = "full") && content.peek(Token![~]) && content.peek2(Token![const]);
        if is_tilde_const {
            content.parse::<Token![~]>()?;
            content.parse::<Token![const]>()?;
        }
        let mut bound: TraitBound = content.parse()?;
        bound.paren = paren_token;
        if is_tilde_const {
            Ok(TypeParamBound::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(TypeParamBound::Trait(bound))
        }
    }
}

impl TypeParamBound {
    pub(crate) fn parse_multiple(input: ParseStream, allow_plus: bool) -> Result<Punctuated<Self, Token![+]>> {
        let mut bounds = Punctuated::new();
        loop {
            bounds.push_value(input.parse()?);
            if !(allow_plus && input.peek(Token![+])) {
                break;
            }
            bounds.push_punct(input.parse()?);
            if !(input.peek(Ident::peek_any)
                || input.peek(Token![::])
                || input.peek(Token![?])
                || input.peek(Lifetime)
                || input.peek(tok::Paren)
                || input.peek(Token![~]))
            {
                break;
            }
        }
        Ok(bounds)
    }
}

impl Parse for TraitBound {
    fn parse(input: ParseStream) -> Result<Self> {
        let modifier: TraitBoundModifier = input.parse()?;
        let lifetimes: Option<BoundLifetimes> = input.parse()?;
        let mut path: Path = input.parse()?;
        if path.segments.last().unwrap().arguments.is_empty()
            && (input.peek(tok::Paren) || input.peek(Token![::]) && input.peek3(tok::Paren))
        {
            input.parse::<Option<Token![::]>>()?;
            let args: ParenthesizedGenericArguments = input.parse()?;
            let parenthesized = PathArguments::Parenthesized(args);
            path.segments.last_mut().unwrap().arguments = parenthesized;
        }
        Ok(TraitBound {
            paren: None,
            modifier,
            lifetimes,
            path,
        })
    }
}
impl Parse for TraitBoundModifier {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![?]) {
            input.parse().map(TraitBoundModifier::Maybe)
        } else {
            Ok(TraitBoundModifier::None)
        }
    }
}
impl Parse for ConstParam {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut default = None;
        Ok(ConstParam {
            attrs: input.call(Attribute::parse_outer)?,
            const_: input.parse()?,
            ident: input.parse()?,
            colon: input.parse()?,
            ty: input.parse()?,
            equal: {
                if input.peek(Token![=]) {
                    let eq_token = input.parse()?;
                    default = Some(const_argument(input)?);
                    Some(eq_token)
                } else {
                    None
                }
            },
            default,
        })
    }
}
impl Parse for WhereClause {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(WhereClause {
            where_: input.parse()?,
            preds: {
                let mut predicates = Punctuated::new();
                loop {
                    if input.is_empty()
                        || input.peek(tok::Brace)
                        || input.peek(Token![,])
                        || input.peek(Token![;])
                        || input.peek(Token![:]) && !input.peek(Token![::])
                        || input.peek(Token![=])
                    {
                        break;
                    }
                    let value = input.parse()?;
                    predicates.push_value(value);
                    if !input.peek(Token![,]) {
                        break;
                    }
                    let punct = input.parse()?;
                    predicates.push_punct(punct);
                }
                predicates
            },
        })
    }
}
impl Parse for Option<WhereClause> {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![where]) {
            input.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Parse for WherePred {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Lifetime) && input.peek2(Token![:]) {
            Ok(WherePred::Lifetime(PredLifetime {
                lifetime: input.parse()?,
                colon: input.parse()?,
                bounds: {
                    let mut bounds = Punctuated::new();
                    loop {
                        if input.is_empty()
                            || input.peek(tok::Brace)
                            || input.peek(Token![,])
                            || input.peek(Token![;])
                            || input.peek(Token![:])
                            || input.peek(Token![=])
                        {
                            break;
                        }
                        let value = input.parse()?;
                        bounds.push_value(value);
                        if !input.peek(Token![+]) {
                            break;
                        }
                        let punct = input.parse()?;
                        bounds.push_punct(punct);
                    }
                    bounds
                },
            }))
        } else {
            Ok(WherePred::Type(PredType {
                lifetimes: input.parse()?,
                bounded_ty: input.parse()?,
                colon: input.parse()?,
                bounds: {
                    let mut bounds = Punctuated::new();
                    loop {
                        if input.is_empty()
                            || input.peek(tok::Brace)
                            || input.peek(Token![,])
                            || input.peek(Token![;])
                            || input.peek(Token![:]) && !input.peek(Token![::])
                            || input.peek(Token![=])
                        {
                            break;
                        }
                        let value = input.parse()?;
                        bounds.push_value(value);
                        if !input.peek(Token![+]) {
                            break;
                        }
                        let punct = input.parse()?;
                        bounds.push_punct(punct);
                    }
                    bounds
                },
            }))
        }
    }
}

mod kw {
    crate::custom_keyword!(builtin);
    crate::custom_keyword!(raw);
}

pub(crate) struct AllowStruct(bool);
enum Precedence {
    Any,
    Assign,
    Range,
    Or,
    And,
    Compare,
    BitOr,
    BitXor,
    BitAnd,
    Shift,
    Arithmetic,
    Term,
    Cast,
}
impl Precedence {
    fn of(op: &BinOp) -> Self {
        match op {
            BinOp::Add(_) | BinOp::Sub(_) => Precedence::Arithmetic,
            BinOp::Mul(_) | BinOp::Div(_) | BinOp::Rem(_) => Precedence::Term,
            BinOp::And(_) => Precedence::And,
            BinOp::Or(_) => Precedence::Or,
            BinOp::BitXor(_) => Precedence::BitXor,
            BinOp::BitAnd(_) => Precedence::BitAnd,
            BinOp::BitOr(_) => Precedence::BitOr,
            BinOp::Shl(_) | BinOp::Shr(_) => Precedence::Shift,
            BinOp::Eq(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Ne(_) | BinOp::Ge(_) | BinOp::Gt(_) => {
                Precedence::Compare
            },
            BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_) => Precedence::Assign,
        }
    }
}

impl Parse for Expr {
    fn parse(input: ParseStream) -> Result<Self> {
        ambiguous_expr(input, AllowStruct(true))
    }
}
impl Expr {
    pub fn parse_without_eager_brace(input: ParseStream) -> Result<Expr> {
        ambiguous_expr(input, AllowStruct(false))
    }
}
impl Copy for AllowStruct {}
impl Clone for AllowStruct {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Precedence {}
impl Clone for Precedence {
    fn clone(&self) -> Self {
        *self
    }
}
impl PartialEq for Precedence {
    fn eq(&self, other: &Self) -> bool {
        *self as u8 == *other as u8
    }
}
impl PartialOrd for Precedence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let this = *self as u8;
        let other = *other as u8;
        Some(this.cmp(&other))
    }
}

fn parse_expr(input: ParseStream, mut lhs: Expr, allow_struct: AllowStruct, base: Precedence) -> Result<Expr> {
    loop {
        let ahead = input.fork();
        if let Some(op) = match ahead.parse::<BinOp>() {
            Ok(op) if Precedence::of(&op) >= base => Some(op),
            _ => None,
        } {
            input.advance_to(&ahead);
            let precedence = Precedence::of(&op);
            let mut rhs = unary_expr(input, allow_struct)?;
            loop {
                let next = peek_precedence(input);
                if next > precedence || next == precedence && precedence == Precedence::Assign {
                    rhs = parse_expr(input, rhs, allow_struct, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Binary(ExprBinary {
                attrs: Vec::new(),
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            });
        } else if Precedence::Assign >= base
            && input.peek(Token![=])
            && !input.peek(Token![==])
            && !input.peek(Token![=>])
        {
            let eq_token: Token![=] = input.parse()?;
            let mut rhs = unary_expr(input, allow_struct)?;
            loop {
                let next = peek_precedence(input);
                if next >= Precedence::Assign {
                    rhs = parse_expr(input, rhs, allow_struct, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Assign(ExprAssign {
                attrs: Vec::new(),
                left: Box::new(lhs),
                eq_token,
                right: Box::new(rhs),
            });
        } else if Precedence::Range >= base && input.peek(Token![..]) {
            let limits: RangeLimits = input.parse()?;
            let rhs = if matches!(limits, RangeLimits::HalfOpen(_))
                && (input.is_empty()
                    || input.peek(Token![,])
                    || input.peek(Token![;])
                    || input.peek(Token![.]) && !input.peek(Token![..])
                    || !allow_struct.0 && input.peek(tok::Brace))
            {
                None
            } else {
                let mut rhs = unary_expr(input, allow_struct)?;
                loop {
                    let next = peek_precedence(input);
                    if next > Precedence::Range {
                        rhs = parse_expr(input, rhs, allow_struct, next)?;
                    } else {
                        break;
                    }
                }
                Some(rhs)
            };
            lhs = Expr::Range(ExprRange {
                attrs: Vec::new(),
                start: Some(Box::new(lhs)),
                limits,
                end: rhs.map(Box::new),
            });
        } else if Precedence::Cast >= base && input.peek(Token![as]) {
            let as_token: Token![as] = input.parse()?;
            let allow_plus = false;
            let allow_group_generic = false;
            let ty = ambig_ty(input, allow_plus, allow_group_generic)?;
            check_cast(input)?;
            lhs = Expr::Cast(ExprCast {
                attrs: Vec::new(),
                expr: Box::new(lhs),
                as_token,
                ty: Box::new(ty),
            });
        } else {
            break;
        }
    }
    Ok(lhs)
}
fn peek_precedence(input: ParseStream) -> Precedence {
    if let Ok(op) = input.fork().parse() {
        Precedence::of(&op)
    } else if input.peek(Token![=]) && !input.peek(Token![=>]) {
        Precedence::Assign
    } else if input.peek(Token![..]) {
        Precedence::Range
    } else if input.peek(Token![as]) {
        Precedence::Cast
    } else {
        Precedence::Any
    }
}
fn ambiguous_expr(input: ParseStream, #[cfg(feature = "full")] allow_struct: AllowStruct) -> Result<Expr> {
    let lhs = unary_expr(input, allow_struct)?;
    parse_expr(input, lhs, allow_struct, Precedence::Any)
}
fn expr_attrs(input: ParseStream) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();
    loop {
        if input.peek(tok::Group) {
            let ahead = input.fork();
            let group = crate::group::parse_group(&ahead)?;
            if !group.content.peek(Token![#]) || group.content.peek2(Token![!]) {
                break;
            }
            let attr = group.content.call(single_parse_outer)?;
            if !group.content.is_empty() {
                break;
            }
            attrs.push(attr);
        } else if input.peek(Token![#]) {
            attrs.push(input.call(single_parse_outer)?);
        } else {
            break;
        }
    }
    Ok(attrs)
}
fn unary_expr(input: ParseStream, allow_struct: AllowStruct) -> Result<Expr> {
    let begin = input.fork();
    let attrs = input.call(expr_attrs)?;
    if input.peek(Token![&]) {
        let and_token: Token![&] = input.parse()?;
        let raw: Option<kw::raw> = if input.peek(kw::raw) && (input.peek2(Token![mut]) || input.peek2(Token![const])) {
            Some(input.parse()?)
        } else {
            None
        };
        let mutability: Option<Token![mut]> = input.parse()?;
        if raw.is_some() && mutability.is_none() {
            input.parse::<Token![const]>()?;
        }
        let expr = Box::new(unary_expr(input, allow_struct)?);
        if raw.is_some() {
            Ok(Expr::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(Expr::Reference(ExprReference {
                attrs,
                and_token,
                mutability,
                expr,
            }))
        }
    } else if input.peek(Token![*]) || input.peek(Token![!]) || input.peek(Token![-]) {
        expr_unary(input, attrs, allow_struct).map(Expr::Unary)
    } else {
        trailer_expr(begin, attrs, input, allow_struct)
    }
}
fn trailer_expr(
    begin: ParseBuffer,
    mut attrs: Vec<Attribute>,
    input: ParseStream,
    allow_struct: AllowStruct,
) -> Result<Expr> {
    let atom = atom_expr(input, allow_struct)?;
    let mut e = trailer_helper(input, atom)?;
    if let Expr::Verbatim(tokens) = &mut e {
        *tokens = verbatim_between(&begin, input);
    } else {
        let inner_attrs = e.replace_attrs(Vec::new());
        attrs.extend(inner_attrs);
        e.replace_attrs(attrs);
    }
    Ok(e)
}
fn trailer_helper(input: ParseStream, mut e: Expr) -> Result<Expr> {
    loop {
        if input.peek(tok::Paren) {
            let content;
            e = Expr::Call(ExprCall {
                attrs: Vec::new(),
                func: Box::new(e),
                paren_token: parenthesized!(content in input),
                args: content.parse_terminated(Expr::parse, Token![,])?,
            });
        } else if input.peek(Token![.])
            && !input.peek(Token![..])
            && match e {
                Expr::Range(_) => false,
                _ => true,
            }
        {
            let mut dot_token: Token![.] = input.parse()?;
            let float_token: Option<LitFloat> = input.parse()?;
            if let Some(float_token) = float_token {
                if multi_index(&mut e, &mut dot_token, float_token)? {
                    continue;
                }
            }
            let await_token: Option<Token![await]> = input.parse()?;
            if let Some(await_token) = await_token {
                e = Expr::Await(ExprAwait {
                    attrs: Vec::new(),
                    base: Box::new(e),
                    dot_token,
                    await_token,
                });
                continue;
            }
            let member: Member = input.parse()?;
            let turbofish = if member.is_named() && input.peek(Token![::]) {
                Some(AngleBracketedGenericArguments::parse_turbofish(input)?)
            } else {
                None
            };
            if turbofish.is_some() || input.peek(tok::Paren) {
                if let Member::Named(method) = member {
                    let content;
                    e = Expr::MethodCall(ExprMethodCall {
                        attrs: Vec::new(),
                        receiver: Box::new(e),
                        dot_token,
                        method,
                        turbofish,
                        paren_token: parenthesized!(content in input),
                        args: content.parse_terminated(Expr::parse, Token![,])?,
                    });
                    continue;
                }
            }
            e = Expr::Field(ExprField {
                attrs: Vec::new(),
                base: Box::new(e),
                dot_token,
                member,
            });
        } else if input.peek(tok::Bracket) {
            let content;
            e = Expr::Index(ExprIndex {
                attrs: Vec::new(),
                expr: Box::new(e),
                bracket_token: bracketed!(content in input),
                index: content.parse()?,
            });
        } else if input.peek(Token![?]) {
            e = Expr::Try(ExprTry {
                attrs: Vec::new(),
                expr: Box::new(e),
                question_token: input.parse()?,
            });
        } else {
            break;
        }
    }
    Ok(e)
}
fn atom_expr(input: ParseStream, allow_struct: AllowStruct) -> Result<Expr> {
    if input.peek(tok::Group) && !input.peek2(Token![::]) && !input.peek2(Token![!]) && !input.peek2(tok::Brace) {
        input.call(expr_group).map(Expr::Group)
    } else if input.peek(Lit) {
        input.parse().map(Expr::Lit)
    } else if input.peek(Token![async])
        && (input.peek2(tok::Brace) || input.peek2(Token![move]) && input.peek3(tok::Brace))
    {
        input.parse().map(Expr::Async)
    } else if input.peek(Token![try]) && input.peek2(tok::Brace) {
        input.parse().map(Expr::TryBlock)
    } else if input.peek(Token![|])
        || input.peek(Token![move])
        || input.peek(Token![for]) && input.peek2(Token![<]) && (input.peek3(Lifetime) || input.peek3(Token![>]))
        || input.peek(Token![const]) && !input.peek2(tok::Brace)
        || input.peek(Token![static])
        || input.peek(Token![async]) && (input.peek2(Token![|]) || input.peek2(Token![move]))
    {
        expr_closure(input, allow_struct).map(Expr::Closure)
    } else if input.peek(kw::builtin) && input.peek2(Token![#]) {
        expr_builtin(input)
    } else if input.peek(Ident)
        || input.peek(Token![::])
        || input.peek(Token![<])
        || input.peek(Token![self])
        || input.peek(Token![Self])
        || input.peek(Token![super])
        || input.peek(Token![crate])
        || input.peek(Token![try]) && (input.peek2(Token![!]) || input.peek2(Token![::]))
    {
        path_or_macro_or_struct(input, allow_struct)
    } else if input.peek(tok::Paren) {
        paren_or_tuple(input)
    } else if input.peek(Token![break]) {
        expr_break(input, allow_struct).map(Expr::Break)
    } else if input.peek(Token![continue]) {
        input.parse().map(Expr::Continue)
    } else if input.peek(Token![return]) {
        expr_ret(input, allow_struct).map(Expr::Return)
    } else if input.peek(tok::Bracket) {
        array_or_repeat(input)
    } else if input.peek(Token![let]) {
        input.parse().map(Expr::Let)
    } else if input.peek(Token![if]) {
        input.parse().map(Expr::If)
    } else if input.peek(Token![while]) {
        input.parse().map(Expr::While)
    } else if input.peek(Token![for]) {
        input.parse().map(Expr::ForLoop)
    } else if input.peek(Token![loop]) {
        input.parse().map(Expr::Loop)
    } else if input.peek(Token![match]) {
        input.parse().map(Expr::Match)
    } else if input.peek(Token![yield]) {
        input.parse().map(Expr::Yield)
    } else if input.peek(Token![unsafe]) {
        input.parse().map(Expr::Unsafe)
    } else if input.peek(Token![const]) {
        input.parse().map(Expr::Const)
    } else if input.peek(tok::Brace) {
        input.parse().map(Expr::Block)
    } else if input.peek(Token![..]) {
        expr_range(input, allow_struct).map(Expr::Range)
    } else if input.peek(Token![_]) {
        input.parse().map(Expr::Infer)
    } else if input.peek(Lifetime) {
        let the_label: Label = input.parse()?;
        let mut expr = if input.peek(Token![while]) {
            Expr::While(input.parse()?)
        } else if input.peek(Token![for]) {
            Expr::ForLoop(input.parse()?)
        } else if input.peek(Token![loop]) {
            Expr::Loop(input.parse()?)
        } else if input.peek(tok::Brace) {
            Expr::Block(input.parse()?)
        } else {
            return Err(input.error("expected loop or block expression"));
        };
        match &mut expr {
            Expr::While(ExprWhile { label, .. })
            | Expr::ForLoop(ExprForLoop { label, .. })
            | Expr::Loop(ExprLoop { label, .. })
            | Expr::Block(ExprBlock { label, .. }) => *label = Some(the_label),
            _ => unreachable!(),
        }
        Ok(expr)
    } else {
        Err(input.error("expected expression"))
    }
}
fn expr_builtin(input: ParseStream) -> Result<Expr> {
    let begin = input.fork();
    input.parse::<kw::builtin>()?;
    input.parse::<Token![#]>()?;
    input.parse::<Ident>()?;
    let args;
    parenthesized!(args in input);
    args.parse::<TokenStream>()?;
    Ok(Expr::Verbatim(verbatim_between(&begin, input)))
}
fn path_or_macro_or_struct(input: ParseStream, #[cfg(feature = "full")] allow_struct: AllowStruct) -> Result<Expr> {
    let (qself, path) = qpath(input, true)?;
    if qself.is_none() && input.peek(Token![!]) && !input.peek(Token![!=]) && path.is_mod_style() {
        let bang_token: Token![!] = input.parse()?;
        let (delimiter, tokens) = mac_parse_delimiter(input)?;
        return Ok(Expr::Macro(ExprMacro {
            attrs: Vec::new(),
            mac: Macro {
                path,
                bang: bang_token,
                delim: delimiter,
                toks: tokens,
            },
        }));
    }
    if allow_struct.0 && input.peek(tok::Brace) {
        return expr_struct_helper(input, qself, path).map(Expr::Struct);
    }
    Ok(Expr::Path(ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    }))
}
impl Parse for ExprMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprMacro {
            attrs: Vec::new(),
            mac: input.parse()?,
        })
    }
}
fn paren_or_tuple(input: ParseStream) -> Result<Expr> {
    let content;
    let paren_token = parenthesized!(content in input);
    if content.is_empty() {
        return Ok(Expr::Tuple(ExprTuple {
            attrs: Vec::new(),
            paren_token,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = content.parse()?;
    if content.is_empty() {
        return Ok(Expr::Paren(ExprParen {
            attrs: Vec::new(),
            paren_token,
            expr: Box::new(first),
        }));
    }
    let mut elems = Punctuated::new();
    elems.push_value(first);
    while !content.is_empty() {
        let punct = content.parse()?;
        elems.push_punct(punct);
        if content.is_empty() {
            break;
        }
        let value = content.parse()?;
        elems.push_value(value);
    }
    Ok(Expr::Tuple(ExprTuple {
        attrs: Vec::new(),
        paren_token,
        elems,
    }))
}
fn array_or_repeat(input: ParseStream) -> Result<Expr> {
    let content;
    let bracket_token = bracketed!(content in input);
    if content.is_empty() {
        return Ok(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = content.parse()?;
    if content.is_empty() || content.peek(Token![,]) {
        let mut elems = Punctuated::new();
        elems.push_value(first);
        while !content.is_empty() {
            let punct = content.parse()?;
            elems.push_punct(punct);
            if content.is_empty() {
                break;
            }
            let value = content.parse()?;
            elems.push_value(value);
        }
        Ok(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token,
            elems,
        }))
    } else if content.peek(Token![;]) {
        let semi_token: Token![;] = content.parse()?;
        let len: Expr = content.parse()?;
        Ok(Expr::Repeat(ExprRepeat {
            attrs: Vec::new(),
            bracket_token,
            expr: Box::new(first),
            semi_token,
            len: Box::new(len),
        }))
    } else {
        Err(content.error("expected `,` or `;`"))
    }
}
impl Parse for ExprArray {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        let bracket_token = bracketed!(content in input);
        let mut elems = Punctuated::new();
        while !content.is_empty() {
            let first: Expr = content.parse()?;
            elems.push_value(first);
            if content.is_empty() {
                break;
            }
            let punct = content.parse()?;
            elems.push_punct(punct);
        }
        Ok(ExprArray {
            attrs: Vec::new(),
            bracket_token,
            elems,
        })
    }
}
impl Parse for ExprRepeat {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(ExprRepeat {
            bracket_token: bracketed!(content in input),
            attrs: Vec::new(),
            expr: content.parse()?,
            semi_token: content.parse()?,
            len: content.parse()?,
        })
    }
}
pub(crate) fn expr_early(input: ParseStream) -> Result<Expr> {
    let mut attrs = input.call(expr_attrs)?;
    let mut expr = if input.peek(Token![if]) {
        Expr::If(input.parse()?)
    } else if input.peek(Token![while]) {
        Expr::While(input.parse()?)
    } else if input.peek(Token![for]) && !(input.peek2(Token![<]) && (input.peek3(Lifetime) || input.peek3(Token![>])))
    {
        Expr::ForLoop(input.parse()?)
    } else if input.peek(Token![loop]) {
        Expr::Loop(input.parse()?)
    } else if input.peek(Token![match]) {
        Expr::Match(input.parse()?)
    } else if input.peek(Token![try]) && input.peek2(tok::Brace) {
        Expr::TryBlock(input.parse()?)
    } else if input.peek(Token![unsafe]) {
        Expr::Unsafe(input.parse()?)
    } else if input.peek(Token![const]) && input.peek2(tok::Brace) {
        Expr::Const(input.parse()?)
    } else if input.peek(tok::Brace) {
        Expr::Block(input.parse()?)
    } else {
        let allow_struct = AllowStruct(true);
        let mut expr = unary_expr(input, allow_struct)?;
        attrs.extend(expr.replace_attrs(Vec::new()));
        expr.replace_attrs(attrs);
        return parse_expr(input, expr, allow_struct, Precedence::Any);
    };
    if input.peek(Token![.]) && !input.peek(Token![..]) || input.peek(Token![?]) {
        expr = trailer_helper(input, expr)?;
        attrs.extend(expr.replace_attrs(Vec::new()));
        expr.replace_attrs(attrs);
        let allow_struct = AllowStruct(true);
        return parse_expr(input, expr, allow_struct, Precedence::Any);
    }
    attrs.extend(expr.replace_attrs(Vec::new()));
    expr.replace_attrs(attrs);
    Ok(expr)
}
impl Parse for ExprLit {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprLit {
            attrs: Vec::new(),
            lit: input.parse()?,
        })
    }
}
fn expr_group(input: ParseStream) -> Result<ExprGroup> {
    let group = crate::group::parse_group(input)?;
    Ok(ExprGroup {
        attrs: Vec::new(),
        group_token: group.token,
        expr: group.content.parse()?,
    })
}
impl Parse for ExprParen {
    fn parse(input: ParseStream) -> Result<Self> {
        expr_paren(input)
    }
}
fn expr_paren(input: ParseStream) -> Result<ExprParen> {
    let content;
    Ok(ExprParen {
        attrs: Vec::new(),
        paren_token: parenthesized!(content in input),
        expr: content.parse()?,
    })
}
impl Parse for ExprLet {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprLet {
            attrs: Vec::new(),
            let_token: input.parse()?,
            pat: Box::new(Pat::parse_multi_with_leading_vert(input)?),
            eq_token: input.parse()?,
            expr: Box::new({
                let allow_struct = AllowStruct(false);
                let lhs = unary_expr(input, allow_struct)?;
                parse_expr(input, lhs, allow_struct, Precedence::Compare)?
            }),
        })
    }
}
impl Parse for ExprIf {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        Ok(ExprIf {
            attrs,
            if_token: input.parse()?,
            cond: Box::new(input.call(Expr::parse_without_eager_brace)?),
            then_branch: input.parse()?,
            else_branch: {
                if input.peek(Token![else]) {
                    Some(input.call(else_block)?)
                } else {
                    None
                }
            },
        })
    }
}
fn else_block(input: ParseStream) -> Result<(Token![else], Box<Expr>)> {
    let else_token: Token![else] = input.parse()?;
    let lookahead = input.lookahead1();
    let else_branch = if input.peek(Token![if]) {
        input.parse().map(Expr::If)?
    } else if input.peek(tok::Brace) {
        Expr::Block(ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: input.parse()?,
        })
    } else {
        return Err(lookahead.error());
    };
    Ok((else_token, Box::new(else_branch)))
}
impl Parse for ExprInfer {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprInfer {
            attrs: input.call(Attribute::parse_outer)?,
            underscore_token: input.parse()?,
        })
    }
}
impl Parse for ExprForLoop {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let for_token: Token![for] = input.parse()?;
        let pat = Pat::parse_multi_with_leading_vert(input)?;
        let in_token: Token![in] = input.parse()?;
        let expr: Expr = input.call(Expr::parse_without_eager_brace)?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprForLoop {
            attrs,
            label,
            for_token,
            pat: Box::new(pat),
            in_token,
            expr: Box::new(expr),
            body: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
impl Parse for ExprLoop {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let loop_token: Token![loop] = input.parse()?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprLoop {
            attrs,
            label,
            loop_token,
            body: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
impl Parse for ExprMatch {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let match_token: Token![match] = input.parse()?;
        let expr = Expr::parse_without_eager_brace(input)?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let mut arms = Vec::new();
        while !content.is_empty() {
            arms.push(content.call(Arm::parse)?);
        }
        Ok(ExprMatch {
            attrs,
            match_token,
            expr: Box::new(expr),
            brace_token,
            arms,
        })
    }
}
macro_rules! impl_by_parsing_expr {
        (
            $(
                $expr_type:ty, $variant:ident, $msg:expr,
            )*
        ) => {
            $(
                impl Parse for $expr_type {
                    fn parse(input: ParseStream) -> Result<Self> {
                        let mut expr: Expr = input.parse()?;
                        loop {
                            match expr {
                                Expr::$variant(inner) => return Ok(inner),
                                Expr::Group(next) => expr = *next.expr,
                                _ => return Err(Error::new_spanned(expr, $msg)),
                            }
                        }
                    }
                }
            )*
        };
    }
impl_by_parsing_expr! {
    ExprAssign, Assign, "expected assignment expression",
    ExprAwait, Await, "expected await expression",
    ExprBinary, Binary, "expected binary operation",
    ExprCall, Call, "expected function call expression",
    ExprCast, Cast, "expected cast expression",
    ExprField, Field, "expected struct field access",
    ExprIndex, Index, "expected indexing expression",
    ExprMethodCall, MethodCall, "expected method call expression",
    ExprRange, Range, "expected range expression",
    ExprTry, Try, "expected try expression",
    ExprTuple, Tuple, "expected tuple expression",
}
impl Parse for ExprUnary {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = Vec::new();
        let allow_struct = AllowStruct(true);
        expr_unary(input, attrs, allow_struct)
    }
}
fn expr_unary(input: ParseStream, attrs: Vec<Attribute>, allow_struct: AllowStruct) -> Result<ExprUnary> {
    Ok(ExprUnary {
        attrs,
        op: input.parse()?,
        expr: Box::new(unary_expr(input, allow_struct)?),
    })
}
impl Parse for ExprClosure {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_closure(input, allow_struct)
    }
}
impl Parse for ExprReference {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        Ok(ExprReference {
            attrs: Vec::new(),
            and_token: input.parse()?,
            mutability: input.parse()?,
            expr: Box::new(unary_expr(input, allow_struct)?),
        })
    }
}
impl Parse for ExprBreak {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_break(input, allow_struct)
    }
}
impl Parse for ExprReturn {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_ret(input, allow_struct)
    }
}
impl Parse for ExprTryBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprTryBlock {
            attrs: Vec::new(),
            try_token: input.parse()?,
            block: input.parse()?,
        })
    }
}
impl Parse for ExprYield {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprYield {
            attrs: Vec::new(),
            yield_token: input.parse()?,
            expr: {
                if !input.is_empty() && !input.peek(Token![,]) && !input.peek(Token![;]) {
                    Some(input.parse()?)
                } else {
                    None
                }
            },
        })
    }
}
fn expr_closure(input: ParseStream, allow_struct: AllowStruct) -> Result<ExprClosure> {
    let lifetimes: Option<BoundLifetimes> = input.parse()?;
    let constness: Option<Token![const]> = input.parse()?;
    let movability: Option<Token![static]> = input.parse()?;
    let asyncness: Option<Token![async]> = input.parse()?;
    let capture: Option<Token![move]> = input.parse()?;
    let or1_token: Token![|] = input.parse()?;
    let mut inputs = Punctuated::new();
    loop {
        if input.peek(Token![|]) {
            break;
        }
        let value = closure_arg(input)?;
        inputs.push_value(value);
        if input.peek(Token![|]) {
            break;
        }
        let punct: Token![,] = input.parse()?;
        inputs.push_punct(punct);
    }
    let or2_token: Token![|] = input.parse()?;
    let (output, body) = if input.peek(Token![->]) {
        let arrow_token: Token![->] = input.parse()?;
        let ty: Type = input.parse()?;
        let body: Block = input.parse()?;
        let output = ReturnType::Type(arrow_token, Box::new(ty));
        let block = Expr::Block(ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: body,
        });
        (output, block)
    } else {
        let body = ambiguous_expr(input, allow_struct)?;
        (ReturnType::Default, body)
    };
    Ok(ExprClosure {
        attrs: Vec::new(),
        lifetimes,
        constness,
        movability,
        asyncness,
        capture,
        or1_token,
        inputs,
        or2_token,
        output,
        body: Box::new(body),
    })
}
impl Parse for ExprAsync {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprAsync {
            attrs: Vec::new(),
            async_token: input.parse()?,
            capture: input.parse()?,
            block: input.parse()?,
        })
    }
}
fn closure_arg(input: ParseStream) -> Result<Pat> {
    let attrs = input.call(Attribute::parse_outer)?;
    let mut pat = Pat::parse_single(input)?;
    if input.peek(Token![:]) {
        Ok(Pat::Type(PatType {
            attrs,
            pat: Box::new(pat),
            colon: input.parse()?,
            ty: input.parse()?,
        }))
    } else {
        match &mut pat {
            Pat::Const(pat) => pat.attrs = attrs,
            Pat::Ident(pat) => pat.attrs = attrs,
            Pat::Lit(pat) => pat.attrs = attrs,
            Pat::Macro(pat) => pat.attrs = attrs,
            Pat::Or(pat) => pat.attrs = attrs,
            Pat::Paren(pat) => pat.attrs = attrs,
            Pat::Path(pat) => pat.attrs = attrs,
            Pat::Range(pat) => pat.attrs = attrs,
            Pat::Reference(pat) => pat.attrs = attrs,
            Pat::Rest(pat) => pat.attrs = attrs,
            Pat::Slice(pat) => pat.attrs = attrs,
            Pat::Struct(pat) => pat.attrs = attrs,
            Pat::Tuple(pat) => pat.attrs = attrs,
            Pat::TupleStruct(pat) => pat.attrs = attrs,
            Pat::Type(_) => unreachable!(),
            Pat::Verbatim(_) => {},
            Pat::Wild(pat) => pat.attrs = attrs,
        }
        Ok(pat)
    }
}
impl Parse for ExprWhile {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let while_token: Token![while] = input.parse()?;
        let cond = Expr::parse_without_eager_brace(input)?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprWhile {
            attrs,
            label,
            while_token,
            cond: Box::new(cond),
            body: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
impl Parse for ExprConst {
    fn parse(input: ParseStream) -> Result<Self> {
        let const_token: Token![const] = input.parse()?;
        let content;
        let brace_token = braced!(content in input);
        let inner_attrs = content.call(Attribute::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprConst {
            attrs: inner_attrs,
            const_token,
            block: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
impl Parse for Label {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Label {
            name: input.parse()?,
            colon_token: input.parse()?,
        })
    }
}
impl Parse for Option<Label> {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Lifetime) {
            input.parse().map(Some)
        } else {
            Ok(None)
        }
    }
}
impl Parse for ExprContinue {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ExprContinue {
            attrs: Vec::new(),
            continue_token: input.parse()?,
            label: input.parse()?,
        })
    }
}
fn expr_break(input: ParseStream, allow_struct: AllowStruct) -> Result<ExprBreak> {
    Ok(ExprBreak {
        attrs: Vec::new(),
        break_token: input.parse()?,
        label: input.parse()?,
        expr: {
            if input.is_empty()
                || input.peek(Token![,])
                || input.peek(Token![;])
                || !allow_struct.0 && input.peek(tok::Brace)
            {
                None
            } else {
                let expr = ambiguous_expr(input, allow_struct)?;
                Some(Box::new(expr))
            }
        },
    })
}
fn expr_ret(input: ParseStream, allow_struct: AllowStruct) -> Result<ExprReturn> {
    Ok(ExprReturn {
        attrs: Vec::new(),
        return_token: input.parse()?,
        expr: {
            if input.is_empty() || input.peek(Token![,]) || input.peek(Token![;]) {
                None
            } else {
                let expr = ambiguous_expr(input, allow_struct)?;
                Some(Box::new(expr))
            }
        },
    })
}
impl Parse for FieldValue {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let member: Member = input.parse()?;
        let (colon_token, value) = if input.peek(Token![:]) || !member.is_named() {
            let colon_token: Token![:] = input.parse()?;
            let value: Expr = input.parse()?;
            (Some(colon_token), value)
        } else if let Member::Named(ident) = &member {
            let value = Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: Path::from(ident.clone()),
            });
            (None, value)
        } else {
            unreachable!()
        };
        Ok(FieldValue {
            attrs,
            member,
            colon_token,
            expr: value,
        })
    }
}
impl Parse for ExprStruct {
    fn parse(input: ParseStream) -> Result<Self> {
        let (qself, path) = qpath(input, true)?;
        expr_struct_helper(input, qself, path)
    }
}
fn expr_struct_helper(input: ParseStream, qself: Option<QSelf>, path: Path) -> Result<ExprStruct> {
    let content;
    let brace_token = braced!(content in input);
    let mut fields = Punctuated::new();
    while !content.is_empty() {
        if content.peek(Token![..]) {
            return Ok(ExprStruct {
                attrs: Vec::new(),
                qself,
                path,
                brace_token,
                fields,
                dot2_token: Some(content.parse()?),
                rest: if content.is_empty() {
                    None
                } else {
                    Some(Box::new(content.parse()?))
                },
            });
        }
        fields.push(content.parse()?);
        if content.is_empty() {
            break;
        }
        let punct: Token![,] = content.parse()?;
        fields.push_punct(punct);
    }
    Ok(ExprStruct {
        attrs: Vec::new(),
        qself,
        path,
        brace_token,
        fields,
        dot2_token: None,
        rest: None,
    })
}
impl Parse for ExprUnsafe {
    fn parse(input: ParseStream) -> Result<Self> {
        let unsafe_token: Token![unsafe] = input.parse()?;
        let content;
        let brace_token = braced!(content in input);
        let inner_attrs = content.call(Attribute::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprUnsafe {
            attrs: inner_attrs,
            unsafe_token,
            block: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
impl Parse for ExprBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(ExprBlock {
            attrs,
            label,
            block: Block {
                brace: brace_token,
                stmts,
            },
        })
    }
}
fn expr_range(input: ParseStream, allow_struct: AllowStruct) -> Result<ExprRange> {
    let limits: RangeLimits = input.parse()?;
    let end = if matches!(limits, RangeLimits::HalfOpen(_))
        && (input.is_empty()
            || input.peek(Token![,])
            || input.peek(Token![;])
            || input.peek(Token![.]) && !input.peek(Token![..])
            || !allow_struct.0 && input.peek(tok::Brace))
    {
        None
    } else {
        let to = ambiguous_expr(input, allow_struct)?;
        Some(Box::new(to))
    };
    Ok(ExprRange {
        attrs: Vec::new(),
        start: None,
        limits,
        end,
    })
}
impl Parse for RangeLimits {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        let dot_dot = lookahead.peek(Token![..]);
        let dot_dot_eq = dot_dot && lookahead.peek(Token![..=]);
        let dot_dot_dot = dot_dot && input.peek(Token![...]);
        if dot_dot_eq {
            input.parse().map(RangeLimits::Closed)
        } else if dot_dot && !dot_dot_dot {
            input.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(lookahead.error())
        }
    }
}
impl RangeLimits {
    pub(crate) fn parse_obsolete(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        let dot_dot = lookahead.peek(Token![..]);
        let dot_dot_eq = dot_dot && lookahead.peek(Token![..=]);
        let dot_dot_dot = dot_dot && input.peek(Token![...]);
        if dot_dot_eq {
            input.parse().map(RangeLimits::Closed)
        } else if dot_dot_dot {
            let dot3: Token![...] = input.parse()?;
            Ok(RangeLimits::Closed(Token![..=](dot3.spans)))
        } else if dot_dot {
            input.parse().map(RangeLimits::HalfOpen)
        } else {
            Err(lookahead.error())
        }
    }
}
impl Parse for ExprPath {
    fn parse(input: ParseStream) -> Result<Self> {
        #[cfg(not(feature = "full"))]
        let attrs = Vec::new();
        let attrs = input.call(Attribute::parse_outer)?;
        let (qself, path) = qpath(input, true)?;
        Ok(ExprPath { attrs, qself, path })
    }
}
impl Parse for Member {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Ident) {
            input.parse().map(Member::Named)
        } else if input.peek(LitInt) {
            input.parse().map(Member::Unnamed)
        } else {
            Err(input.error("expected identifier or integer"))
        }
    }
}
impl Parse for Arm {
    fn parse(input: ParseStream) -> Result<Arm> {
        let requires_comma;
        Ok(Arm {
            attrs: input.call(Attribute::parse_outer)?,
            pat: Pat::parse_multi_with_leading_vert(input)?,
            guard: {
                if input.peek(Token![if]) {
                    let if_token: Token![if] = input.parse()?;
                    let guard: Expr = input.parse()?;
                    Some((if_token, Box::new(guard)))
                } else {
                    None
                }
            },
            fat_arrow_token: input.parse()?,
            body: {
                let body = input.call(expr_early)?;
                requires_comma = requires_terminator(&body);
                Box::new(body)
            },
            comma: {
                if requires_comma && !input.is_empty() {
                    Some(input.parse()?)
                } else {
                    input.parse()?
                }
            },
        })
    }
}
impl Parse for Index {
    fn parse(input: ParseStream) -> Result<Self> {
        let lit: LitInt = input.parse()?;
        if lit.suffix().is_empty() {
            Ok(Index {
                index: lit.base10_digits().parse().map_err(|err| Err::new(lit.span(), err))?,
                span: lit.span(),
            })
        } else {
            Err(Err::new(lit.span(), "expected unsuffixed integer"))
        }
    }
}
fn multi_index(e: &mut Expr, dot_token: &mut Token![.], float: LitFloat) -> Result<bool> {
    let float_token = float.token();
    let float_span = float_token.span();
    let mut float_repr = float_token.to_string();
    let trailing_dot = float_repr.ends_with('.');
    if trailing_dot {
        float_repr.truncate(float_repr.len() - 1);
    }
    let mut offset = 0;
    for part in float_repr.split('.') {
        let mut index: Index = crate::parse_str(part).map_err(|err| Err::new(float_span, err))?;
        let part_end = offset + part.len();
        index.span = float_token.subspan(offset..part_end).unwrap_or(float_span);
        let base = mem::replace(e, Expr::DUMMY);
        *e = Expr::Field(ExprField {
            attrs: Vec::new(),
            base: Box::new(base),
            dot_token: Token![.](dot_token.span),
            member: Member::Unnamed(index),
        });
        let dot_span = float_token.subspan(part_end..part_end + 1).unwrap_or(float_span);
        *dot_token = Token![.](dot_span);
        offset = part_end + 1;
    }
    Ok(!trailing_dot)
}
impl Member {
    fn is_named(&self) -> bool {
        match self {
            Member::Named(_) => true,
            Member::Unnamed(_) => false,
        }
    }
}
fn check_cast(input: ParseStream) -> Result<()> {
    let kind = if input.peek(Token![.]) && !input.peek(Token![..]) {
        if input.peek2(Token![await]) {
            "`.await`"
        } else if input.peek2(Ident) && (input.peek3(tok::Paren) || input.peek3(Token![::])) {
            "a method call"
        } else {
            "a field access"
        }
    } else if input.peek(Token![?]) {
        "`?`"
    } else if input.peek(tok::Bracket) {
        "indexing"
    } else if input.peek(tok::Paren) {
        "a function call"
    } else {
        return Ok(());
    };
    let msg = format!("casts cannot be followed by {}", kind);
    Err(input.error(msg))
}

impl Parse for Item {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let attrs = input.call(Attribute::parse_outer)?;
        parse_rest_of_item(begin, attrs, input)
    }
}
pub(crate) fn parse_rest_of_item(begin: ParseBuffer, mut attrs: Vec<Attribute>, input: ParseStream) -> Result<Item> {
    let ahead = input.fork();
    let vis: Visibility = ahead.parse()?;
    let lookahead = ahead.lookahead1();
    let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
        let vis: Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            parse_rest_of_fn(input, Vec::new(), vis, sig).map(Item::Fn)
        }
    } else if lookahead.peek(Token![extern]) {
        ahead.parse::<Token![extern]>()?;
        let lookahead = ahead.lookahead1();
        if lookahead.peek(Token![crate]) {
            input.parse().map(Item::ExternCrate)
        } else if lookahead.peek(tok::Brace) {
            input.parse().map(Item::ForeignMod)
        } else if lookahead.peek(LitStr) {
            ahead.parse::<LitStr>()?;
            let lookahead = ahead.lookahead1();
            if lookahead.peek(tok::Brace) {
                input.parse().map(Item::ForeignMod)
            } else {
                Err(lookahead.error())
            }
        } else {
            Err(lookahead.error())
        }
    } else if lookahead.peek(Token![use]) {
        let allow_crate_root_in_path = true;
        match parse_item_use(input, allow_crate_root_in_path)? {
            Some(item_use) => Ok(Item::Use(item_use)),
            None => Ok(Item::Verbatim(verbatim_between(&begin, input))),
        }
    } else if lookahead.peek(Token![static]) {
        let vis = input.parse()?;
        let static_token = input.parse()?;
        let mutability = input.parse()?;
        let ident = input.parse()?;
        if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            input.parse::<Expr>()?;
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            let colon_token = input.parse()?;
            let ty = input.parse()?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                Ok(Item::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(Item::Static(ItemStatic {
                    attrs: Vec::new(),
                    vis,
                    static_token,
                    mutability,
                    ident,
                    colon_token,
                    ty,
                    eq_token: input.parse()?,
                    expr: input.parse()?,
                    semi_token: input.parse()?,
                }))
            }
        }
    } else if lookahead.peek(Token![const]) {
        let vis = input.parse()?;
        let const_token: Token![const] = input.parse()?;
        let lookahead = input.lookahead1();
        let ident = if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
            input.call(Ident::parse_any)?
        } else {
            return Err(lookahead.error());
        };
        let colon_token = input.parse()?;
        let ty = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(Item::Const(ItemConst {
                attrs: Vec::new(),
                vis,
                const_token,
                ident,
                generics: Generics::default(),
                colon_token,
                ty,
                eq_token: input.parse()?,
                expr: input.parse()?,
                semi_token: input.parse()?,
            }))
        }
    } else if lookahead.peek(Token![unsafe]) {
        ahead.parse::<Token![unsafe]>()?;
        let lookahead = ahead.lookahead1();
        if lookahead.peek(Token![trait]) || lookahead.peek(Token![auto]) && ahead.peek2(Token![trait]) {
            input.parse().map(Item::Trait)
        } else if lookahead.peek(Token![impl]) {
            let allow_verbatim_impl = true;
            if let Some(item) = parse_impl(input, allow_verbatim_impl)? {
                Ok(Item::Impl(item))
            } else {
                Ok(Item::Verbatim(verbatim_between(&begin, input)))
            }
        } else if lookahead.peek(Token![extern]) {
            input.parse().map(Item::ForeignMod)
        } else if lookahead.peek(Token![mod]) {
            input.parse().map(Item::Mod)
        } else {
            Err(lookahead.error())
        }
    } else if lookahead.peek(Token![mod]) {
        input.parse().map(Item::Mod)
    } else if lookahead.peek(Token![type]) {
        parse_item_type(begin, input)
    } else if lookahead.peek(Token![struct]) {
        input.parse().map(Item::Struct)
    } else if lookahead.peek(Token![enum]) {
        input.parse().map(Item::Enum)
    } else if lookahead.peek(Token![union]) && ahead.peek2(Ident) {
        input.parse().map(Item::Union)
    } else if lookahead.peek(Token![trait]) {
        input.call(parse_trait_or_trait_alias)
    } else if lookahead.peek(Token![auto]) && ahead.peek2(Token![trait]) {
        input.parse().map(Item::Trait)
    } else if lookahead.peek(Token![impl]) || lookahead.peek(Token![default]) && !ahead.peek2(Token![!]) {
        let allow_verbatim_impl = true;
        if let Some(item) = parse_impl(input, allow_verbatim_impl)? {
            Ok(Item::Impl(item))
        } else {
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        }
    } else if lookahead.peek(Token![macro]) {
        input.advance_to(&ahead);
        parse_macro2(begin, vis, input)
    } else if vis.is_inherited()
        && (lookahead.peek(Ident)
            || lookahead.peek(Token![self])
            || lookahead.peek(Token![super])
            || lookahead.peek(Token![crate])
            || lookahead.peek(Token![::]))
    {
        input.parse().map(Item::Macro)
    } else {
        Err(lookahead.error())
    }?;
    attrs.extend(item.replace_attrs(Vec::new()));
    item.replace_attrs(attrs);
    Ok(item)
}
struct FlexibleItemType {
    vis: Visibility,
    defaultness: Option<Token![default]>,
    type_token: Token![type],
    ident: Ident,
    generics: Generics,
    colon_token: Option<Token![:]>,
    bounds: Punctuated<TypeParamBound, Token![+]>,
    ty: Option<(Token![=], Type)>,
    semi_token: Token![;],
}
enum TypeDefaultness {
    Optional,
    Disallowed,
}
enum WhereClauseLocation {
    BeforeEq,
    AfterEq,
    Both,
}
impl FlexibleItemType {
    fn parse(
        input: ParseStream,
        allow_defaultness: TypeDefaultness,
        where_clause_location: WhereClauseLocation,
    ) -> Result<Self> {
        let vis: Visibility = input.parse()?;
        let defaultness: Option<Token![default]> = match allow_defaultness {
            TypeDefaultness::Optional => input.parse()?,
            TypeDefaultness::Disallowed => None,
        };
        let type_token: Token![type] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        let (colon_token, bounds) = Self::parse_optional_bounds(input)?;
        match where_clause_location {
            WhereClauseLocation::BeforeEq | WhereClauseLocation::Both => {
                generics.clause = input.parse()?;
            },
            WhereClauseLocation::AfterEq => {},
        }
        let ty = Self::parse_optional_definition(input)?;
        match where_clause_location {
            WhereClauseLocation::AfterEq | WhereClauseLocation::Both if generics.clause.is_none() => {
                generics.clause = input.parse()?;
            },
            _ => {},
        }
        let semi_token: Token![;] = input.parse()?;
        Ok(FlexibleItemType {
            vis,
            defaultness,
            type_token,
            ident,
            generics,
            colon_token,
            bounds,
            ty,
            semi_token,
        })
    }
    fn parse_optional_bounds(input: ParseStream) -> Result<(Option<Token![:]>, Punctuated<TypeParamBound, Token![+]>)> {
        let colon_token: Option<Token![:]> = input.parse()?;
        let mut bounds = Punctuated::new();
        if colon_token.is_some() {
            loop {
                if input.peek(Token![where]) || input.peek(Token![=]) || input.peek(Token![;]) {
                    break;
                }
                bounds.push_value(input.parse::<TypeParamBound>()?);
                if input.peek(Token![where]) || input.peek(Token![=]) || input.peek(Token![;]) {
                    break;
                }
                bounds.push_punct(input.parse::<Token![+]>()?);
            }
        }
        Ok((colon_token, bounds))
    }
    fn parse_optional_definition(input: ParseStream) -> Result<Option<(Token![=], Type)>> {
        let eq_token: Option<Token![=]> = input.parse()?;
        if let Some(eq_token) = eq_token {
            let definition: Type = input.parse()?;
            Ok(Some((eq_token, definition)))
        } else {
            Ok(None)
        }
    }
}
impl Parse for ItemMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let path = input.call(Path::parse_mod_style)?;
        let bang_token: Token![!] = input.parse()?;
        let ident: Option<Ident> = if input.peek(Token![try]) {
            input.call(Ident::parse_any).map(Some)
        } else {
            input.parse()
        }?;
        let (delimiter, tokens) = input.call(mac::mac_parse_delimiter)?;
        let semi_token: Option<Token![;]> = if !delimiter.is_brace() {
            Some(input.parse()?)
        } else {
            None
        };
        Ok(ItemMacro {
            attrs,
            ident,
            mac: Macro {
                path,
                bang: bang_token,
                delim: delimiter,
                toks: tokens,
            },
            semi_token,
        })
    }
}
fn parse_macro2(begin: ParseBuffer, _vis: Visibility, input: ParseStream) -> Result<Item> {
    input.parse::<Token![macro]>()?;
    input.parse::<Ident>()?;
    let mut lookahead = input.lookahead1();
    if lookahead.peek(tok::Paren) {
        let paren_content;
        parenthesized!(paren_content in input);
        paren_content.parse::<TokenStream>()?;
        lookahead = input.lookahead1();
    }
    if lookahead.peek(tok::Brace) {
        let brace_content;
        braced!(brace_content in input);
        brace_content.parse::<TokenStream>()?;
    } else {
        return Err(lookahead.error());
    }
    Ok(Item::Verbatim(verbatim_between(&begin, input)))
}
impl Parse for ItemExternCrate {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ItemExternCrate {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            extern_token: input.parse()?,
            crate_token: input.parse()?,
            ident: {
                if input.peek(Token![self]) {
                    input.call(Ident::parse_any)?
                } else {
                    input.parse()?
                }
            },
            rename: {
                if input.peek(Token![as]) {
                    let as_token: Token![as] = input.parse()?;
                    let rename: Ident = if input.peek(Token![_]) {
                        Ident::from(input.parse::<Token![_]>()?)
                    } else {
                        input.parse()?
                    };
                    Some((as_token, rename))
                } else {
                    None
                }
            },
            semi_token: input.parse()?,
        })
    }
}
impl Parse for ItemUse {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_crate_root_in_path = false;
        parse_item_use(input, allow_crate_root_in_path).map(Option::unwrap)
    }
}
fn parse_item_use(input: ParseStream, allow_crate_root_in_path: bool) -> Result<Option<ItemUse>> {
    let attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let use_token: Token![use] = input.parse()?;
    let leading_colon: Option<Token![::]> = input.parse()?;
    let tree = parse_use_tree(input, allow_crate_root_in_path && leading_colon.is_none())?;
    let semi_token: Token![;] = input.parse()?;
    let tree = match tree {
        Some(tree) => tree,
        None => return Ok(None),
    };
    Ok(Some(ItemUse {
        attrs,
        vis,
        use_token,
        leading_colon,
        tree,
        semi_token,
    }))
}
impl Parse for UseTree {
    fn parse(input: ParseStream) -> Result<UseTree> {
        let allow_crate_root_in_path = false;
        parse_use_tree(input, allow_crate_root_in_path).map(Option::unwrap)
    }
}
fn parse_use_tree(input: ParseStream, allow_crate_root_in_path: bool) -> Result<Option<UseTree>> {
    let lookahead = input.lookahead1();
    if lookahead.peek(Ident)
        || lookahead.peek(Token![self])
        || lookahead.peek(Token![super])
        || lookahead.peek(Token![crate])
        || lookahead.peek(Token![try])
    {
        let ident = input.call(Ident::parse_any)?;
        if input.peek(Token![::]) {
            Ok(Some(UseTree::Path(UsePath {
                ident,
                colon2_token: input.parse()?,
                tree: Box::new(input.parse()?),
            })))
        } else if input.peek(Token![as]) {
            Ok(Some(UseTree::Rename(UseRename {
                ident,
                as_token: input.parse()?,
                rename: {
                    if input.peek(Ident) {
                        input.parse()?
                    } else if input.peek(Token![_]) {
                        Ident::from(input.parse::<Token![_]>()?)
                    } else {
                        return Err(input.error("expected identifier or underscore"));
                    }
                },
            })))
        } else {
            Ok(Some(UseTree::Name(UseName { ident })))
        }
    } else if lookahead.peek(Token![*]) {
        Ok(Some(UseTree::Glob(UseGlob {
            star_token: input.parse()?,
        })))
    } else if lookahead.peek(tok::Brace) {
        let content;
        let brace_token = braced!(content in input);
        let mut items = Punctuated::new();
        let mut has_any_crate_root_in_path = false;
        loop {
            if content.is_empty() {
                break;
            }
            let this_tree_starts_with_crate_root =
                allow_crate_root_in_path && content.parse::<Option<Token![::]>>()?.is_some();
            has_any_crate_root_in_path |= this_tree_starts_with_crate_root;
            match parse_use_tree(&content, allow_crate_root_in_path && !this_tree_starts_with_crate_root)? {
                Some(tree) => items.push_value(tree),
                None => has_any_crate_root_in_path = true,
            }
            if content.is_empty() {
                break;
            }
            let comma: Token![,] = content.parse()?;
            items.push_punct(comma);
        }
        if has_any_crate_root_in_path {
            Ok(None)
        } else {
            Ok(Some(UseTree::Group(UseGroup { brace_token, items })))
        }
    } else {
        Err(lookahead.error())
    }
}
impl Parse for ItemStatic {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ItemStatic {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            static_token: input.parse()?,
            mutability: input.parse()?,
            ident: input.parse()?,
            colon_token: input.parse()?,
            ty: input.parse()?,
            eq_token: input.parse()?,
            expr: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}
impl Parse for ItemConst {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ItemConst {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            const_token: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            generics: Generics::default(),
            colon_token: input.parse()?,
            ty: input.parse()?,
            eq_token: input.parse()?,
            expr: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}
fn peek_signature(input: ParseStream) -> bool {
    let fork = input.fork();
    fork.parse::<Option<Token![const]>>().is_ok()
        && fork.parse::<Option<Token![async]>>().is_ok()
        && fork.parse::<Option<Token![unsafe]>>().is_ok()
        && fork.parse::<Option<Abi>>().is_ok()
        && fork.peek(Token![fn])
}
impl Parse for Signature {
    fn parse(input: ParseStream) -> Result<Self> {
        let constness: Option<Token![const]> = input.parse()?;
        let asyncness: Option<Token![async]> = input.parse()?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let abi: Option<Abi> = input.parse()?;
        let fn_token: Token![fn] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        let content;
        let paren_token = parenthesized!(content in input);
        let (inputs, variadic) = parse_fn_args(&content)?;
        let output: ReturnType = input.parse()?;
        generics.clause = input.parse()?;
        Ok(Signature {
            constness,
            asyncness,
            unsafety,
            abi,
            fn_token,
            ident,
            generics,
            paren_token,
            inputs,
            variadic,
            output,
        })
    }
}
impl Parse for ItemFn {
    fn parse(input: ParseStream) -> Result<Self> {
        let outer_attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        parse_rest_of_fn(input, outer_attrs, vis, sig)
    }
}
fn parse_rest_of_fn(input: ParseStream, mut attrs: Vec<Attribute>, vis: Visibility, sig: Signature) -> Result<ItemFn> {
    let content;
    let brace_token = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let stmts = content.call(Block::parse_within)?;
    Ok(ItemFn {
        attrs,
        vis,
        sig,
        block: Box::new(Block {
            brace: brace_token,
            stmts,
        }),
    })
}
impl Parse for FnArg {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_variadic = false;
        let attrs = input.call(Attribute::parse_outer)?;
        match parse_fn_arg_or_variadic(input, attrs, allow_variadic)? {
            FnArgOrVariadic::FnArg(arg) => Ok(arg),
            FnArgOrVariadic::Variadic(_) => unreachable!(),
        }
    }
}
enum FnArgOrVariadic {
    FnArg(FnArg),
    Variadic(Variadic),
}
fn parse_fn_arg_or_variadic(
    input: ParseStream,
    attrs: Vec<Attribute>,
    allow_variadic: bool,
) -> Result<FnArgOrVariadic> {
    let ahead = input.fork();
    if let Ok(mut receiver) = ahead.parse::<Receiver>() {
        input.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(FnArg::Receiver(receiver)));
    }
    if input.peek(Ident) && input.peek2(Token![<]) {
        let span = input.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVariadic::FnArg(FnArg::Typed(PatType {
            attrs,
            pat: Box::new(Pat::Wild(PatWild {
                attrs: Vec::new(),
                underscore: Token![_](span),
            })),
            colon: Token![:](span),
            ty: input.parse()?,
        })));
    }
    let pat = Box::new(Pat::parse_single(input)?);
    let colon_token: Token![:] = input.parse()?;
    if allow_variadic {
        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            return Ok(FnArgOrVariadic::Variadic(Variadic {
                attrs,
                pat: Some((pat, colon_token)),
                dots,
                comma: None,
            }));
        }
    }
    Ok(FnArgOrVariadic::FnArg(FnArg::Typed(PatType {
        attrs,
        pat,
        colon: colon_token,
        ty: input.parse()?,
    })))
}
impl Parse for Receiver {
    fn parse(input: ParseStream) -> Result<Self> {
        let reference = if input.peek(Token![&]) {
            let ampersand: Token![&] = input.parse()?;
            let lifetime: Option<Lifetime> = input.parse()?;
            Some((ampersand, lifetime))
        } else {
            None
        };
        let mutability: Option<Token![mut]> = input.parse()?;
        let self_token: Token![self] = input.parse()?;
        let colon_token: Option<Token![:]> = if reference.is_some() { None } else { input.parse()? };
        let ty: Type = if colon_token.is_some() {
            input.parse()?
        } else {
            let mut ty = Type::Path(TypePath {
                qself: None,
                path: Path::from(Ident::new("Self", self_token.span)),
            });
            if let Some((ampersand, lifetime)) = reference.as_ref() {
                ty = Type::Reference(TypeReference {
                    and_: Token![&](ampersand.span),
                    lifetime: lifetime.clone(),
                    mut_: mutability.as_ref().map(|m| Token![mut](m.span)),
                    elem: Box::new(ty),
                });
            }
            ty
        };
        Ok(Receiver {
            attrs: Vec::new(),
            reference,
            mutability,
            self_token,
            colon_token,
            ty: Box::new(ty),
        })
    }
}
fn parse_fn_args(input: ParseStream) -> Result<(Punctuated<FnArg, Token![,]>, Option<Variadic>)> {
    let mut args = Punctuated::new();
    let mut variadic = None;
    let mut has_receiver = false;
    while !input.is_empty() {
        let attrs = input.call(Attribute::parse_outer)?;
        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            variadic = Some(Variadic {
                attrs,
                pat: None,
                dots,
                comma: if input.is_empty() { None } else { Some(input.parse()?) },
            });
            break;
        }
        let allow_variadic = true;
        let arg = match parse_fn_arg_or_variadic(input, attrs, allow_variadic)? {
            FnArgOrVariadic::FnArg(arg) => arg,
            FnArgOrVariadic::Variadic(arg) => {
                variadic = Some(Variadic {
                    comma: if input.is_empty() { None } else { Some(input.parse()?) },
                    ..arg
                });
                break;
            },
        };
        match &arg {
            FnArg::Receiver(receiver) if has_receiver => {
                return Err(Err::new(receiver.self_token.span, "unexpected second method receiver"));
            },
            FnArg::Receiver(receiver) if !args.is_empty() => {
                return Err(Err::new(receiver.self_token.span, "unexpected method receiver"));
            },
            FnArg::Receiver(_) => has_receiver = true,
            FnArg::Typed(_) => {},
        }
        args.push_value(arg);
        if input.is_empty() {
            break;
        }
        let comma: Token![,] = input.parse()?;
        args.push_punct(comma);
    }
    Ok((args, variadic))
}
impl Parse for ItemMod {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let mod_token: Token![mod] = input.parse()?;
        let ident: Ident = if input.peek(Token![try]) {
            input.call(Ident::parse_any)
        } else {
            input.parse()
        }?;
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![;]) {
            Ok(ItemMod {
                attrs,
                vis,
                unsafety,
                mod_token,
                ident,
                content: None,
                semi: Some(input.parse()?),
            })
        } else if lookahead.peek(tok::Brace) {
            let content;
            let brace_token = braced!(content in input);
            parse_inner(&content, &mut attrs)?;
            let mut items = Vec::new();
            while !content.is_empty() {
                items.push(content.parse()?);
            }
            Ok(ItemMod {
                attrs,
                vis,
                unsafety,
                mod_token,
                ident,
                content: Some((brace_token, items)),
                semi: None,
            })
        } else {
            Err(lookahead.error())
        }
    }
}
impl Parse for ItemForeignMod {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let abi: Abi = input.parse()?;
        let content;
        let brace_token = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let mut items = Vec::new();
        while !content.is_empty() {
            items.push(content.parse()?);
        }
        Ok(ItemForeignMod {
            attrs,
            unsafety,
            abi,
            brace_token,
            items,
        })
    }
}
impl Parse for ForeignItem {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let mut attrs = input.call(Attribute::parse_outer)?;
        let ahead = input.fork();
        let vis: Visibility = ahead.parse()?;
        let lookahead = ahead.lookahead1();
        let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
            let vis: Visibility = input.parse()?;
            let sig: Signature = input.parse()?;
            if input.peek(tok::Brace) {
                let content;
                braced!(content in input);
                content.call(Attribute::parse_inner)?;
                content.call(Block::parse_within)?;
                Ok(ForeignItem::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(ForeignItem::Fn(ForeignItemFn {
                    attrs: Vec::new(),
                    vis,
                    sig,
                    semi_token: input.parse()?,
                }))
            }
        } else if lookahead.peek(Token![static]) {
            let vis = input.parse()?;
            let static_token = input.parse()?;
            let mutability = input.parse()?;
            let ident = input.parse()?;
            let colon_token = input.parse()?;
            let ty = input.parse()?;
            if input.peek(Token![=]) {
                input.parse::<Token![=]>()?;
                input.parse::<Expr>()?;
                input.parse::<Token![;]>()?;
                Ok(ForeignItem::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(ForeignItem::Static(ForeignItemStatic {
                    attrs: Vec::new(),
                    vis,
                    static_token,
                    mutability,
                    ident,
                    colon_token,
                    ty,
                    semi_token: input.parse()?,
                }))
            }
        } else if lookahead.peek(Token![type]) {
            parse_foreign_item_type(begin, input)
        } else if vis.is_inherited()
            && (lookahead.peek(Ident)
                || lookahead.peek(Token![self])
                || lookahead.peek(Token![super])
                || lookahead.peek(Token![crate])
                || lookahead.peek(Token![::]))
        {
            input.parse().map(ForeignItem::Macro)
        } else {
            Err(lookahead.error())
        }?;
        let item_attrs = match &mut item {
            ForeignItem::Fn(item) => &mut item.attrs,
            ForeignItem::Static(item) => &mut item.attrs,
            ForeignItem::Type(item) => &mut item.attrs,
            ForeignItem::Macro(item) => &mut item.attrs,
            ForeignItem::Verbatim(_) => return Ok(item),
        };
        attrs.append(item_attrs);
        *item_attrs = attrs;
        Ok(item)
    }
}
impl Parse for ForeignItemFn {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let semi_token: Token![;] = input.parse()?;
        Ok(ForeignItemFn {
            attrs,
            vis,
            sig,
            semi_token,
        })
    }
}
impl Parse for ForeignItemStatic {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ForeignItemStatic {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            static_token: input.parse()?,
            mutability: input.parse()?,
            ident: input.parse()?,
            colon_token: input.parse()?,
            ty: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}
impl Parse for ForeignItemType {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ForeignItemType {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            type_token: input.parse()?,
            ident: input.parse()?,
            generics: {
                let mut generics: Generics = input.parse()?;
                generics.clause = input.parse()?;
                generics
            },
            semi_token: input.parse()?,
        })
    }
}
fn parse_foreign_item_type(begin: ParseBuffer, input: ParseStream) -> Result<ForeignItem> {
    let FlexibleItemType {
        vis,
        defaultness: _,
        type_token,
        ident,
        generics,
        colon_token,
        bounds: _,
        ty,
        semi_token,
    } = FlexibleItemType::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::Both)?;
    if colon_token.is_some() || ty.is_some() {
        Ok(ForeignItem::Verbatim(verbatim_between(&begin, input)))
    } else {
        Ok(ForeignItem::Type(ForeignItemType {
            attrs: Vec::new(),
            vis,
            type_token,
            ident,
            generics,
            semi_token,
        }))
    }
}
impl Parse for ForeignItemMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi_token: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(ForeignItemMacro { attrs, mac, semi_token })
    }
}
impl Parse for ItemType {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ItemType {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            type_token: input.parse()?,
            ident: input.parse()?,
            generics: {
                let mut generics: Generics = input.parse()?;
                generics.clause = input.parse()?;
                generics
            },
            eq_token: input.parse()?,
            ty: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}
fn parse_item_type(begin: ParseBuffer, input: ParseStream) -> Result<Item> {
    let FlexibleItemType {
        vis,
        defaultness: _,
        type_token,
        ident,
        generics,
        colon_token,
        bounds: _,
        ty,
        semi_token,
    } = FlexibleItemType::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::BeforeEq)?;
    let (eq_token, ty) = match ty {
        Some(ty) if colon_token.is_none() => ty,
        _ => return Ok(Item::Verbatim(verbatim_between(&begin, input))),
    };
    Ok(Item::Type(ItemType {
        attrs: Vec::new(),
        vis,
        type_token,
        ident,
        generics,
        eq_token,
        ty: Box::new(ty),
        semi_token,
    }))
}
impl Parse for ItemStruct {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let struct_token = input.parse::<Token![struct]>()?;
        let ident = input.parse::<Ident>()?;
        let generics = input.parse::<Generics>()?;
        let (where_clause, fields, semi_token) = data_struct(input)?;
        Ok(ItemStruct {
            attrs,
            vis,
            struct_token,
            ident,
            generics: Generics {
                clause: where_clause,
                ..generics
            },
            fields,
            semi_token,
        })
    }
}
impl Parse for ItemEnum {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let enum_token = input.parse::<Token![enum]>()?;
        let ident = input.parse::<Ident>()?;
        let generics = input.parse::<Generics>()?;
        let (where_clause, brace_token, variants) = data_enum(input)?;
        Ok(ItemEnum {
            attrs,
            vis,
            enum_token,
            ident,
            generics: Generics {
                clause: where_clause,
                ..generics
            },
            brace_token,
            variants,
        })
    }
}
impl Parse for ItemUnion {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let union_token = input.parse::<Token![union]>()?;
        let ident = input.parse::<Ident>()?;
        let generics = input.parse::<Generics>()?;
        let (where_clause, fields) = data_union(input)?;
        Ok(ItemUnion {
            attrs,
            vis,
            union_token,
            ident,
            generics: Generics {
                clause: where_clause,
                ..generics
            },
            fields,
        })
    }
}
fn parse_trait_or_trait_alias(input: ParseStream) -> Result<Item> {
    let (attrs, vis, trait_token, ident, generics) = parse_start_of_trait_alias(input)?;
    let lookahead = input.lookahead1();
    if lookahead.peek(tok::Brace) || lookahead.peek(Token![:]) || lookahead.peek(Token![where]) {
        let unsafety = None;
        let auto_token = None;
        parse_rest_of_trait(input, attrs, vis, unsafety, auto_token, trait_token, ident, generics).map(Item::Trait)
    } else if lookahead.peek(Token![=]) {
        parse_rest_of_trait_alias(input, attrs, vis, trait_token, ident, generics).map(Item::TraitAlias)
    } else {
        Err(lookahead.error())
    }
}
impl Parse for ItemTrait {
    fn parse(input: ParseStream) -> Result<Self> {
        let outer_attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let auto_token: Option<Token![auto]> = input.parse()?;
        let trait_token: Token![trait] = input.parse()?;
        let ident: Ident = input.parse()?;
        let generics: Generics = input.parse()?;
        parse_rest_of_trait(
            input,
            outer_attrs,
            vis,
            unsafety,
            auto_token,
            trait_token,
            ident,
            generics,
        )
    }
}
fn parse_rest_of_trait(
    input: ParseStream,
    mut attrs: Vec<Attribute>,
    vis: Visibility,
    unsafety: Option<Token![unsafe]>,
    auto_token: Option<Token![auto]>,
    trait_token: Token![trait],
    ident: Ident,
    mut generics: Generics,
) -> Result<ItemTrait> {
    let colon_token: Option<Token![:]> = input.parse()?;
    let mut supertraits = Punctuated::new();
    if colon_token.is_some() {
        loop {
            if input.peek(Token![where]) || input.peek(tok::Brace) {
                break;
            }
            supertraits.push_value(input.parse()?);
            if input.peek(Token![where]) || input.peek(tok::Brace) {
                break;
            }
            supertraits.push_punct(input.parse()?);
        }
    }
    generics.clause = input.parse()?;
    let content;
    let brace_token = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let mut items = Vec::new();
    while !content.is_empty() {
        items.push(content.parse()?);
    }
    Ok(ItemTrait {
        attrs,
        vis,
        unsafety,
        auto_token,
        restriction: None,
        trait_token,
        ident,
        generics,
        colon_token,
        supertraits,
        brace_token,
        items,
    })
}
impl Parse for ItemTraitAlias {
    fn parse(input: ParseStream) -> Result<Self> {
        let (attrs, vis, trait_token, ident, generics) = parse_start_of_trait_alias(input)?;
        parse_rest_of_trait_alias(input, attrs, vis, trait_token, ident, generics)
    }
}
fn parse_start_of_trait_alias(
    input: ParseStream,
) -> Result<(Vec<Attribute>, Visibility, Token![trait], Ident, Generics)> {
    let attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let trait_token: Token![trait] = input.parse()?;
    let ident: Ident = input.parse()?;
    let generics: Generics = input.parse()?;
    Ok((attrs, vis, trait_token, ident, generics))
}
fn parse_rest_of_trait_alias(
    input: ParseStream,
    attrs: Vec<Attribute>,
    vis: Visibility,
    trait_token: Token![trait],
    ident: Ident,
    mut generics: Generics,
) -> Result<ItemTraitAlias> {
    let eq_token: Token![=] = input.parse()?;
    let mut bounds = Punctuated::new();
    loop {
        if input.peek(Token![where]) || input.peek(Token![;]) {
            break;
        }
        bounds.push_value(input.parse()?);
        if input.peek(Token![where]) || input.peek(Token![;]) {
            break;
        }
        bounds.push_punct(input.parse()?);
    }
    generics.clause = input.parse()?;
    let semi_token: Token![;] = input.parse()?;
    Ok(ItemTraitAlias {
        attrs,
        vis,
        trait_token,
        ident,
        generics,
        eq_token,
        bounds,
        semi_token,
    })
}
impl Parse for TraitItem {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let mut attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let defaultness: Option<Token![default]> = input.parse()?;
        let ahead = input.fork();
        let lookahead = ahead.lookahead1();
        let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
            input.parse().map(TraitItem::Fn)
        } else if lookahead.peek(Token![const]) {
            ahead.parse::<Token![const]>()?;
            let lookahead = ahead.lookahead1();
            if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                input.parse().map(TraitItem::Const)
            } else if lookahead.peek(Token![async])
                || lookahead.peek(Token![unsafe])
                || lookahead.peek(Token![extern])
                || lookahead.peek(Token![fn])
            {
                input.parse().map(TraitItem::Fn)
            } else {
                Err(lookahead.error())
            }
        } else if lookahead.peek(Token![type]) {
            parse_trait_item_type(begin.fork(), input)
        } else if vis.is_inherited()
            && defaultness.is_none()
            && (lookahead.peek(Ident)
                || lookahead.peek(Token![self])
                || lookahead.peek(Token![super])
                || lookahead.peek(Token![crate])
                || lookahead.peek(Token![::]))
        {
            input.parse().map(TraitItem::Macro)
        } else {
            Err(lookahead.error())
        }?;
        match (vis, defaultness) {
            (Visibility::Inherited, None) => {},
            _ => return Ok(TraitItem::Verbatim(verbatim_between(&begin, input))),
        }
        let item_attrs = match &mut item {
            TraitItem::Const(item) => &mut item.attrs,
            TraitItem::Fn(item) => &mut item.attrs,
            TraitItem::Type(item) => &mut item.attrs,
            TraitItem::Macro(item) => &mut item.attrs,
            TraitItem::Verbatim(_) => unreachable!(),
        };
        attrs.append(item_attrs);
        *item_attrs = attrs;
        Ok(item)
    }
}
impl Parse for TraitItemConst {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(TraitItemConst {
            attrs: input.call(Attribute::parse_outer)?,
            const_token: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            generics: Generics::default(),
            colon_token: input.parse()?,
            ty: input.parse()?,
            default: {
                if input.peek(Token![=]) {
                    let eq_token: Token![=] = input.parse()?;
                    let default: Expr = input.parse()?;
                    Some((eq_token, default))
                } else {
                    None
                }
            },
            semi_token: input.parse()?,
        })
    }
}
impl Parse for TraitItemFn {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let sig: Signature = input.parse()?;
        let lookahead = input.lookahead1();
        let (brace_token, stmts, semi_token) = if lookahead.peek(tok::Brace) {
            let content;
            let brace_token = braced!(content in input);
            parse_inner(&content, &mut attrs)?;
            let stmts = content.call(Block::parse_within)?;
            (Some(brace_token), stmts, None)
        } else if lookahead.peek(Token![;]) {
            let semi_token: Token![;] = input.parse()?;
            (None, Vec::new(), Some(semi_token))
        } else {
            return Err(lookahead.error());
        };
        Ok(TraitItemFn {
            attrs,
            sig,
            default: brace_token.map(|brace_token| Block {
                brace: brace_token,
                stmts,
            }),
            semi_token,
        })
    }
}
impl Parse for TraitItemType {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let type_token: Token![type] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        let (colon_token, bounds) = FlexibleItemType::parse_optional_bounds(input)?;
        let default = FlexibleItemType::parse_optional_definition(input)?;
        generics.clause = input.parse()?;
        let semi_token: Token![;] = input.parse()?;
        Ok(TraitItemType {
            attrs,
            type_token,
            ident,
            generics,
            colon_token,
            bounds,
            default,
            semi_token,
        })
    }
}
fn parse_trait_item_type(begin: ParseBuffer, input: ParseStream) -> Result<TraitItem> {
    let FlexibleItemType {
        vis,
        defaultness: _,
        type_token,
        ident,
        generics,
        colon_token,
        bounds,
        ty,
        semi_token,
    } = FlexibleItemType::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::AfterEq)?;
    if vis.is_some() {
        Ok(TraitItem::Verbatim(verbatim_between(&begin, input)))
    } else {
        Ok(TraitItem::Type(TraitItemType {
            attrs: Vec::new(),
            type_token,
            ident,
            generics,
            colon_token,
            bounds,
            default: ty,
            semi_token,
        }))
    }
}
impl Parse for TraitItemMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi_token: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(TraitItemMacro { attrs, mac, semi_token })
    }
}
impl Parse for ItemImpl {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_verbatim_impl = false;
        parse_impl(input, allow_verbatim_impl).map(Option::unwrap)
    }
}
fn parse_impl(input: ParseStream, allow_verbatim_impl: bool) -> Result<Option<ItemImpl>> {
    let mut attrs = input.call(Attribute::parse_outer)?;
    let has_visibility = allow_verbatim_impl && input.parse::<Visibility>()?.is_some();
    let defaultness: Option<Token![default]> = input.parse()?;
    let unsafety: Option<Token![unsafe]> = input.parse()?;
    let impl_token: Token![impl] = input.parse()?;
    let has_generics = input.peek(Token![<])
        && (input.peek2(Token![>])
            || input.peek2(Token![#])
            || (input.peek2(Ident) || input.peek2(Lifetime))
                && (input.peek3(Token![:])
                    || input.peek3(Token![,])
                    || input.peek3(Token![>])
                    || input.peek3(Token![=]))
            || input.peek2(Token![const]));
    let mut generics: Generics = if has_generics {
        input.parse()?
    } else {
        Generics::default()
    };
    let is_const_impl =
        allow_verbatim_impl && (input.peek(Token![const]) || input.peek(Token![?]) && input.peek2(Token![const]));
    if is_const_impl {
        input.parse::<Option<Token![?]>>()?;
        input.parse::<Token![const]>()?;
    }
    let begin = input.fork();
    let polarity = if input.peek(Token![!]) && !input.peek2(tok::Brace) {
        Some(input.parse::<Token![!]>()?)
    } else {
        None
    };
    #[cfg(not(feature = "printing"))]
    let first_ty_span = input.span();
    let mut first_ty: Type = input.parse()?;
    let self_ty: Type;
    let trait_;
    let is_impl_for = input.peek(Token![for]);
    if is_impl_for {
        let for_token: Token![for] = input.parse()?;
        let mut first_ty_ref = &first_ty;
        while let Type::Group(ty) = first_ty_ref {
            first_ty_ref = &ty.elem;
        }
        if let Type::Path(TypePath { qself: None, .. }) = first_ty_ref {
            while let Type::Group(ty) = first_ty {
                first_ty = *ty.elem;
            }
            if let Type::Path(TypePath { qself: None, path }) = first_ty {
                trait_ = Some((polarity, path, for_token));
            } else {
                unreachable!();
            }
        } else if !allow_verbatim_impl {
            return Err(Err::new_spanned(first_ty_ref, "expected trait path"));
            #[cfg(not(feature = "printing"))]
            return Err(Err::new(first_ty_span, "expected trait path"));
        } else {
            trait_ = None;
        }
        self_ty = input.parse()?;
    } else {
        trait_ = None;
        self_ty = if polarity.is_none() {
            first_ty
        } else {
            Type::Verbatim(verbatim_between(&begin, input))
        };
    }
    generics.clause = input.parse()?;
    let content;
    let brace_token = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let mut items = Vec::new();
    while !content.is_empty() {
        items.push(content.parse()?);
    }
    if has_visibility || is_const_impl || is_impl_for && trait_.is_none() {
        Ok(None)
    } else {
        Ok(Some(ItemImpl {
            attrs,
            defaultness,
            unsafety,
            impl_token,
            generics,
            trait_,
            self_ty: Box::new(self_ty),
            brace_token,
            items,
        }))
    }
}
impl Parse for ImplItem {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let mut attrs = input.call(Attribute::parse_outer)?;
        let ahead = input.fork();
        let vis: Visibility = ahead.parse()?;
        let mut lookahead = ahead.lookahead1();
        let defaultness = if lookahead.peek(Token![default]) && !ahead.peek2(Token![!]) {
            let defaultness: Token![default] = ahead.parse()?;
            lookahead = ahead.lookahead1();
            Some(defaultness)
        } else {
            None
        };
        let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
            let allow_omitted_body = true;
            if let Some(item) = parse_impl_item_fn(input, allow_omitted_body)? {
                Ok(ImplItem::Fn(item))
            } else {
                Ok(ImplItem::Verbatim(verbatim_between(&begin, input)))
            }
        } else if lookahead.peek(Token![const]) {
            input.advance_to(&ahead);
            let const_token: Token![const] = input.parse()?;
            let lookahead = input.lookahead1();
            let ident = if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                input.call(Ident::parse_any)?
            } else {
                return Err(lookahead.error());
            };
            let colon_token: Token![:] = input.parse()?;
            let ty: Type = input.parse()?;
            if let Some(eq_token) = input.parse()? {
                return Ok(ImplItem::Const(ImplItemConst {
                    attrs,
                    vis,
                    defaultness,
                    const_token,
                    ident,
                    generics: Generics::default(),
                    colon_token,
                    ty,
                    eq_token,
                    expr: input.parse()?,
                    semi_token: input.parse()?,
                }));
            } else {
                input.parse::<Token![;]>()?;
                return Ok(ImplItem::Verbatim(verbatim_between(&begin, input)));
            }
        } else if lookahead.peek(Token![type]) {
            parse_impl_item_type(begin, input)
        } else if vis.is_inherited()
            && defaultness.is_none()
            && (lookahead.peek(Ident)
                || lookahead.peek(Token![self])
                || lookahead.peek(Token![super])
                || lookahead.peek(Token![crate])
                || lookahead.peek(Token![::]))
        {
            input.parse().map(ImplItem::Macro)
        } else {
            Err(lookahead.error())
        }?;
        {
            let item_attrs = match &mut item {
                ImplItem::Const(item) => &mut item.attrs,
                ImplItem::Fn(item) => &mut item.attrs,
                ImplItem::Type(item) => &mut item.attrs,
                ImplItem::Macro(item) => &mut item.attrs,
                ImplItem::Verbatim(_) => return Ok(item),
            };
            attrs.append(item_attrs);
            *item_attrs = attrs;
        }
        Ok(item)
    }
}
impl Parse for ImplItemConst {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(ImplItemConst {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            defaultness: input.parse()?,
            const_token: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            generics: Generics::default(),
            colon_token: input.parse()?,
            ty: input.parse()?,
            eq_token: input.parse()?,
            expr: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}
impl Parse for ImplItemFn {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_omitted_body = false;
        parse_impl_item_fn(input, allow_omitted_body).map(Option::unwrap)
    }
}
fn parse_impl_item_fn(input: ParseStream, allow_omitted_body: bool) -> Result<Option<ImplItemFn>> {
    let mut attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let defaultness: Option<Token![default]> = input.parse()?;
    let sig: Signature = input.parse()?;
    if allow_omitted_body && input.parse::<Option<Token![;]>>()?.is_some() {
        return Ok(None);
    }
    let content;
    let brace_token = braced!(content in input);
    attrs.extend(content.call(Attribute::parse_inner)?);
    let block = Block {
        brace: brace_token,
        stmts: content.call(Block::parse_within)?,
    };
    Ok(Some(ImplItemFn {
        attrs,
        vis,
        defaultness,
        sig,
        block,
    }))
}
impl Parse for ImplItemType {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let defaultness: Option<Token![default]> = input.parse()?;
        let type_token: Token![type] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        let eq_token: Token![=] = input.parse()?;
        let ty: Type = input.parse()?;
        generics.clause = input.parse()?;
        let semi_token: Token![;] = input.parse()?;
        Ok(ImplItemType {
            attrs,
            vis,
            defaultness,
            type_token,
            ident,
            generics,
            eq_token,
            ty,
            semi_token,
        })
    }
}
fn parse_impl_item_type(begin: ParseBuffer, input: ParseStream) -> Result<ImplItem> {
    let FlexibleItemType {
        vis,
        defaultness,
        type_token,
        ident,
        generics,
        colon_token,
        bounds: _,
        ty,
        semi_token,
    } = FlexibleItemType::parse(input, TypeDefaultness::Optional, WhereClauseLocation::AfterEq)?;
    let (eq_token, ty) = match ty {
        Some(ty) if colon_token.is_none() => ty,
        _ => return Ok(ImplItem::Verbatim(verbatim_between(&begin, input))),
    };
    Ok(ImplItem::Type(ImplItemType {
        attrs: Vec::new(),
        vis,
        defaultness,
        type_token,
        ident,
        generics,
        eq_token,
        ty,
        semi_token,
    }))
}
impl Parse for ImplItemMacro {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi_token: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(ImplItemMacro { attrs, mac, semi_token })
    }
}
impl Visibility {
    fn is_inherited(&self) -> bool {
        match self {
            Visibility::Inherited => true,
            _ => false,
        }
    }
}
impl MacroDelimiter {
    pub(crate) fn is_brace(&self) -> bool {
        match self {
            MacroDelimiter::Brace(_) => true,
            MacroDelimiter::Paren(_) | MacroDelimiter::Bracket(_) => false,
        }
    }
}
impl Parse for StaticMutability {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut_token: Option<Token![mut]> = input.parse()?;
        Ok(mut_token.map_or(StaticMutability::None, StaticMutability::Mut))
    }
}

mod parsing {
    struct AllowNoSemi(bool);
    impl Block {
        pub fn parse_within(x: ParseStream) -> Result<Vec<Stmt>> {
            let mut ys = Vec::new();
            loop {
                while let semi @ Some(_) = x.parse()? {
                    ys.push(Stmt::Expr(Expr::Verbatim(TokenStream::new()), semi));
                }
                if x.is_empty() {
                    break;
                }
                let stmt = parse_stmt(x, AllowNoSemi(true))?;
                let requires_semicolon = match &stmt {
                    Stmt::Expr(x, None) => expr::requires_terminator(x),
                    Stmt::Macro(x) => x.semi_token.is_none() && !x.mac.delimiter.is_brace(),
                    Stmt::Local(_) | Stmt::Item(_) | Stmt::Expr(_, Some(_)) => false,
                };
                ys.push(stmt);
                if x.is_empty() {
                    break;
                } else if requires_semicolon {
                    return Err(x.error("unexpected token, expected `;`"));
                }
            }
            Ok(ys)
        }
    }
    impl Parse for Block {
        fn parse(x: ParseStream) -> Result<Self> {
            let content;
            Ok(Block {
                brace_token: braced!(content in x),
                stmts: content.call(Block::parse_within)?,
            })
        }
    }
    impl Parse for Stmt {
        fn parse(x: ParseStream) -> Result<Self> {
            let allow_nosemi = AllowNoSemi(false);
            parse_stmt(x, allow_nosemi)
        }
    }
    fn parse_stmt(x: ParseStream, allow_nosemi: AllowNoSemi) -> Result<Stmt> {
        let begin = x.fork();
        let attrs = x.call(Attribute::parse_outer)?;
        let ahead = x.fork();
        let mut is_item_macro = false;
        if let Ok(path) = ahead.call(Path::parse_mod_style) {
            if ahead.peek(Token![!]) {
                if ahead.peek2(Ident) || ahead.peek2(Token![try]) {
                    is_item_macro = true;
                } else if ahead.peek2(token::Brace) && !(ahead.peek3(Token![.]) || ahead.peek3(Token![?])) {
                    x.advance_to(&ahead);
                    return stmt_mac(x, attrs, path).map(Stmt::Macro);
                }
            }
        }
        if x.peek(Token![let]) {
            stmt_local(x, attrs).map(Stmt::Local)
        } else if x.peek(Token![pub])
            || x.peek(Token![crate]) && !x.peek2(Token![::])
            || x.peek(Token![extern])
            || x.peek(Token![use])
            || x.peek(Token![static])
                && (x.peek2(Token![mut])
                    || x.peek2(Ident) && !(x.peek2(Token![async]) && (x.peek3(Token![move]) || x.peek3(Token![|]))))
            || x.peek(Token![const])
                && !(x.peek2(token::Brace)
                    || x.peek2(Token![static])
                    || x.peek2(Token![async])
                        && !(x.peek3(Token![unsafe]) || x.peek3(Token![extern]) || x.peek3(Token![fn]))
                    || x.peek2(Token![move])
                    || x.peek2(Token![|]))
            || x.peek(Token![unsafe]) && !x.peek2(token::Brace)
            || x.peek(Token![async]) && (x.peek2(Token![unsafe]) || x.peek2(Token![extern]) || x.peek2(Token![fn]))
            || x.peek(Token![fn])
            || x.peek(Token![mod])
            || x.peek(Token![type])
            || x.peek(Token![struct])
            || x.peek(Token![enum])
            || x.peek(Token![union]) && x.peek2(Ident)
            || x.peek(Token![auto]) && x.peek2(Token![trait])
            || x.peek(Token![trait])
            || x.peek(Token![default]) && (x.peek2(Token![unsafe]) || x.peek2(Token![impl]))
            || x.peek(Token![impl])
            || x.peek(Token![macro])
            || is_item_macro
        {
            let item = parse_rest_of_item(begin, attrs, x)?;
            Ok(Stmt::Item(item))
        } else {
            stmt_expr(x, allow_nosemi, attrs)
        }
    }
    fn stmt_mac(x: ParseStream, attrs: Vec<Attribute>, path: Path) -> Result<StmtMacro> {
        let bang_token: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac_parse_delimiter(x)?;
        let semi_token: Option<Token![;]> = x.parse()?;
        Ok(StmtMacro {
            attrs,
            mac: Macro {
                path,
                bang_token,
                delimiter,
                tokens,
            },
            semi_token,
        })
    }
    fn stmt_local(x: ParseStream, attrs: Vec<Attribute>) -> Result<Local> {
        let let_token: Token![let] = x.parse()?;
        let mut pat = Pat::parse_single(x)?;
        if x.peek(Token![:]) {
            let colon_token: Token![:] = x.parse()?;
            let ty: Type = x.parse()?;
            pat = Pat::Type(PatType {
                attrs: Vec::new(),
                pat: Box::new(pat),
                colon_token,
                ty: Box::new(ty),
            });
        }
        let init = if let Some(eq_token) = x.parse()? {
            let eq_token: Token![=] = eq_token;
            let expr: Expr = x.parse()?;
            let diverge = if let Some(else_token) = x.parse()? {
                let else_token: Token![else] = else_token;
                let diverge = ExprBlock {
                    attrs: Vec::new(),
                    label: None,
                    block: x.parse()?,
                };
                Some((else_token, Box::new(Expr::Block(diverge))))
            } else {
                None
            };
            Some(LocalInit {
                eq_token,
                expr: Box::new(expr),
                diverge,
            })
        } else {
            None
        };
        let semi_token: Token![;] = x.parse()?;
        Ok(Local {
            attrs,
            let_token,
            pat,
            init,
            semi_token,
        })
    }
    fn stmt_expr(x: ParseStream, allow_nosemi: AllowNoSemi, mut attrs: Vec<Attribute>) -> Result<Stmt> {
        let mut e = expr_early(x)?;
        let mut attr_target = &mut e;
        loop {
            attr_target = match attr_target {
                Expr::Assign(e) => &mut e.left,
                Expr::Binary(e) => &mut e.left,
                Expr::Cast(e) => &mut e.expr,
                Expr::Array(_)
                | Expr::Async(_)
                | Expr::Await(_)
                | Expr::Block(_)
                | Expr::Break(_)
                | Expr::Call(_)
                | Expr::Closure(_)
                | Expr::Const(_)
                | Expr::Continue(_)
                | Expr::Field(_)
                | Expr::ForLoop(_)
                | Expr::Group(_)
                | Expr::If(_)
                | Expr::Index(_)
                | Expr::Infer(_)
                | Expr::Let(_)
                | Expr::Lit(_)
                | Expr::Loop(_)
                | Expr::Macro(_)
                | Expr::Match(_)
                | Expr::MethodCall(_)
                | Expr::Paren(_)
                | Expr::Path(_)
                | Expr::Range(_)
                | Expr::Reference(_)
                | Expr::Repeat(_)
                | Expr::Return(_)
                | Expr::Struct(_)
                | Expr::Try(_)
                | Expr::TryBlock(_)
                | Expr::Tuple(_)
                | Expr::Unary(_)
                | Expr::Unsafe(_)
                | Expr::While(_)
                | Expr::Yield(_)
                | Expr::Verbatim(_) => break,
            };
        }
        attrs.extend(attr_target.replace_attrs(Vec::new()));
        attr_target.replace_attrs(attrs);
        let semi_token: Option<Token![;]> = x.parse()?;
        match e {
            Expr::Macro(ExprMacro { attrs, mac }) if semi_token.is_some() || mac.delimiter.is_brace() => {
                return Ok(Stmt::Macro(StmtMacro { attrs, mac, semi_token }));
            },
            _ => {},
        }
        if semi_token.is_some() {
            Ok(Stmt::Expr(e, semi_token))
        } else if allow_nosemi.0 || !expr::requires_terminator(&e) {
            Ok(Stmt::Expr(e, None))
        } else {
            Err(x.error("expected semicolon"))
        }
    }
}

mod parsing {
    impl Parse for Variant {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let _visibility: Visibility = input.parse()?;
            let ident: Ident = input.parse()?;
            let fields = if input.peek(token::Brace) {
                Fields::Named(input.parse()?)
            } else if input.peek(token::Paren) {
                Fields::Unnamed(input.parse()?)
            } else {
                Fields::Unit
            };
            let discriminant = if input.peek(Token![=]) {
                let eq_token: Token![=] = input.parse()?;
                let discriminant: Expr = input.parse()?;
                Some((eq_token, discriminant))
            } else {
                None
            };
            Ok(Variant {
                attrs,
                ident,
                fields,
                discriminant,
            })
        }
    }
    impl Parse for FieldsNamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsNamed {
                brace_token: braced!(content in input),
                named: content.parse_terminated(Field::parse_named, Token![,])?,
            })
        }
    }
    impl Parse for FieldsUnnamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsUnnamed {
                paren_token: parenthesized!(content in input),
                unnamed: content.parse_terminated(Field::parse_unnamed, Token![,])?,
            })
        }
    }
    impl Field {
        pub fn parse_named(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: Some(if input.peek(Token![_]) {
                    input.call(Ident::parse_any)
                } else {
                    input.parse()
                }?),
                colon_token: Some(input.parse()?),
                ty: input.parse()?,
            })
        }
        pub fn parse_unnamed(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: None,
                colon_token: None,
                ty: input.parse()?,
            })
        }
    }
}

mod parsing {
    impl Parse for DeriveInput {
        fn parse(input: ParseStream) -> Result<Self> {
            let attrs = input.call(Attribute::parse_outer)?;
            let vis = input.parse::<Visibility>()?;
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![struct]) {
                let struct_token = input.parse::<Token![struct]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, fields, semi) = data_struct(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Struct(DataStruct {
                        struct_token,
                        fields,
                        semi_token: semi,
                    }),
                })
            } else if lookahead.peek(Token![enum]) {
                let enum_token = input.parse::<Token![enum]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, brace, variants) = data_enum(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Enum(DataEnum {
                        enum_token,
                        brace_token: brace,
                        variants,
                    }),
                })
            } else if lookahead.peek(Token![union]) {
                let union_token = input.parse::<Token![union]>()?;
                let ident = input.parse::<Ident>()?;
                let generics = input.parse::<Generics>()?;
                let (where_clause, fields) = data_union(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    generics: Generics {
                        where_clause,
                        ..generics
                    },
                    data: Data::Union(DataUnion { union_token, fields }),
                })
            } else {
                Err(lookahead.error())
            }
        }
    }
    pub fn data_struct(input: ParseStream) -> Result<(Option<WhereClause>, Fields, Option<Token![;]>)> {
        let mut lookahead = input.lookahead1();
        let mut where_clause = None;
        if lookahead.peek(Token![where]) {
            where_clause = Some(input.parse()?);
            lookahead = input.lookahead1();
        }
        if where_clause.is_none() && lookahead.peek(token::Paren) {
            let fields = input.parse()?;
            lookahead = input.lookahead1();
            if lookahead.peek(Token![where]) {
                where_clause = Some(input.parse()?);
                lookahead = input.lookahead1();
            }
            if lookahead.peek(Token![;]) {
                let semi = input.parse()?;
                Ok((where_clause, Fields::Unnamed(fields), Some(semi)))
            } else {
                Err(lookahead.error())
            }
        } else if lookahead.peek(token::Brace) {
            let fields = input.parse()?;
            Ok((where_clause, Fields::Named(fields), None))
        } else if lookahead.peek(Token![;]) {
            let semi = input.parse()?;
            Ok((where_clause, Fields::Unit, Some(semi)))
        } else {
            Err(lookahead.error())
        }
    }
    pub fn data_enum(
        input: ParseStream,
    ) -> Result<(Option<WhereClause>, token::Brace, Punctuated<Variant, Token![,]>)> {
        let where_clause = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let variants = content.parse_terminated(Variant::parse, Token![,])?;
        Ok((where_clause, brace, variants))
    }
    pub fn data_union(input: ParseStream) -> Result<(Option<WhereClause>, FieldsNamed)> {
        let where_clause = input.parse()?;
        let fields = input.parse()?;
        Ok((where_clause, fields))
    }
}

mod parsing {
    impl Parse for File {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(File {
                shebang: None,
                attrs: input.call(Attribute::parse_inner)?,
                items: {
                    let mut items = Vec::new();
                    while !input.is_empty() {
                        items.push(input.parse()?);
                    }
                    items
                },
            })
        }
    }
}

mod parsing {
    fn accept_as_ident(ident: &Ident) -> bool {
        match ident.to_string().as_str() {
            "_" | "abstract" | "as" | "async" | "await" | "become" | "box" | "break" | "const" | "continue"
            | "crate" | "do" | "dyn" | "else" | "enum" | "extern" | "false" | "final" | "fn" | "for" | "if"
            | "impl" | "in" | "let" | "loop" | "macro" | "match" | "mod" | "move" | "mut" | "override" | "priv"
            | "pub" | "ref" | "return" | "Self" | "self" | "static" | "struct" | "super" | "trait" | "true" | "try"
            | "type" | "typeof" | "unsafe" | "unsized" | "use" | "virtual" | "where" | "while" | "yield" => false,
            _ => true,
        }
    }
    impl Parse for Ident {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| {
                if let Some((ident, rest)) = cursor.ident() {
                    if accept_as_ident(&ident) {
                        Ok((ident, rest))
                    } else {
                        Err(cursor.error(format_args!("expected identifier, found keyword `{}`", ident,)))
                    }
                } else {
                    Err(cursor.error("expected identifier"))
                }
            })
        }
    }
    impl Token for Ident {
        fn peek(cursor: Cursor) -> bool {
            if let Some((ident, _rest)) = cursor.ident() {
                accept_as_ident(&ident)
            } else {
                false
            }
        }
        fn display() -> &'static str {
            "identifier"
        }
    }
}

mod parsing {
    impl Parse for Lifetime {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| cursor.lifetime().ok_or_else(|| cursor.error("expected lifetime")))
        }
    }
}

mod parsing {
    impl Parse for Macro {
        fn parse(input: ParseStream) -> Result<Self> {
            let tokens;
            Ok(Macro {
                path: input.call(Path::parse_mod_style)?,
                bang_token: input.parse()?,
                delimiter: {
                    let (delimiter, content) = mac_parse_delimiter(input)?;
                    tokens = content;
                    delimiter
                },
                tokens,
            })
        }
    }
}

mod parsing {
    fn parse_binop(input: ParseStream) -> Result<BinOp> {
        if input.peek(Token![&&]) {
            input.parse().map(BinOp::And)
        } else if input.peek(Token![||]) {
            input.parse().map(BinOp::Or)
        } else if input.peek(Token![<<]) {
            input.parse().map(BinOp::Shl)
        } else if input.peek(Token![>>]) {
            input.parse().map(BinOp::Shr)
        } else if input.peek(Token![==]) {
            input.parse().map(BinOp::Eq)
        } else if input.peek(Token![<=]) {
            input.parse().map(BinOp::Le)
        } else if input.peek(Token![!=]) {
            input.parse().map(BinOp::Ne)
        } else if input.peek(Token![>=]) {
            input.parse().map(BinOp::Ge)
        } else if input.peek(Token![+]) {
            input.parse().map(BinOp::Add)
        } else if input.peek(Token![-]) {
            input.parse().map(BinOp::Sub)
        } else if input.peek(Token![*]) {
            input.parse().map(BinOp::Mul)
        } else if input.peek(Token![/]) {
            input.parse().map(BinOp::Div)
        } else if input.peek(Token![%]) {
            input.parse().map(BinOp::Rem)
        } else if input.peek(Token![^]) {
            input.parse().map(BinOp::BitXor)
        } else if input.peek(Token![&]) {
            input.parse().map(BinOp::BitAnd)
        } else if input.peek(Token![|]) {
            input.parse().map(BinOp::BitOr)
        } else if input.peek(Token![<]) {
            input.parse().map(BinOp::Lt)
        } else if input.peek(Token![>]) {
            input.parse().map(BinOp::Gt)
        } else {
            Err(input.error("expected binary operator"))
        }
    }
    impl Parse for BinOp {
        #[cfg(not(feature = "full"))]
        fn parse(input: ParseStream) -> Result<Self> {
            parse_binop(input)
        }
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![+=]) {
                input.parse().map(BinOp::AddAssign)
            } else if input.peek(Token![-=]) {
                input.parse().map(BinOp::SubAssign)
            } else if input.peek(Token![*=]) {
                input.parse().map(BinOp::MulAssign)
            } else if input.peek(Token![/=]) {
                input.parse().map(BinOp::DivAssign)
            } else if input.peek(Token![%=]) {
                input.parse().map(BinOp::RemAssign)
            } else if input.peek(Token![^=]) {
                input.parse().map(BinOp::BitXorAssign)
            } else if input.peek(Token![&=]) {
                input.parse().map(BinOp::BitAndAssign)
            } else if input.peek(Token![|=]) {
                input.parse().map(BinOp::BitOrAssign)
            } else if input.peek(Token![<<=]) {
                input.parse().map(BinOp::ShlAssign)
            } else if input.peek(Token![>>=]) {
                input.parse().map(BinOp::ShrAssign)
            } else {
                parse_binop(input)
            }
        }
    }
    impl Parse for UnOp {
        fn parse(input: ParseStream) -> Result<Self> {
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![*]) {
                input.parse().map(UnOp::Deref)
            } else if lookahead.peek(Token![!]) {
                input.parse().map(UnOp::Not)
            } else if lookahead.peek(Token![-]) {
                input.parse().map(UnOp::Neg)
            } else {
                Err(lookahead.error())
            }
        }
    }
}

mod parsing {
    impl Parse for Visibility {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(token::Group) {
                let ahead = input.fork();
                let group = crate::group::parse_group(&ahead)?;
                if group.content.is_empty() {
                    input.advance_to(&ahead);
                    return Ok(Visibility::Inherited);
                }
            }
            if input.peek(Token![pub]) {
                Self::parse_pub(input)
            } else {
                Ok(Visibility::Inherited)
            }
        }
    }
    impl Visibility {
        fn parse_pub(input: ParseStream) -> Result<Self> {
            let pub_token = input.parse::<Token![pub]>()?;
            if input.peek(token::Paren) {
                let ahead = input.fork();
                let content;
                let paren_token = parenthesized!(content in ahead);
                if content.peek(Token![crate]) || content.peek(Token![self]) || content.peek(Token![super]) {
                    let path = content.call(Ident::parse_any)?;
                    if content.is_empty() {
                        input.advance_to(&ahead);
                        return Ok(Visibility::Restricted(VisRestricted {
                            pub_token,
                            paren_token,
                            in_token: None,
                            path: Box::new(Path::from(path)),
                        }));
                    }
                } else if content.peek(Token![in]) {
                    let in_token: Token![in] = content.parse()?;
                    let path = content.call(Path::parse_mod_style)?;
                    input.advance_to(&ahead);
                    return Ok(Visibility::Restricted(VisRestricted {
                        pub_token,
                        paren_token,
                        in_token: Some(in_token),
                        path: Box::new(path),
                    }));
                }
            }
            Ok(Visibility::Public(pub_token))
        }
        pub fn is_some(&self) -> bool {
            match self {
                Visibility::Inherited => false,
                _ => true,
            }
        }
    }
}

pub(crate) mod parsing {
    impl Parse for Lit {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| {
                if let Some((lit, rest)) = cursor.literal() {
                    return Ok((Lit::new(lit), rest));
                }
                if let Some((ident, rest)) = cursor.ident() {
                    let value = ident == "true";
                    if value || ident == "false" {
                        let lit_bool = LitBool {
                            value,
                            span: ident.span(),
                        };
                        return Ok((Lit::Bool(lit_bool), rest));
                    }
                }
                if let Some((punct, rest)) = cursor.punct() {
                    if punct.as_char() == '-' {
                        if let Some((lit, rest)) = parse_negative_lit(punct, rest) {
                            return Ok((lit, rest));
                        }
                    }
                }
                Err(cursor.error("expected literal"))
            })
        }
    }
    fn parse_negative_lit(neg: Punct, cursor: Cursor) -> Option<(Lit, Cursor)> {
        let (lit, rest) = cursor.literal()?;
        let mut span = neg.span();
        span = span.join(lit.span()).unwrap_or(span);
        let mut repr = lit.to_string();
        repr.insert(0, '-');
        if let Some((digits, suffix)) = value::parse_lit_int(&repr) {
            let mut token: Literal = repr.parse().unwrap();
            token.set_span(span);
            return Some((
                Lit::Int(LitInt {
                    repr: Box::new(LitIntRepr { token, digits, suffix }),
                }),
                rest,
            ));
        }
        let (digits, suffix) = value::parse_lit_float(&repr)?;
        let mut token: Literal = repr.parse().unwrap();
        token.set_span(span);
        Some((
            Lit::Float(LitFloat {
                repr: Box::new(LitFloatRepr { token, digits, suffix }),
            }),
            rest,
        ))
    }
    impl Parse for LitStr {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Str(lit)) => Ok(lit),
                _ => Err(head.error("expected string literal")),
            }
        }
    }
    impl Parse for LitByteStr {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::ByteStr(lit)) => Ok(lit),
                _ => Err(head.error("expected byte string literal")),
            }
        }
    }
    impl Parse for LitByte {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Byte(lit)) => Ok(lit),
                _ => Err(head.error("expected byte literal")),
            }
        }
    }
    impl Parse for LitChar {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Char(lit)) => Ok(lit),
                _ => Err(head.error("expected character literal")),
            }
        }
    }
    impl Parse for LitInt {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Int(lit)) => Ok(lit),
                _ => Err(head.error("expected integer literal")),
            }
        }
    }
    impl Parse for LitFloat {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Float(lit)) => Ok(lit),
                _ => Err(head.error("expected floating point literal")),
            }
        }
    }
    impl Parse for LitBool {
        fn parse(input: ParseStream) -> Result<Self> {
            let head = input.fork();
            match input.parse() {
                Ok(Lit::Bool(lit)) => Ok(lit),
                _ => Err(head.error("expected boolean literal")),
            }
        }
    }
}

pub(crate) mod parsing {
    impl Pat {
        pub fn parse_single(input: ParseStream) -> Result<Self> {
            let begin = input.fork();
            let lookahead = input.lookahead1();
            if lookahead.peek(Ident)
                && (input.peek2(Token![::])
                    || input.peek2(Token![!])
                    || input.peek2(token::Brace)
                    || input.peek2(token::Paren)
                    || input.peek2(Token![..]))
                || input.peek(Token![self]) && input.peek2(Token![::])
                || lookahead.peek(Token![::])
                || lookahead.peek(Token![<])
                || input.peek(Token![Self])
                || input.peek(Token![super])
                || input.peek(Token![crate])
            {
                pat_path_or_macro_or_struct_or_range(input)
            } else if lookahead.peek(Token![_]) {
                input.call(pat_wild).map(Pat::Wild)
            } else if input.peek(Token![box]) {
                pat_box(begin, input)
            } else if input.peek(Token![-]) || lookahead.peek(Lit) || lookahead.peek(Token![const]) {
                pat_lit_or_range(input)
            } else if lookahead.peek(Token![ref])
                || lookahead.peek(Token![mut])
                || input.peek(Token![self])
                || input.peek(Ident)
            {
                input.call(pat_ident).map(Pat::Ident)
            } else if lookahead.peek(Token![&]) {
                input.call(pat_reference).map(Pat::Reference)
            } else if lookahead.peek(token::Paren) {
                input.call(pat_paren_or_tuple)
            } else if lookahead.peek(token::Bracket) {
                input.call(pat_slice).map(Pat::Slice)
            } else if lookahead.peek(Token![..]) && !input.peek(Token![...]) {
                pat_range_half_open(input)
            } else if lookahead.peek(Token![const]) {
                input.call(pat_const).map(Pat::Verbatim)
            } else {
                Err(lookahead.error())
            }
        }
        pub fn parse_multi(input: ParseStream) -> Result<Self> {
            multi_pat_impl(input, None)
        }
        pub fn parse_multi_with_leading_vert(input: ParseStream) -> Result<Self> {
            let leading_vert: Option<Token![|]> = input.parse()?;
            multi_pat_impl(input, leading_vert)
        }
    }
    fn multi_pat_impl(input: ParseStream, leading_vert: Option<Token![|]>) -> Result<Pat> {
        let mut pat = Pat::parse_single(input)?;
        if leading_vert.is_some() || input.peek(Token![|]) && !input.peek(Token![||]) && !input.peek(Token![|=]) {
            let mut cases = Punctuated::new();
            cases.push_value(pat);
            while input.peek(Token![|]) && !input.peek(Token![||]) && !input.peek(Token![|=]) {
                let punct = input.parse()?;
                cases.push_punct(punct);
                let pat = Pat::parse_single(input)?;
                cases.push_value(pat);
            }
            pat = Pat::Or(PatOr {
                attrs: Vec::new(),
                leading_vert,
                cases,
            });
        }
        Ok(pat)
    }
    fn pat_path_or_macro_or_struct_or_range(input: ParseStream) -> Result<Pat> {
        let (qself, path) = qpath(input, true)?;
        if qself.is_none() && input.peek(Token![!]) && !input.peek(Token![!=]) && path.is_mod_style() {
            let bang_token: Token![!] = input.parse()?;
            let (delimiter, tokens) = mac_parse_delimiter(input)?;
            return Ok(Pat::Macro(ExprMacro {
                attrs: Vec::new(),
                mac: Macro {
                    path,
                    bang_token,
                    delimiter,
                    tokens,
                },
            }));
        }
        if input.peek(token::Brace) {
            pat_struct(input, qself, path).map(Pat::Struct)
        } else if input.peek(token::Paren) {
            pat_tuple_struct(input, qself, path).map(Pat::TupleStruct)
        } else if input.peek(Token![..]) {
            pat_range(input, qself, path)
        } else {
            Ok(Pat::Path(ExprPath {
                attrs: Vec::new(),
                qself,
                path,
            }))
        }
    }
    fn pat_wild(input: ParseStream) -> Result<PatWild> {
        Ok(PatWild {
            attrs: Vec::new(),
            underscore_token: input.parse()?,
        })
    }
    fn pat_box(begin: ParseBuffer, input: ParseStream) -> Result<Pat> {
        input.parse::<Token![box]>()?;
        Pat::parse_single(input)?;
        Ok(Pat::Verbatim(verbatim_between(&begin, input)))
    }
    fn pat_ident(input: ParseStream) -> Result<PatIdent> {
        Ok(PatIdent {
            attrs: Vec::new(),
            by_ref: input.parse()?,
            mutability: input.parse()?,
            ident: input.call(Ident::parse_any)?,
            subpat: {
                if input.peek(Token![@]) {
                    let at_token: Token![@] = input.parse()?;
                    let subpat = Pat::parse_single(input)?;
                    Some((at_token, Box::new(subpat)))
                } else {
                    None
                }
            },
        })
    }
    fn pat_tuple_struct(input: ParseStream, qself: Option<QSelf>, path: Path) -> Result<PatTupleStruct> {
        let content;
        let paren_token = parenthesized!(content in input);
        let mut elems = Punctuated::new();
        while !content.is_empty() {
            let value = Pat::parse_multi_with_leading_vert(&content)?;
            elems.push_value(value);
            if content.is_empty() {
                break;
            }
            let punct = content.parse()?;
            elems.push_punct(punct);
        }
        Ok(PatTupleStruct {
            attrs: Vec::new(),
            qself,
            path,
            paren_token,
            elems,
        })
    }
    fn pat_struct(input: ParseStream, qself: Option<QSelf>, path: Path) -> Result<PatStruct> {
        let content;
        let brace_token = braced!(content in input);
        let mut fields = Punctuated::new();
        let mut rest = None;
        while !content.is_empty() {
            let attrs = content.call(Attribute::parse_outer)?;
            if content.peek(Token![..]) {
                rest = Some(PatRest {
                    attrs,
                    dot2_token: content.parse()?,
                });
                break;
            }
            let mut value = content.call(field_pat)?;
            value.attrs = attrs;
            fields.push_value(value);
            if content.is_empty() {
                break;
            }
            let punct: Token![,] = content.parse()?;
            fields.push_punct(punct);
        }
        Ok(PatStruct {
            attrs: Vec::new(),
            qself,
            path,
            brace_token,
            fields,
            rest,
        })
    }
    impl Member {
        fn is_unnamed(&self) -> bool {
            match self {
                Member::Named(_) => false,
                Member::Unnamed(_) => true,
            }
        }
    }
    fn field_pat(input: ParseStream) -> Result<FieldPat> {
        let begin = input.fork();
        let boxed: Option<Token![box]> = input.parse()?;
        let by_ref: Option<Token![ref]> = input.parse()?;
        let mutability: Option<Token![mut]> = input.parse()?;
        let member = if boxed.is_some() || by_ref.is_some() || mutability.is_some() {
            input.parse().map(Member::Named)
        } else {
            input.parse()
        }?;
        if boxed.is_none() && by_ref.is_none() && mutability.is_none() && input.peek(Token![:]) || member.is_unnamed() {
            return Ok(FieldPat {
                attrs: Vec::new(),
                member,
                colon_token: Some(input.parse()?),
                pat: Box::new(Pat::parse_multi_with_leading_vert(input)?),
            });
        }
        let ident = match member {
            Member::Named(ident) => ident,
            Member::Unnamed(_) => unreachable!(),
        };
        let pat = if boxed.is_some() {
            Pat::Verbatim(verbatim_between(&begin, input))
        } else {
            Pat::Ident(PatIdent {
                attrs: Vec::new(),
                by_ref,
                mutability,
                ident: ident.clone(),
                subpat: None,
            })
        };
        Ok(FieldPat {
            attrs: Vec::new(),
            member: Member::Named(ident),
            colon_token: None,
            pat: Box::new(pat),
        })
    }
    fn pat_range(input: ParseStream, qself: Option<QSelf>, path: Path) -> Result<Pat> {
        let limits = RangeLimits::parse_obsolete(input)?;
        let end = input.call(pat_range_bound)?;
        if let (RangeLimits::Closed(_), None) = (&limits, &end) {
            return Err(input.error("expected range upper bound"));
        }
        Ok(Pat::Range(ExprRange {
            attrs: Vec::new(),
            start: Some(Box::new(Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself,
                path,
            }))),
            limits,
            end: end.map(PatRangeBound::into_expr),
        }))
    }
    fn pat_range_half_open(input: ParseStream) -> Result<Pat> {
        let limits: RangeLimits = input.parse()?;
        let end = input.call(pat_range_bound)?;
        if end.is_some() {
            Ok(Pat::Range(ExprRange {
                attrs: Vec::new(),
                start: None,
                limits,
                end: end.map(PatRangeBound::into_expr),
            }))
        } else {
            match limits {
                RangeLimits::HalfOpen(dot2_token) => Ok(Pat::Rest(PatRest {
                    attrs: Vec::new(),
                    dot2_token,
                })),
                RangeLimits::Closed(_) => Err(input.error("expected range upper bound")),
            }
        }
    }
    fn pat_paren_or_tuple(input: ParseStream) -> Result<Pat> {
        let content;
        let paren_token = parenthesized!(content in input);
        let mut elems = Punctuated::new();
        while !content.is_empty() {
            let value = Pat::parse_multi_with_leading_vert(&content)?;
            if content.is_empty() {
                if elems.is_empty() && !matches!(value, Pat::Rest(_)) {
                    return Ok(Pat::Paren(PatParen {
                        attrs: Vec::new(),
                        paren_token,
                        pat: Box::new(value),
                    }));
                }
                elems.push_value(value);
                break;
            }
            elems.push_value(value);
            let punct = content.parse()?;
            elems.push_punct(punct);
        }
        Ok(Pat::Tuple(PatTuple {
            attrs: Vec::new(),
            paren_token,
            elems,
        }))
    }
    fn pat_reference(input: ParseStream) -> Result<PatReference> {
        Ok(PatReference {
            attrs: Vec::new(),
            and_token: input.parse()?,
            mutability: input.parse()?,
            pat: Box::new(Pat::parse_single(input)?),
        })
    }
    fn pat_lit_or_range(input: ParseStream) -> Result<Pat> {
        let start = input.call(pat_range_bound)?.unwrap();
        if input.peek(Token![..]) {
            let limits = RangeLimits::parse_obsolete(input)?;
            let end = input.call(pat_range_bound)?;
            if let (RangeLimits::Closed(_), None) = (&limits, &end) {
                return Err(input.error("expected range upper bound"));
            }
            Ok(Pat::Range(ExprRange {
                attrs: Vec::new(),
                start: Some(start.into_expr()),
                limits,
                end: end.map(PatRangeBound::into_expr),
            }))
        } else {
            Ok(start.into_pat())
        }
    }
    enum PatRangeBound {
        Const(ExprConst),
        Lit(ExprLit),
        Path(ExprPath),
    }
    impl PatRangeBound {
        fn into_expr(self) -> Box<Expr> {
            Box::new(match self {
                PatRangeBound::Const(pat) => Expr::Const(pat),
                PatRangeBound::Lit(pat) => Expr::Lit(pat),
                PatRangeBound::Path(pat) => Expr::Path(pat),
            })
        }
        fn into_pat(self) -> Pat {
            match self {
                PatRangeBound::Const(pat) => Pat::Const(pat),
                PatRangeBound::Lit(pat) => Pat::Lit(pat),
                PatRangeBound::Path(pat) => Pat::Path(pat),
            }
        }
    }
    fn pat_range_bound(input: ParseStream) -> Result<Option<PatRangeBound>> {
        if input.is_empty()
            || input.peek(Token![|])
            || input.peek(Token![=])
            || input.peek(Token![:]) && !input.peek(Token![::])
            || input.peek(Token![,])
            || input.peek(Token![;])
            || input.peek(Token![if])
        {
            return Ok(None);
        }
        let lookahead = input.lookahead1();
        let expr = if lookahead.peek(Lit) {
            PatRangeBound::Lit(input.parse()?)
        } else if lookahead.peek(Ident)
            || lookahead.peek(Token![::])
            || lookahead.peek(Token![<])
            || lookahead.peek(Token![self])
            || lookahead.peek(Token![Self])
            || lookahead.peek(Token![super])
            || lookahead.peek(Token![crate])
        {
            PatRangeBound::Path(input.parse()?)
        } else if lookahead.peek(Token![const]) {
            PatRangeBound::Const(input.parse()?)
        } else {
            return Err(lookahead.error());
        };
        Ok(Some(expr))
    }
    fn pat_slice(input: ParseStream) -> Result<PatSlice> {
        let content;
        let bracket_token = bracketed!(content in input);
        let mut elems = Punctuated::new();
        while !content.is_empty() {
            let value = Pat::parse_multi_with_leading_vert(&content)?;
            match value {
                Pat::Range(pat) if pat.start.is_none() || pat.end.is_none() => {
                    let (start, end) = match pat.limits {
                        RangeLimits::HalfOpen(dot_dot) => (dot_dot.spans[0], dot_dot.spans[1]),
                        RangeLimits::Closed(dot_dot_eq) => (dot_dot_eq.spans[0], dot_dot_eq.spans[2]),
                    };
                    let msg = "range pattern is not allowed unparenthesized inside slice pattern";
                    return Err(err::new2(start, end, msg));
                },
                _ => {},
            }
            elems.push_value(value);
            if content.is_empty() {
                break;
            }
            let punct = content.parse()?;
            elems.push_punct(punct);
        }
        Ok(PatSlice {
            attrs: Vec::new(),
            bracket_token,
            elems,
        })
    }
    fn pat_const(input: ParseStream) -> Result<TokenStream> {
        let begin = input.fork();
        input.parse::<Token![const]>()?;
        let content;
        braced!(content in input);
        content.call(Attribute::parse_inner)?;
        content.call(Block::parse_within)?;
        Ok(verbatim_between(&begin, input))
    }
}

pub(crate) mod parsing {
    impl Parse for Path {
        fn parse(input: ParseStream) -> Result<Self> {
            Self::parse_helper(input, false)
        }
    }
    impl Parse for GenericArgument {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Lifetime) && !input.peek2(Token![+]) {
                return Ok(GenericArgument::Lifetime(input.parse()?));
            }
            if input.peek(Lit) || input.peek(token::Brace) {
                return const_argument(input).map(GenericArgument::Const);
            }
            let mut argument: Type = input.parse()?;
            match argument {
                Type::Path(mut ty)
                    if ty.qself.is_none()
                        && ty.path.leading_colon.is_none()
                        && ty.path.segments.len() == 1
                        && match &ty.path.segments[0].arguments {
                            PathArguments::None | PathArguments::AngleBracketed(_) => true,
                            PathArguments::Parenthesized(_) => false,
                        } =>
                {
                    if let Some(eq_token) = input.parse::<Option<Token![=]>>()? {
                        let segment = ty.path.segments.pop().unwrap().into_value();
                        let ident = segment.ident;
                        let generics = match segment.arguments {
                            PathArguments::None => None,
                            PathArguments::AngleBracketed(arguments) => Some(arguments),
                            PathArguments::Parenthesized(_) => unreachable!(),
                        };
                        return if input.peek(Lit) || input.peek(token::Brace) {
                            Ok(GenericArgument::AssocConst(AssocConst {
                                ident,
                                generics,
                                eq_token,
                                value: const_argument(input)?,
                            }))
                        } else {
                            Ok(GenericArgument::AssocType(AssocType {
                                ident,
                                generics,
                                eq_token,
                                ty: input.parse()?,
                            }))
                        };
                    }
                    if let Some(colon_token) = input.parse::<Option<Token![:]>>()? {
                        let segment = ty.path.segments.pop().unwrap().into_value();
                        return Ok(GenericArgument::Constraint(Constraint {
                            ident: segment.ident,
                            generics: match segment.arguments {
                                PathArguments::None => None,
                                PathArguments::AngleBracketed(arguments) => Some(arguments),
                                PathArguments::Parenthesized(_) => unreachable!(),
                            },
                            colon_token,
                            bounds: {
                                let mut bounds = Punctuated::new();
                                loop {
                                    if input.peek(Token![,]) || input.peek(Token![>]) {
                                        break;
                                    }
                                    let value: TypeParamBound = input.parse()?;
                                    bounds.push_value(value);
                                    if !input.peek(Token![+]) {
                                        break;
                                    }
                                    let punct: Token![+] = input.parse()?;
                                    bounds.push_punct(punct);
                                }
                                bounds
                            },
                        }));
                    }
                    argument = Type::Path(ty);
                },
                _ => {},
            }
            Ok(GenericArgument::Type(argument))
        }
    }
    pub(crate) fn const_argument(input: ParseStream) -> Result<Expr> {
        let lookahead = input.lookahead1();
        if input.peek(Lit) {
            let lit = input.parse()?;
            return Ok(Expr::Lit(lit));
        }
        if input.peek(Ident) {
            let ident: Ident = input.parse()?;
            return Ok(Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: Path::from(ident),
            }));
        }
        if input.peek(token::Brace) {
            {
                let block: ExprBlock = input.parse()?;
                return Ok(Expr::Block(block));
            }
            #[cfg(not(feature = "full"))]
            {
                let begin = input.fork();
                let content;
                braced!(content in input);
                content.parse::<Expr>()?;
                let verbatim = verbatim_between(&begin, input);
                return Ok(Expr::Verbatim(verbatim));
            }
        }
        Err(lookahead.error())
    }
    impl AngleBracketedGenericArguments {
        pub fn parse_turbofish(input: ParseStream) -> Result<Self> {
            let colon2_token: Token![::] = input.parse()?;
            Self::do_parse(Some(colon2_token), input)
        }
        fn do_parse(colon2_token: Option<Token![::]>, input: ParseStream) -> Result<Self> {
            Ok(AngleBracketedGenericArguments {
                colon2_token,
                lt_token: input.parse()?,
                args: {
                    let mut args = Punctuated::new();
                    loop {
                        if input.peek(Token![>]) {
                            break;
                        }
                        let value: GenericArgument = input.parse()?;
                        args.push_value(value);
                        if input.peek(Token![>]) {
                            break;
                        }
                        let punct: Token![,] = input.parse()?;
                        args.push_punct(punct);
                    }
                    args
                },
                gt_token: input.parse()?,
            })
        }
    }
    impl Parse for AngleBracketedGenericArguments {
        fn parse(input: ParseStream) -> Result<Self> {
            let colon2_token: Option<Token![::]> = input.parse()?;
            Self::do_parse(colon2_token, input)
        }
    }
    impl Parse for ParenthesizedGenericArguments {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(ParenthesizedGenericArguments {
                paren_token: parenthesized!(content in input),
                inputs: content.parse_terminated(Type::parse, Token![,])?,
                output: input.call(ReturnType::without_plus)?,
            })
        }
    }
    impl Parse for PathSegment {
        fn parse(input: ParseStream) -> Result<Self> {
            Self::parse_helper(input, false)
        }
    }
    impl PathSegment {
        fn parse_helper(input: ParseStream, expr_style: bool) -> Result<Self> {
            if input.peek(Token![super])
                || input.peek(Token![self])
                || input.peek(Token![crate])
                || cfg!(feature = "full") && input.peek(Token![try])
            {
                let ident = input.call(Ident::parse_any)?;
                return Ok(PathSegment::from(ident));
            }
            let ident = if input.peek(Token![Self]) {
                input.call(Ident::parse_any)?
            } else {
                input.parse()?
            };
            if !expr_style && input.peek(Token![<]) && !input.peek(Token![<=])
                || input.peek(Token![::]) && input.peek3(Token![<])
            {
                Ok(PathSegment {
                    ident,
                    arguments: PathArguments::AngleBracketed(input.parse()?),
                })
            } else {
                Ok(PathSegment::from(ident))
            }
        }
    }
    impl Path {
        pub fn parse_mod_style(input: ParseStream) -> Result<Self> {
            Ok(Path {
                leading_colon: input.parse()?,
                segments: {
                    let mut segments = Punctuated::new();
                    loop {
                        if !input.peek(Ident)
                            && !input.peek(Token![super])
                            && !input.peek(Token![self])
                            && !input.peek(Token![Self])
                            && !input.peek(Token![crate])
                        {
                            break;
                        }
                        let ident = Ident::parse_any(input)?;
                        segments.push_value(PathSegment::from(ident));
                        if !input.peek(Token![::]) {
                            break;
                        }
                        let punct = input.parse()?;
                        segments.push_punct(punct);
                    }
                    if segments.is_empty() {
                        return Err(input.parse::<Ident>().unwrap_err());
                    } else if segments.trailing_punct() {
                        return Err(input.error("expected path segment after `::`"));
                    }
                    segments
                },
            })
        }
        pub(crate) fn parse_helper(input: ParseStream, expr_style: bool) -> Result<Self> {
            let mut path = Path {
                leading_colon: input.parse()?,
                segments: {
                    let mut segments = Punctuated::new();
                    let value = PathSegment::parse_helper(input, expr_style)?;
                    segments.push_value(value);
                    segments
                },
            };
            Path::parse_rest(input, &mut path, expr_style)?;
            Ok(path)
        }
        pub(crate) fn parse_rest(input: ParseStream, path: &mut Self, expr_style: bool) -> Result<()> {
            while input.peek(Token![::]) && !input.peek3(token::Paren) {
                let punct: Token![::] = input.parse()?;
                path.segments.push_punct(punct);
                let value = PathSegment::parse_helper(input, expr_style)?;
                path.segments.push_value(value);
            }
            Ok(())
        }
        pub(crate) fn is_mod_style(&self) -> bool {
            self.segments.iter().all(|segment| segment.arguments.is_none())
        }
    }
    pub(crate) fn qpath(input: ParseStream, expr_style: bool) -> Result<(Option<QSelf>, Path)> {
        if input.peek(Token![<]) {
            let lt_token: Token![<] = input.parse()?;
            let this: Type = input.parse()?;
            let path = if input.peek(Token![as]) {
                let as_token: Token![as] = input.parse()?;
                let path: Path = input.parse()?;
                Some((as_token, path))
            } else {
                None
            };
            let gt_token: Token![>] = input.parse()?;
            let colon2_token: Token![::] = input.parse()?;
            let mut rest = Punctuated::new();
            loop {
                let path = PathSegment::parse_helper(input, expr_style)?;
                rest.push_value(path);
                if !input.peek(Token![::]) {
                    break;
                }
                let punct: Token![::] = input.parse()?;
                rest.push_punct(punct);
            }
            let (position, as_token, path) = match path {
                Some((as_token, mut path)) => {
                    let pos = path.segments.len();
                    path.segments.push_punct(colon2_token);
                    path.segments.extend(rest.into_pairs());
                    (pos, Some(as_token), path)
                },
                None => {
                    let path = Path {
                        leading_colon: Some(colon2_token),
                        segments: rest,
                    };
                    (0, None, path)
                },
            };
            let qself = QSelf {
                lt_token,
                ty: Box::new(this),
                position,
                as_token,
                gt_token,
            };
            Ok((Some(qself), path))
        } else {
            let path = Path::parse_helper(input, expr_style)?;
            Ok((None, path))
        }
    }
}

pub mod parsing {
    pub fn keyword(input: ParseStream, token: &str) -> Result<Span> {
        input.step(|cursor| {
            if let Some((ident, rest)) = cursor.ident() {
                if ident == token {
                    return Ok((ident.span(), rest));
                }
            }
            Err(cursor.error(format!("expected `{}`", token)))
        })
    }
    pub fn peek_keyword(cursor: Cursor, token: &str) -> bool {
        if let Some((ident, _rest)) = cursor.ident() {
            ident == token
        } else {
            false
        }
    }
    pub fn punct<const N: usize>(input: ParseStream, token: &str) -> Result<[Span; N]> {
        let mut spans = [input.span(); N];
        punct_helper(input, token, &mut spans)?;
        Ok(spans)
    }
    fn punct_helper(input: ParseStream, token: &str, spans: &mut [Span]) -> Result<()> {
        input.step(|cursor| {
            let mut cursor = *cursor;
            assert_eq!(token.len(), spans.len());
            for (i, ch) in token.chars().enumerate() {
                match cursor.punct() {
                    Some((punct, rest)) => {
                        spans[i] = punct.span();
                        if punct.as_char() != ch {
                            break;
                        } else if i == token.len() - 1 {
                            return Ok(((), rest));
                        } else if punct.spacing() != Spacing::Joint {
                            break;
                        }
                        cursor = rest;
                    },
                    None => break,
                }
            }
            Err(Err::new(spans[0], format!("expected `{}`", token)))
        })
    }
    pub fn peek_punct(mut cursor: Cursor, token: &str) -> bool {
        for (i, ch) in token.chars().enumerate() {
            match cursor.punct() {
                Some((punct, rest)) => {
                    if punct.as_char() != ch {
                        break;
                    } else if i == token.len() - 1 {
                        return true;
                    } else if punct.spacing() != Spacing::Joint {
                        break;
                    }
                    cursor = rest;
                },
                None => break,
            }
        }
        false
    }
}

pub(crate) mod parsing {
    impl Parse for Type {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_plus = true;
            let allow_group_generic = true;
            ambig_ty(input, allow_plus, allow_group_generic)
        }
    }
    impl Type {
        pub fn without_plus(input: ParseStream) -> Result<Self> {
            let allow_plus = false;
            let allow_group_generic = true;
            ambig_ty(input, allow_plus, allow_group_generic)
        }
    }
    pub(crate) fn ambig_ty(input: ParseStream, allow_plus: bool, allow_group_generic: bool) -> Result<Type> {
        let begin = input.fork();
        if input.peek(token::Group) {
            let mut group: TypeGroup = input.parse()?;
            if input.peek(Token![::]) && input.peek3(Ident::peek_any) {
                if let Type::Path(mut ty) = *group.elem {
                    Path::parse_rest(input, &mut ty.path, false)?;
                    return Ok(Type::Path(ty));
                } else {
                    return Ok(Type::Path(TypePath {
                        qself: Some(QSelf {
                            lt_token: Token![<](group.group_token.span),
                            position: 0,
                            as_token: None,
                            gt_token: Token![>](group.group_token.span),
                            ty: group.elem,
                        }),
                        path: Path::parse_helper(input, false)?,
                    }));
                }
            } else if input.peek(Token![<]) && allow_group_generic || input.peek(Token![::]) && input.peek3(Token![<]) {
                if let Type::Path(mut ty) = *group.elem {
                    let arguments = &mut ty.path.segments.last_mut().unwrap().arguments;
                    if arguments.is_none() {
                        *arguments = PathArguments::AngleBracketed(input.parse()?);
                        Path::parse_rest(input, &mut ty.path, false)?;
                        return Ok(Type::Path(ty));
                    } else {
                        group.elem = Box::new(Type::Path(ty));
                    }
                }
            }
            return Ok(Type::Group(group));
        }
        let mut lifetimes = None::<BoundLifetimes>;
        let mut lookahead = input.lookahead1();
        if lookahead.peek(Token![for]) {
            lifetimes = input.parse()?;
            lookahead = input.lookahead1();
            if !lookahead.peek(Ident)
                && !lookahead.peek(Token![fn])
                && !lookahead.peek(Token![unsafe])
                && !lookahead.peek(Token![extern])
                && !lookahead.peek(Token![super])
                && !lookahead.peek(Token![self])
                && !lookahead.peek(Token![Self])
                && !lookahead.peek(Token![crate])
                || input.peek(Token![dyn])
            {
                return Err(lookahead.error());
            }
        }
        if lookahead.peek(token::Paren) {
            let content;
            let paren_token = parenthesized!(content in input);
            if content.is_empty() {
                return Ok(Type::Tuple(TypeTuple {
                    paren_token,
                    elems: Punctuated::new(),
                }));
            }
            if content.peek(Lifetime) {
                return Ok(Type::Paren(TypeParen {
                    paren_token,
                    elem: Box::new(Type::TraitObject(content.parse()?)),
                }));
            }
            if content.peek(Token![?]) {
                return Ok(Type::TraitObject(TypeTraitObject {
                    dyn_token: None,
                    bounds: {
                        let mut bounds = Punctuated::new();
                        bounds.push_value(TypeParamBound::Trait(TraitBound {
                            paren_token: Some(paren_token),
                            ..content.parse()?
                        }));
                        while let Some(plus) = input.parse()? {
                            bounds.push_punct(plus);
                            bounds.push_value(input.parse()?);
                        }
                        bounds
                    },
                }));
            }
            let mut first: Type = content.parse()?;
            if content.peek(Token![,]) {
                return Ok(Type::Tuple(TypeTuple {
                    paren_token,
                    elems: {
                        let mut elems = Punctuated::new();
                        elems.push_value(first);
                        elems.push_punct(content.parse()?);
                        while !content.is_empty() {
                            elems.push_value(content.parse()?);
                            if content.is_empty() {
                                break;
                            }
                            elems.push_punct(content.parse()?);
                        }
                        elems
                    },
                }));
            }
            if allow_plus && input.peek(Token![+]) {
                loop {
                    let first = match first {
                        Type::Path(TypePath { qself: None, path }) => TypeParamBound::Trait(TraitBound {
                            paren_token: Some(paren_token),
                            modifier: TraitBoundModifier::None,
                            lifetimes: None,
                            path,
                        }),
                        Type::TraitObject(TypeTraitObject {
                            dyn_token: None,
                            bounds,
                        }) => {
                            if bounds.len() > 1 || bounds.trailing_punct() {
                                first = Type::TraitObject(TypeTraitObject {
                                    dyn_token: None,
                                    bounds,
                                });
                                break;
                            }
                            match bounds.into_iter().next().unwrap() {
                                TypeParamBound::Trait(trait_bound) => TypeParamBound::Trait(TraitBound {
                                    paren_token: Some(paren_token),
                                    ..trait_bound
                                }),
                                other @ (TypeParamBound::Lifetime(_) | TypeParamBound::Verbatim(_)) => other,
                            }
                        },
                        _ => break,
                    };
                    return Ok(Type::TraitObject(TypeTraitObject {
                        dyn_token: None,
                        bounds: {
                            let mut bounds = Punctuated::new();
                            bounds.push_value(first);
                            while let Some(plus) = input.parse()? {
                                bounds.push_punct(plus);
                                bounds.push_value(input.parse()?);
                            }
                            bounds
                        },
                    }));
                }
            }
            Ok(Type::Paren(TypeParen {
                paren_token,
                elem: Box::new(first),
            }))
        } else if lookahead.peek(Token![fn]) || lookahead.peek(Token![unsafe]) || lookahead.peek(Token![extern]) {
            let mut bare_fn: TypeBareFn = input.parse()?;
            bare_fn.lifetimes = lifetimes;
            Ok(Type::BareFn(bare_fn))
        } else if lookahead.peek(Ident)
            || input.peek(Token![super])
            || input.peek(Token![self])
            || input.peek(Token![Self])
            || input.peek(Token![crate])
            || lookahead.peek(Token![::])
            || lookahead.peek(Token![<])
        {
            let ty: TypePath = input.parse()?;
            if ty.qself.is_some() {
                return Ok(Type::Path(ty));
            }
            if input.peek(Token![!]) && !input.peek(Token![!=]) && ty.path.is_mod_style() {
                let bang_token: Token![!] = input.parse()?;
                let (delimiter, tokens) = mac_parse_delimiter(input)?;
                return Ok(Type::Macro(TypeMacro {
                    mac: Macro {
                        path: ty.path,
                        bang_token,
                        delimiter,
                        tokens,
                    },
                }));
            }
            if lifetimes.is_some() || allow_plus && input.peek(Token![+]) {
                let mut bounds = Punctuated::new();
                bounds.push_value(TypeParamBound::Trait(TraitBound {
                    paren_token: None,
                    modifier: TraitBoundModifier::None,
                    lifetimes,
                    path: ty.path,
                }));
                if allow_plus {
                    while input.peek(Token![+]) {
                        bounds.push_punct(input.parse()?);
                        if !(input.peek(Ident::peek_any)
                            || input.peek(Token![::])
                            || input.peek(Token![?])
                            || input.peek(Lifetime)
                            || input.peek(token::Paren))
                        {
                            break;
                        }
                        bounds.push_value(input.parse()?);
                    }
                }
                return Ok(Type::TraitObject(TypeTraitObject {
                    dyn_token: None,
                    bounds,
                }));
            }
            Ok(Type::Path(ty))
        } else if lookahead.peek(Token![dyn]) {
            let dyn_token: Token![dyn] = input.parse()?;
            let dyn_span = dyn_token.span;
            let star_token: Option<Token![*]> = input.parse()?;
            let bounds = TypeTraitObject::parse_bounds(dyn_span, input, allow_plus)?;
            return Ok(if star_token.is_some() {
                Type::Verbatim(verbatim_between(&begin, input))
            } else {
                Type::TraitObject(TypeTraitObject {
                    dyn_token: Some(dyn_token),
                    bounds,
                })
            });
        } else if lookahead.peek(token::Bracket) {
            let content;
            let bracket_token = bracketed!(content in input);
            let elem: Type = content.parse()?;
            if content.peek(Token![;]) {
                Ok(Type::Array(TypeArray {
                    bracket_token,
                    elem: Box::new(elem),
                    semi_token: content.parse()?,
                    len: content.parse()?,
                }))
            } else {
                Ok(Type::Slice(TypeSlice {
                    bracket_token,
                    elem: Box::new(elem),
                }))
            }
        } else if lookahead.peek(Token![*]) {
            input.parse().map(Type::Ptr)
        } else if lookahead.peek(Token![&]) {
            input.parse().map(Type::Reference)
        } else if lookahead.peek(Token![!]) && !input.peek(Token![=]) {
            input.parse().map(Type::Never)
        } else if lookahead.peek(Token![impl]) {
            TypeImplTrait::parse(input, allow_plus).map(Type::ImplTrait)
        } else if lookahead.peek(Token![_]) {
            input.parse().map(Type::Infer)
        } else if lookahead.peek(Lifetime) {
            input.parse().map(Type::TraitObject)
        } else {
            Err(lookahead.error())
        }
    }
    impl Parse for TypeSlice {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(TypeSlice {
                bracket_token: bracketed!(content in input),
                elem: content.parse()?,
            })
        }
    }
    impl Parse for TypeArray {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(TypeArray {
                bracket_token: bracketed!(content in input),
                elem: content.parse()?,
                semi_token: content.parse()?,
                len: content.parse()?,
            })
        }
    }
    impl Parse for TypePtr {
        fn parse(input: ParseStream) -> Result<Self> {
            let star_token: Token![*] = input.parse()?;
            let lookahead = input.lookahead1();
            let (const_token, mutability) = if lookahead.peek(Token![const]) {
                (Some(input.parse()?), None)
            } else if lookahead.peek(Token![mut]) {
                (None, Some(input.parse()?))
            } else {
                return Err(lookahead.error());
            };
            Ok(TypePtr {
                star_token,
                const_token,
                mutability,
                elem: Box::new(input.call(Type::without_plus)?),
            })
        }
    }
    impl Parse for TypeReference {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(TypeReference {
                and_token: input.parse()?,
                lifetime: input.parse()?,
                mutability: input.parse()?,
                elem: Box::new(input.call(Type::without_plus)?),
            })
        }
    }
    impl Parse for TypeBareFn {
        fn parse(input: ParseStream) -> Result<Self> {
            let args;
            let mut variadic = None;
            Ok(TypeBareFn {
                lifetimes: input.parse()?,
                unsafety: input.parse()?,
                abi: input.parse()?,
                fn_token: input.parse()?,
                paren_token: parenthesized!(args in input),
                inputs: {
                    let mut inputs = Punctuated::new();
                    while !args.is_empty() {
                        let attrs = args.call(Attribute::parse_outer)?;
                        if inputs.empty_or_trailing()
                            && (args.peek(Token![...])
                                || args.peek(Ident) && args.peek2(Token![:]) && args.peek3(Token![...]))
                        {
                            variadic = Some(parse_bare_variadic(&args, attrs)?);
                            break;
                        }
                        let allow_self = inputs.is_empty();
                        let arg = parse_bare_fn_arg(&args, allow_self)?;
                        inputs.push_value(BareFnArg { attrs, ..arg });
                        if args.is_empty() {
                            break;
                        }
                        let comma = args.parse()?;
                        inputs.push_punct(comma);
                    }
                    inputs
                },
                variadic,
                output: input.call(ReturnType::without_plus)?,
            })
        }
    }
    impl Parse for TypeNever {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(TypeNever {
                bang_token: input.parse()?,
            })
        }
    }
    impl Parse for TypeInfer {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(TypeInfer {
                underscore_token: input.parse()?,
            })
        }
    }
    impl Parse for TypeTuple {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            let paren_token = parenthesized!(content in input);
            if content.is_empty() {
                return Ok(TypeTuple {
                    paren_token,
                    elems: Punctuated::new(),
                });
            }
            let first: Type = content.parse()?;
            Ok(TypeTuple {
                paren_token,
                elems: {
                    let mut elems = Punctuated::new();
                    elems.push_value(first);
                    elems.push_punct(content.parse()?);
                    while !content.is_empty() {
                        elems.push_value(content.parse()?);
                        if content.is_empty() {
                            break;
                        }
                        elems.push_punct(content.parse()?);
                    }
                    elems
                },
            })
        }
    }
    impl Parse for TypeMacro {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(TypeMacro { mac: input.parse()? })
        }
    }
    impl Parse for TypePath {
        fn parse(input: ParseStream) -> Result<Self> {
            let expr_style = false;
            let (qself, path) = qpath(input, expr_style)?;
            Ok(TypePath { qself, path })
        }
    }
    impl ReturnType {
        pub fn without_plus(input: ParseStream) -> Result<Self> {
            let allow_plus = false;
            Self::parse(input, allow_plus)
        }
        pub(crate) fn parse(input: ParseStream, allow_plus: bool) -> Result<Self> {
            if input.peek(Token![->]) {
                let arrow = input.parse()?;
                let allow_group_generic = true;
                let ty = ambig_ty(input, allow_plus, allow_group_generic)?;
                Ok(ReturnType::Type(arrow, Box::new(ty)))
            } else {
                Ok(ReturnType::Default)
            }
        }
    }
    impl Parse for ReturnType {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_plus = true;
            Self::parse(input, allow_plus)
        }
    }
    impl Parse for TypeTraitObject {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_plus = true;
            Self::parse(input, allow_plus)
        }
    }
    impl TypeTraitObject {
        pub fn without_plus(input: ParseStream) -> Result<Self> {
            let allow_plus = false;
            Self::parse(input, allow_plus)
        }
        pub(crate) fn parse(input: ParseStream, allow_plus: bool) -> Result<Self> {
            let dyn_token: Option<Token![dyn]> = input.parse()?;
            let dyn_span = match &dyn_token {
                Some(token) => token.span,
                None => input.span(),
            };
            let bounds = Self::parse_bounds(dyn_span, input, allow_plus)?;
            Ok(TypeTraitObject { dyn_token, bounds })
        }
        fn parse_bounds(
            dyn_span: Span,
            input: ParseStream,
            allow_plus: bool,
        ) -> Result<Punctuated<TypeParamBound, Token![+]>> {
            let bounds = TypeParamBound::parse_multiple(input, allow_plus)?;
            let mut last_lifetime_span = None;
            let mut at_least_one_trait = false;
            for bound in &bounds {
                match bound {
                    TypeParamBound::Trait(_) | TypeParamBound::Verbatim(_) => {
                        at_least_one_trait = true;
                        break;
                    },
                    TypeParamBound::Lifetime(lifetime) => {
                        last_lifetime_span = Some(lifetime.ident.span());
                    },
                }
            }
            if !at_least_one_trait {
                let msg = "at least one trait is required for an object type";
                return Err(err::new2(dyn_span, last_lifetime_span.unwrap(), msg));
            }
            Ok(bounds)
        }
    }
    impl Parse for TypeImplTrait {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_plus = true;
            Self::parse(input, allow_plus)
        }
    }
    impl TypeImplTrait {
        pub fn without_plus(input: ParseStream) -> Result<Self> {
            let allow_plus = false;
            Self::parse(input, allow_plus)
        }
        pub(crate) fn parse(input: ParseStream, allow_plus: bool) -> Result<Self> {
            let impl_token: Token![impl] = input.parse()?;
            let bounds = TypeParamBound::parse_multiple(input, allow_plus)?;
            let mut last_lifetime_span = None;
            let mut at_least_one_trait = false;
            for bound in &bounds {
                match bound {
                    TypeParamBound::Trait(_) | TypeParamBound::Verbatim(_) => {
                        at_least_one_trait = true;
                        break;
                    },
                    TypeParamBound::Lifetime(lifetime) => {
                        last_lifetime_span = Some(lifetime.ident.span());
                    },
                }
            }
            if !at_least_one_trait {
                let msg = "at least one trait must be specified";
                return Err(err::new2(impl_token.span, last_lifetime_span.unwrap(), msg));
            }
            Ok(TypeImplTrait { impl_token, bounds })
        }
    }
    impl Parse for TypeGroup {
        fn parse(input: ParseStream) -> Result<Self> {
            let group = crate::group::parse_group(input)?;
            Ok(TypeGroup {
                group_token: group.token,
                elem: group.content.parse()?,
            })
        }
    }
    impl Parse for TypeParen {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_plus = false;
            Self::parse(input, allow_plus)
        }
    }
    impl TypeParen {
        fn parse(input: ParseStream, allow_plus: bool) -> Result<Self> {
            let content;
            Ok(TypeParen {
                paren_token: parenthesized!(content in input),
                elem: Box::new({
                    let allow_group_generic = true;
                    ambig_ty(&content, allow_plus, allow_group_generic)?
                }),
            })
        }
    }
    impl Parse for BareFnArg {
        fn parse(input: ParseStream) -> Result<Self> {
            let allow_self = false;
            parse_bare_fn_arg(input, allow_self)
        }
    }
    fn parse_bare_fn_arg(input: ParseStream, allow_self: bool) -> Result<BareFnArg> {
        let attrs = input.call(Attribute::parse_outer)?;
        let begin = input.fork();
        let has_mut_self = allow_self && input.peek(Token![mut]) && input.peek2(Token![self]);
        if has_mut_self {
            input.parse::<Token![mut]>()?;
        }
        let mut has_self = false;
        let mut name = if (input.peek(Ident) || input.peek(Token![_]) || {
            has_self = allow_self && input.peek(Token![self]);
            has_self
        }) && input.peek2(Token![:])
            && !input.peek2(Token![::])
        {
            let name = input.call(Ident::parse_any)?;
            let colon: Token![:] = input.parse()?;
            Some((name, colon))
        } else {
            has_self = false;
            None
        };
        let ty = if allow_self && !has_self && input.peek(Token![mut]) && input.peek2(Token![self]) {
            input.parse::<Token![mut]>()?;
            input.parse::<Token![self]>()?;
            None
        } else if has_mut_self && name.is_none() {
            input.parse::<Token![self]>()?;
            None
        } else {
            Some(input.parse()?)
        };
        let ty = match ty {
            Some(ty) if !has_mut_self => ty,
            _ => {
                name = None;
                Type::Verbatim(verbatim_between(&begin, input))
            },
        };
        Ok(BareFnArg { attrs, name, ty })
    }
    fn parse_bare_variadic(input: ParseStream, attrs: Vec<Attribute>) -> Result<BareVariadic> {
        Ok(BareVariadic {
            attrs,
            name: if input.peek(Ident) || input.peek(Token![_]) {
                let name = input.call(Ident::parse_any)?;
                let colon: Token![:] = input.parse()?;
                Some((name, colon))
            } else {
                None
            },
            dots: input.parse()?,
            comma: input.parse()?,
        })
    }
    impl Parse for Abi {
        fn parse(input: ParseStream) -> Result<Self> {
            Ok(Abi {
                extern_token: input.parse()?,
                name: input.parse()?,
            })
        }
    }
    impl Parse for Option<Abi> {
        fn parse(input: ParseStream) -> Result<Self> {
            if input.peek(Token![extern]) {
                input.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }
}
