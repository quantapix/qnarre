use super::{
    err::{Err, Result},
    ext::IdentExt,
    parse::{discouraged::Speculative, Parse, ParseBuffer, ParseStream, Result},
    tok::Tok,
};
use proc_macro2::{Ident, Punct, Spacing, Span, Span, TokenStream};
use std::{
    cmp::Ordering,
    fmt::{self, Display},
};

pub fn parse_inner(x: ParseStream, ys: &mut Vec<Attribute>) -> Result<()> {
    while x.peek(Token![#]) && x.peek2(Token![!]) {
        ys.push(x.call(single_parse_inner)?);
    }
    Ok(())
}
pub fn single_parse_inner(x: ParseStream) -> Result<Attribute> {
    let content;
    Ok(Attribute {
        pound: x.parse()?,
        style: AttrStyle::Inner(x.parse()?),
        bracket: bracketed!(content in x),
        meta: content.parse()?,
    })
}
pub fn single_parse_outer(x: ParseStream) -> Result<Attribute> {
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

pub fn parse_meta_after_path(path: Path, x: ParseStream) -> Result<Meta> {
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
    let eq: Token![=] = x.parse()?;
    let ahead = x.fork();
    let lit: Option<Lit> = ahead.parse()?;
    let value = if let (Some(lit), true) = (lit, ahead.is_empty()) {
        x.advance_to(&ahead);
        Expr::Lit(expr::Lit { attrs: Vec::new(), lit })
    } else if x.peek(Token![#]) && x.peek2(tok::Bracket) {
        return Err(x.error("unexpected attribute inside of attribute"));
    } else {
        x.parse()?
    };
    Ok(MetaNameValue { path, eq, expr: value })
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
        for (i, x) in self.0.segs.iter().enumerate() {
            if i > 0 || self.0.colon.is_some() {
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
        let lt: Token![<] = input.parse()?;
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
                    eq: None,
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
        let gt: Token![>] = input.parse()?;
        Ok(Generics {
            lt: Some(lt),
            params,
            gt: Some(gt),
            where_: None,
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
            life: input.parse()?,
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
            lifes: {
                let mut lifetimes = Punctuated::new();
                while !input.peek(Token![>]) {
                    let attrs = input.call(Attribute::parse_outer)?;
                    let lifetime: Lifetime = input.parse()?;
                    lifetimes.push_value(GenericParam::Lifetime(LifetimeParam {
                        attrs,
                        life: lifetime,
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
        let colon: Option<Token![:]> = input.parse()?;
        let mut bounds = Punctuated::new();
        if colon.is_some() {
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
        let eq: Option<Token![=]> = input.parse()?;
        let default = if eq.is_some() {
            Some(input.parse::<ty::Type>()?)
        } else {
            None
        };
        Ok(TypeParam {
            attrs,
            ident,
            colon,
            bounds,
            eq,
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
        let (paren, content) = if input.peek(tok::Paren) {
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
        bound.paren = paren;
        if is_tilde_const {
            Ok(TypeParamBound::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(TypeParamBound::Trait(bound))
        }
    }
}

impl TypeParamBound {
    pub fn parse_multiple(input: ParseStream, allow_plus: bool) -> Result<Punctuated<Self, Token![+]>> {
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
        if path.segs.last().unwrap().args.is_empty()
            && (input.peek(tok::Paren) || input.peek(Token![::]) && input.peek3(tok::Paren))
        {
            input.parse::<Option<Token![::]>>()?;
            let args: ParenthesizedArgs = input.parse()?;
            let parenthesized = Args::Parenthesized(args);
            path.segs.last_mut().unwrap().args = parenthesized;
        }
        Ok(TraitBound {
            paren: None,
            modifier,
            lifes: lifetimes,
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
            typ: input.parse()?,
            eq: {
                if input.peek(Token![=]) {
                    let eq = input.parse()?;
                    default = Some(const_argument(input)?);
                    Some(eq)
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
                life: input.parse()?,
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
                lifes: input.parse()?,
                bounded: input.parse()?,
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

pub struct AllowStruct(bool);
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
            lhs = Expr::Binary(expr::Binary {
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
            let eq: Token![=] = input.parse()?;
            let mut rhs = unary_expr(input, allow_struct)?;
            loop {
                let next = peek_precedence(input);
                if next >= Precedence::Assign {
                    rhs = parse_expr(input, rhs, allow_struct, next)?;
                } else {
                    break;
                }
            }
            lhs = Expr::Assign(expr::Assign {
                attrs: Vec::new(),
                left: Box::new(lhs),
                eq,
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
            lhs = Expr::Range(expr::Range {
                attrs: Vec::new(),
                beg: Some(Box::new(lhs)),
                limits,
                end: rhs.map(Box::new),
            });
        } else if Precedence::Cast >= base && input.peek(Token![as]) {
            let as_: Token![as] = input.parse()?;
            let allow_plus = false;
            let group_gen = false;
            let ty = ambig_ty(input, allow_plus, group_gen)?;
            check_cast(input)?;
            lhs = Expr::Cast(expr::Cast {
                attrs: Vec::new(),
                expr: Box::new(lhs),
                as_,
                typ: Box::new(ty),
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
            let group = super::parse_group(&ahead)?;
            if !group.gist.peek(Token![#]) || group.gist.peek2(Token![!]) {
                break;
            }
            let attr = group.gist.call(single_parse_outer)?;
            if !group.gist.is_empty() {
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
        let and: Token![&] = input.parse()?;
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
            Ok(Expr::Reference(expr::Ref {
                attrs,
                and,
                mut_: mutability,
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
            e = Expr::Call(expr::Call {
                attrs: Vec::new(),
                func: Box::new(e),
                paren: parenthesized!(content in input),
                args: content.parse_terminated(Expr::parse, Token![,])?,
            });
        } else if input.peek(Token![.])
            && !input.peek(Token![..])
            && match e {
                Expr::Range(_) => false,
                _ => true,
            }
        {
            let mut dot: Token![.] = input.parse()?;
            let float_token: Option<lit::Float> = input.parse()?;
            if let Some(float_token) = float_token {
                if multi_index(&mut e, &mut dot, float_token)? {
                    continue;
                }
            }
            let await_: Option<Token![await]> = input.parse()?;
            if let Some(await_) = await_ {
                e = Expr::Await(expr::Await {
                    attrs: Vec::new(),
                    expr: Box::new(e),
                    dot,
                    await_,
                });
                continue;
            }
            let member: Member = input.parse()?;
            let turbofish = if member.is_named() && input.peek(Token![::]) {
                Some(AngledArgs::parse_turbofish(input)?)
            } else {
                None
            };
            if turbofish.is_some() || input.peek(tok::Paren) {
                if let Member::Named(method) = member {
                    let content;
                    e = Expr::MethodCall(expr::MethodCall {
                        attrs: Vec::new(),
                        expr: Box::new(e),
                        dot,
                        method,
                        turbofish,
                        paren: parenthesized!(content in input),
                        args: content.parse_terminated(Expr::parse, Token![,])?,
                    });
                    continue;
                }
            }
            e = Expr::Field(expr::Field {
                attrs: Vec::new(),
                base: Box::new(e),
                dot,
                memb: member,
            });
        } else if input.peek(tok::Bracket) {
            let content;
            e = Expr::Index(expr::Index {
                attrs: Vec::new(),
                expr: Box::new(e),
                bracket: bracketed!(content in input),
                index: content.parse()?,
            });
        } else if input.peek(Token![?]) {
            e = Expr::Try(expr::Try {
                attrs: Vec::new(),
                expr: Box::new(e),
                question: input.parse()?,
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
            Expr::While(expr::While { label, .. })
            | Expr::ForLoop(expr::ForLoop { label, .. })
            | Expr::Loop(expr::Loop { label, .. })
            | Expr::Block(expr::Block { label, .. }) => *label = Some(the_label),
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
        let bang: Token![!] = input.parse()?;
        let (delimiter, tokens) = mac_parse_delimiter(input)?;
        return Ok(Expr::Macro(expr::Mac {
            attrs: Vec::new(),
            mac: Macro {
                path,
                bang,
                delim: delimiter,
                toks: tokens,
            },
        }));
    }
    if allow_struct.0 && input.peek(tok::Brace) {
        return expr_struct_helper(input, qself, path).map(Expr::Struct);
    }
    Ok(Expr::Path(expr::Path {
        attrs: Vec::new(),
        qself,
        path,
    }))
}
impl Parse for expr::Mac {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Mac {
            attrs: Vec::new(),
            mac: input.parse()?,
        })
    }
}
fn paren_or_tuple(input: ParseStream) -> Result<Expr> {
    let content;
    let paren = parenthesized!(content in input);
    if content.is_empty() {
        return Ok(Expr::Tuple(expr::Tuple {
            attrs: Vec::new(),
            paren,
            elems: Punctuated::new(),
        }));
    }
    let first: Expr = content.parse()?;
    if content.is_empty() {
        return Ok(Expr::Paren(expr::Paren {
            attrs: Vec::new(),
            paren,
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
    Ok(Expr::Tuple(expr::Tuple {
        attrs: Vec::new(),
        paren,
        elems,
    }))
}
fn array_or_repeat(input: ParseStream) -> Result<Expr> {
    let content;
    let bracket = bracketed!(content in input);
    if content.is_empty() {
        return Ok(Expr::Array(expr::Array {
            attrs: Vec::new(),
            bracket,
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
        Ok(Expr::Array(expr::Array {
            attrs: Vec::new(),
            bracket,
            elems,
        }))
    } else if content.peek(Token![;]) {
        let semi: Token![;] = content.parse()?;
        let len: Expr = content.parse()?;
        Ok(Expr::Repeat(expr::Repeat {
            attrs: Vec::new(),
            bracket,
            expr: Box::new(first),
            semi,
            len: Box::new(len),
        }))
    } else {
        Err(content.error("expected `,` or `;`"))
    }
}
impl Parse for expr::Array {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        let bracket = bracketed!(content in input);
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
        Ok(expr::Array {
            attrs: Vec::new(),
            bracket,
            elems,
        })
    }
}
impl Parse for expr::Repeat {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(expr::Repeat {
            bracket: bracketed!(content in input),
            attrs: Vec::new(),
            expr: content.parse()?,
            semi: content.parse()?,
            len: content.parse()?,
        })
    }
}
pub fn expr_early(input: ParseStream) -> Result<Expr> {
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
impl Parse for expr::Lit {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Lit {
            attrs: Vec::new(),
            lit: input.parse()?,
        })
    }
}
fn expr_group(input: ParseStream) -> Result<expr::Group> {
    let group = super::parse_group(input)?;
    Ok(expr::Group {
        attrs: Vec::new(),
        group: group.token,
        expr: group.gist.parse()?,
    })
}
impl Parse for expr::Paren {
    fn parse(input: ParseStream) -> Result<Self> {
        expr_paren(input)
    }
}
fn expr_paren(input: ParseStream) -> Result<expr::Paren> {
    let content;
    Ok(expr::Paren {
        attrs: Vec::new(),
        paren: parenthesized!(content in input),
        expr: content.parse()?,
    })
}
impl Parse for expr::Let {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Let {
            attrs: Vec::new(),
            let_: input.parse()?,
            pat: Box::new(patt::Patt::parse_multi(input)?),
            eq: input.parse()?,
            expr: Box::new({
                let allow_struct = AllowStruct(false);
                let lhs = unary_expr(input, allow_struct)?;
                parse_expr(input, lhs, allow_struct, Precedence::Compare)?
            }),
        })
    }
}
impl Parse for expr::If {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        Ok(expr::If {
            attrs,
            if_: input.parse()?,
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
        Expr::Block(expr::Block {
            attrs: Vec::new(),
            label: None,
            block: input.parse()?,
        })
    } else {
        return Err(lookahead.error());
    };
    Ok((else_token, Box::new(else_branch)))
}
impl Parse for expr::Infer {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Infer {
            attrs: input.call(Attribute::parse_outer)?,
            underscore: input.parse()?,
        })
    }
}
impl Parse for expr::ForLoop {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let for_: Token![for] = input.parse()?;
        let pat = patt::Patt::parse_multi(input)?;
        let in_: Token![in] = input.parse()?;
        let expr: Expr = input.call(Expr::parse_without_eager_brace)?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::ForLoop {
            attrs,
            label,
            for_,
            pat: Box::new(pat),
            in_,
            expr: Box::new(expr),
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Loop {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let loop_: Token![loop] = input.parse()?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Loop {
            attrs,
            label,
            loop_,
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Match {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let match_: Token![match] = input.parse()?;
        let expr = Expr::parse_without_eager_brace(input)?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let mut arms = Vec::new();
        while !content.is_empty() {
            arms.push(content.call(Arm::parse)?);
        }
        Ok(expr::Match {
            attrs,
            match_,
            expr: Box::new(expr),
            brace,
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
    expr::Assign, Assign, "expected assignment expression",
    expr::Await, Await, "expected await expression",
    expr::Binary, Binary, "expected binary operation",
    expr::Call, Call, "expected function call expression",
    expr::Cast, Cast, "expected cast expression",
    expr::Field, Field, "expected struct field access",
    expr::Index, Index, "expected indexing expression",
    expr::MethodCall, MethodCall, "expected method call expression",
    expr::Range, Range, "expected range expression",
    expr::Try, Try, "expected try expression",
    expr::Tuple, Tuple, "expected tuple expression",
}
impl Parse for expr::Unary {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = Vec::new();
        let allow_struct = AllowStruct(true);
        expr_unary(input, attrs, allow_struct)
    }
}
fn expr_unary(input: ParseStream, attrs: Vec<Attribute>, allow_struct: AllowStruct) -> Result<expr::Unary> {
    Ok(expr::Unary {
        attrs,
        op: input.parse()?,
        expr: Box::new(unary_expr(input, allow_struct)?),
    })
}
impl Parse for expr::Closure {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_closure(input, allow_struct)
    }
}
impl Parse for expr::Ref {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        Ok(expr::Ref {
            attrs: Vec::new(),
            and: input.parse()?,
            mut_: input.parse()?,
            expr: Box::new(unary_expr(input, allow_struct)?),
        })
    }
}
impl Parse for expr::Break {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_break(input, allow_struct)
    }
}
impl Parse for expr::Return {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_struct = AllowStruct(true);
        expr_ret(input, allow_struct)
    }
}
impl Parse for expr::TryBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::TryBlock {
            attrs: Vec::new(),
            try_: input.parse()?,
            block: input.parse()?,
        })
    }
}
impl Parse for expr::Yield {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Yield {
            attrs: Vec::new(),
            yield_: input.parse()?,
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
fn expr_closure(input: ParseStream, allow_struct: AllowStruct) -> Result<expr::Closure> {
    let lifetimes: Option<BoundLifetimes> = input.parse()?;
    let constness: Option<Token![const]> = input.parse()?;
    let movability: Option<Token![static]> = input.parse()?;
    let asyncness: Option<Token![async]> = input.parse()?;
    let capture: Option<Token![move]> = input.parse()?;
    let or1: Token![|] = input.parse()?;
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
    let or2: Token![|] = input.parse()?;
    let (output, body) = if input.peek(Token![->]) {
        let arrow: Token![->] = input.parse()?;
        let typ: ty::Type = input.parse()?;
        let body: Block = input.parse()?;
        let output = ty::Ret::Type(arrow, Box::new(typ));
        let block = Expr::Block(expr::Block {
            attrs: Vec::new(),
            label: None,
            block: body,
        });
        (output, block)
    } else {
        let body = ambiguous_expr(input, allow_struct)?;
        (ty::Ret::Default, body)
    };
    Ok(expr::Closure {
        attrs: Vec::new(),
        lifes: lifetimes,
        const_: constness,
        static_: movability,
        async_: asyncness,
        move_: capture,
        or1,
        inputs,
        or2,
        ret: output,
        body: Box::new(body),
    })
}
impl Parse for expr::Async {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Async {
            attrs: Vec::new(),
            async_: input.parse()?,
            move_: input.parse()?,
            block: input.parse()?,
        })
    }
}
fn closure_arg(input: ParseStream) -> Result<patt::Patt> {
    let attrs = input.call(Attribute::parse_outer)?;
    let mut pat = patt::Patt::parse_single(input)?;
    if input.peek(Token![:]) {
        Ok(patt::Patt::Type(patt::Type {
            attrs,
            pat: Box::new(pat),
            colon: input.parse()?,
            ty: input.parse()?,
        }))
    } else {
        match &mut pat {
            patt::Patt::Const(pat) => pat.attrs = attrs,
            patt::Patt::Ident(pat) => pat.attrs = attrs,
            patt::Patt::Lit(pat) => pat.attrs = attrs,
            patt::Patt::Macro(pat) => pat.attrs = attrs,
            patt::Patt::Or(pat) => pat.attrs = attrs,
            patt::Patt::Paren(pat) => pat.attrs = attrs,
            patt::Patt::Path(pat) => pat.attrs = attrs,
            patt::Patt::Range(pat) => pat.attrs = attrs,
            patt::Patt::Reference(pat) => pat.attrs = attrs,
            patt::Patt::Rest(pat) => pat.attrs = attrs,
            patt::Patt::Slice(pat) => pat.attrs = attrs,
            patt::Patt::Struct(pat) => pat.attrs = attrs,
            patt::Patt::Tuple(pat) => pat.attrs = attrs,
            patt::Patt::TupleStruct(pat) => pat.attrs = attrs,
            patt::Patt::Type(_) => unreachable!(),
            patt::Patt::Verbatim(_) => {},
            patt::Patt::Wild(pat) => pat.attrs = attrs,
        }
        Ok(pat)
    }
}
impl Parse for expr::While {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let while_: Token![while] = input.parse()?;
        let cond = Expr::parse_without_eager_brace(input)?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::While {
            attrs,
            label,
            while_,
            cond: Box::new(cond),
            body: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Const {
    fn parse(input: ParseStream) -> Result<Self> {
        let const_: Token![const] = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let inner_attrs = content.call(Attribute::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Const {
            attrs: inner_attrs,
            const_,
            block: Block { brace, stmts },
        })
    }
}
impl Parse for Label {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Label {
            name: input.parse()?,
            colon: input.parse()?,
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
impl Parse for expr::Continue {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(expr::Continue {
            attrs: Vec::new(),
            continue_: input.parse()?,
            label: input.parse()?,
        })
    }
}
fn expr_break(input: ParseStream, allow_struct: AllowStruct) -> Result<expr::Break> {
    Ok(expr::Break {
        attrs: Vec::new(),
        break_: input.parse()?,
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
fn expr_ret(input: ParseStream, allow_struct: AllowStruct) -> Result<expr::Return> {
    Ok(expr::Return {
        attrs: Vec::new(),
        return_: input.parse()?,
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
        let (colon, value) = if input.peek(Token![:]) || !member.is_named() {
            let colon: Token![:] = input.parse()?;
            let value: Expr = input.parse()?;
            (Some(colon), value)
        } else if let Member::Named(ident) = &member {
            let value = Expr::Path(expr::Path {
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
            colon,
            expr: value,
        })
    }
}
impl Parse for expr::Struct {
    fn parse(input: ParseStream) -> Result<Self> {
        let (qself, path) = qpath(input, true)?;
        expr_struct_helper(input, qself, path)
    }
}
fn expr_struct_helper(input: ParseStream, qself: Option<QSelf>, path: Path) -> Result<expr::Struct> {
    let content;
    let brace = braced!(content in input);
    let mut fields = Punctuated::new();
    while !content.is_empty() {
        if content.peek(Token![..]) {
            return Ok(expr::Struct {
                attrs: Vec::new(),
                qself,
                path,
                brace,
                fields,
                dot2: Some(content.parse()?),
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
    Ok(expr::Struct {
        attrs: Vec::new(),
        qself,
        path,
        brace,
        fields,
        dot2: None,
        rest: None,
    })
}
impl Parse for expr::Unsafe {
    fn parse(input: ParseStream) -> Result<Self> {
        let unsafe_: Token![unsafe] = input.parse()?;
        let content;
        let brace = braced!(content in input);
        let inner_attrs = content.call(Attribute::parse_inner)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Unsafe {
            attrs: inner_attrs,
            unsafe_,
            block: Block { brace, stmts },
        })
    }
}
impl Parse for expr::Block {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let label: Option<Label> = input.parse()?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let stmts = content.call(Block::parse_within)?;
        Ok(expr::Block {
            attrs,
            label,
            block: Block { brace, stmts },
        })
    }
}
fn expr_range(input: ParseStream, allow_struct: AllowStruct) -> Result<expr::Range> {
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
    Ok(expr::Range {
        attrs: Vec::new(),
        beg: None,
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
    pub fn parse_obsolete(input: ParseStream) -> Result<Self> {
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
impl Parse for expr::Path {
    fn parse(input: ParseStream) -> Result<Self> {
        #[cfg(not(feature = "full"))]
        let attrs = Vec::new();
        let attrs = input.call(Attribute::parse_outer)?;
        let (qself, path) = qpath(input, true)?;
        Ok(expr::Path { attrs, qself, path })
    }
}
impl Parse for Member {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Ident) {
            input.parse().map(Member::Named)
        } else if input.peek(lit::Int) {
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
            pat: patt::Patt::parse_multi(input)?,
            guard: {
                if input.peek(Token![if]) {
                    let if_: Token![if] = input.parse()?;
                    let guard: Expr = input.parse()?;
                    Some((if_, Box::new(guard)))
                } else {
                    None
                }
            },
            fat_arrow: input.parse()?,
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
        let lit: lit::Int = input.parse()?;
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
fn multi_index(e: &mut Expr, dot: &mut Token![.], float: lit::Float) -> Result<bool> {
    let float_token = float.token();
    let float_span = float_token.span();
    let mut float_repr = float_token.to_string();
    let trailing_dot = float_repr.ends_with('.');
    if trailing_dot {
        float_repr.truncate(float_repr.len() - 1);
    }
    let mut offset = 0;
    for part in float_repr.split('.') {
        let mut index: Index = super::parse_str(part).map_err(|err| Err::new(float_span, err))?;
        let part_end = offset + part.len();
        index.span = float_token.subspan(offset..part_end).unwrap_or(float_span);
        let base = mem::replace(e, Expr::DUMMY);
        *e = Expr::Field(expr::Field {
            attrs: Vec::new(),
            base: Box::new(base),
            dot: Token![.](dot.span),
            memb: Member::Unnamed(index),
        });
        let dot_span = float_token.subspan(part_end..part_end + 1).unwrap_or(float_span);
        *dot = Token![.](dot_span);
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
pub fn parse_rest_of_item(begin: ParseBuffer, mut attrs: Vec<Attribute>, input: ParseStream) -> Result<Item> {
    let ahead = input.fork();
    let vis: Visibility = ahead.parse()?;
    let lookahead = ahead.lookahead1();
    let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
        let vis: Visibility = input.parse()?;
        let sig: item::Sig = input.parse()?;
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
            input.parse().map(Item::Foreign)
        } else if lookahead.peek(lit::Str) {
            ahead.parse::<lit::Str>()?;
            let lookahead = ahead.lookahead1();
            if lookahead.peek(tok::Brace) {
                input.parse().map(Item::Foreign)
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
        let static_ = input.parse()?;
        let mutability = input.parse()?;
        let ident = input.parse()?;
        if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            input.parse::<Expr>()?;
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            let colon = input.parse()?;
            let ty = input.parse()?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                Ok(Item::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(Item::Static(item::Static {
                    attrs: Vec::new(),
                    vis,
                    static_,
                    mut_: mutability,
                    ident,
                    colon,
                    typ: ty,
                    eq: input.parse()?,
                    expr: input.parse()?,
                    semi: input.parse()?,
                }))
            }
        }
    } else if lookahead.peek(Token![const]) {
        let vis = input.parse()?;
        let const_: Token![const] = input.parse()?;
        let lookahead = input.lookahead1();
        let ident = if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
            input.call(Ident::parse_any)?
        } else {
            return Err(lookahead.error());
        };
        let colon = input.parse()?;
        let ty = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(Item::Verbatim(verbatim_between(&begin, input)))
        } else {
            Ok(Item::Const(item::Const {
                attrs: Vec::new(),
                vis,
                const_,
                ident,
                gens: Generics::default(),
                colon,
                typ: ty,
                eq: input.parse()?,
                expr: input.parse()?,
                semi: input.parse()?,
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
            input.parse().map(Item::Foreign)
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
    default_: Option<Token![default]>,
    type_: Token![type],
    ident: Ident,
    gens: Generics,
    colon: Option<Token![:]>,
    bounds: Punctuated<TypeParamBound, Token![+]>,
    ty: Option<(Token![=], ty::Type)>,
    semi: Token![;],
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
        let default_: Option<Token![default]> = match allow_defaultness {
            TypeDefaultness::Optional => input.parse()?,
            TypeDefaultness::Disallowed => None,
        };
        let type_: Token![type] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut gens: Generics = input.parse()?;
        let (colon, bounds) = Self::parse_optional_bounds(input)?;
        match where_clause_location {
            WhereClauseLocation::BeforeEq | WhereClauseLocation::Both => {
                gens.where_ = input.parse()?;
            },
            WhereClauseLocation::AfterEq => {},
        }
        let ty = Self::parse_optional_definition(input)?;
        match where_clause_location {
            WhereClauseLocation::AfterEq | WhereClauseLocation::Both if gens.where_.is_none() => {
                gens.where_ = input.parse()?;
            },
            _ => {},
        }
        let semi: Token![;] = input.parse()?;
        Ok(FlexibleItemType {
            vis,
            default_,
            type_,
            ident,
            gens,
            colon,
            bounds,
            ty,
            semi,
        })
    }
    fn parse_optional_bounds(input: ParseStream) -> Result<(Option<Token![:]>, Punctuated<TypeParamBound, Token![+]>)> {
        let colon: Option<Token![:]> = input.parse()?;
        let mut bounds = Punctuated::new();
        if colon.is_some() {
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
        Ok((colon, bounds))
    }
    fn parse_optional_definition(input: ParseStream) -> Result<Option<(Token![=], ty::Type)>> {
        let eq: Option<Token![=]> = input.parse()?;
        if let Some(eq) = eq {
            let definition: ty::Type = input.parse()?;
            Ok(Some((eq, definition)))
        } else {
            Ok(None)
        }
    }
}
impl Parse for item::Mac {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let path = input.call(Path::parse_mod_style)?;
        let bang: Token![!] = input.parse()?;
        let ident: Option<Ident> = if input.peek(Token![try]) {
            input.call(Ident::parse_any).map(Some)
        } else {
            input.parse()
        }?;
        let (delimiter, tokens) = input.call(mac::mac_parse_delimiter)?;
        let semi: Option<Token![;]> = if !delimiter.is_brace() {
            Some(input.parse()?)
        } else {
            None
        };
        Ok(item::Mac {
            attrs,
            ident,
            mac: Macro {
                path,
                bang,
                delim: delimiter,
                toks: tokens,
            },
            semi,
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
impl Parse for item::ExternCrate {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::ExternCrate {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            extern_: input.parse()?,
            crate_: input.parse()?,
            ident: {
                if input.peek(Token![self]) {
                    input.call(Ident::parse_any)?
                } else {
                    input.parse()?
                }
            },
            rename: {
                if input.peek(Token![as]) {
                    let as_: Token![as] = input.parse()?;
                    let rename: Ident = if input.peek(Token![_]) {
                        Ident::from(input.parse::<Token![_]>()?)
                    } else {
                        input.parse()?
                    };
                    Some((as_, rename))
                } else {
                    None
                }
            },
            semi: input.parse()?,
        })
    }
}
impl Parse for item::Use {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_crate_root_in_path = false;
        parse_item_use(input, allow_crate_root_in_path).map(Option::unwrap)
    }
}
fn parse_item_use(input: ParseStream, allow_crate_root_in_path: bool) -> Result<Option<item::Use>> {
    let attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let use_: Token![use] = input.parse()?;
    let leading_colon: Option<Token![::]> = input.parse()?;
    let tree = parse_use_tree(input, allow_crate_root_in_path && leading_colon.is_none())?;
    let semi: Token![;] = input.parse()?;
    let tree = match tree {
        Some(tree) => tree,
        None => return Ok(None),
    };
    Ok(Some(item::Use {
        attrs,
        vis,
        use_,
        colon: leading_colon,
        tree,
        semi,
    }))
}
impl Parse for item::Use::Tree {
    fn parse(input: ParseStream) -> Result<item::Use::Tree> {
        let allow_crate_root_in_path = false;
        parse_use_tree(input, allow_crate_root_in_path).map(Option::unwrap)
    }
}
fn parse_use_tree(input: ParseStream, allow_crate_root_in_path: bool) -> Result<Option<item::Use::Tree>> {
    let lookahead = input.lookahead1();
    if lookahead.peek(Ident)
        || lookahead.peek(Token![self])
        || lookahead.peek(Token![super])
        || lookahead.peek(Token![crate])
        || lookahead.peek(Token![try])
    {
        let ident = input.call(Ident::parse_any)?;
        if input.peek(Token![::]) {
            Ok(Some(item::Use::Tree::Path(item::Use::Path {
                ident,
                colon2: input.parse()?,
                tree: Box::new(input.parse()?),
            })))
        } else if input.peek(Token![as]) {
            Ok(Some(item::Use::Tree::Rename(item::Use::Rename {
                ident,
                as_: input.parse()?,
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
            Ok(Some(item::Use::Tree::Name(item::Use::Name { ident })))
        }
    } else if lookahead.peek(Token![*]) {
        Ok(Some(item::Use::Tree::Glob(item::Use::Glob { star: input.parse()? })))
    } else if lookahead.peek(tok::Brace) {
        let content;
        let brace = braced!(content in input);
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
            Ok(Some(item::Use::Tree::Group(item::Use::Group { brace, elems: items })))
        }
    } else {
        Err(lookahead.error())
    }
}
impl Parse for item::Static {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Static {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            static_: input.parse()?,
            mut_: input.parse()?,
            ident: input.parse()?,
            colon: input.parse()?,
            typ: input.parse()?,
            eq: input.parse()?,
            expr: input.parse()?,
            semi: input.parse()?,
        })
    }
}
impl Parse for item::Const {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Const {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            const_: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            gens: Generics::default(),
            colon: input.parse()?,
            typ: input.parse()?,
            eq: input.parse()?,
            expr: input.parse()?,
            semi: input.parse()?,
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
impl Parse for item::Sig {
    fn parse(x: ParseStream) -> Result<Self> {
        let const_: Option<Token![const]> = x.parse()?;
        let async_: Option<Token![async]> = x.parse()?;
        let unsafe_: Option<Token![unsafe]> = x.parse()?;
        let abi: Option<Abi> = x.parse()?;
        let fn_: Token![fn] = x.parse()?;
        let ident: Ident = x.parse()?;
        let mut gens: Generics = x.parse()?;
        let gist;
        let paren = parenthesized!(gist in x);
        let (args, vari) = parse_fn_args(&gist)?;
        let ret: ty::Ret = x.parse()?;
        gens.where_ = x.parse()?;
        Ok(item::Sig {
            constness: const_,
            async_,
            unsafe_,
            abi,
            fn_,
            ident,
            gens,
            paren,
            args,
            vari,
            ret,
        })
    }
}
impl Parse for item::Fn {
    fn parse(input: ParseStream) -> Result<Self> {
        let outer_attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: item::Sig = input.parse()?;
        parse_rest_of_fn(input, outer_attrs, vis, sig)
    }
}
fn parse_rest_of_fn(
    input: ParseStream,
    mut attrs: Vec<Attribute>,
    vis: Visibility,
    sig: item::Sig,
) -> Result<item::Fn> {
    let content;
    let brace = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let stmts = content.call(Block::parse_within)?;
    Ok(item::Fn {
        attrs,
        vis,
        sig,
        block: Box::new(Block { brace, stmts }),
    })
}
impl Parse for item::FnArg {
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
    FnArg(item::FnArg),
    Variadic(item::Variadic),
}
fn parse_fn_arg_or_variadic(
    input: ParseStream,
    attrs: Vec<Attribute>,
    allow_variadic: bool,
) -> Result<FnArgOrVariadic> {
    let ahead = input.fork();
    if let Ok(mut receiver) = ahead.parse::<item::Receiver>() {
        input.advance_to(&ahead);
        receiver.attrs = attrs;
        return Ok(FnArgOrVariadic::FnArg(item::FnArg::Receiver(receiver)));
    }
    if input.peek(Ident) && input.peek2(Token![<]) {
        let span = input.fork().parse::<Ident>()?.span();
        return Ok(FnArgOrVariadic::FnArg(item::FnArg::Typed(patt::Type {
            attrs,
            pat: Box::new(patt::Patt::Wild(patt::Wild {
                attrs: Vec::new(),
                underscore: Token![_](span),
            })),
            colon: Token![:](span),
            ty: input.parse()?,
        })));
    }
    let pat = Box::new(patt::Patt::parse_single(input)?);
    let colon: Token![:] = input.parse()?;
    if allow_variadic {
        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            return Ok(FnArgOrVariadic::Variadic(item::Variadic {
                attrs,
                pat: Some((pat, colon)),
                dots,
                comma: None,
            }));
        }
    }
    Ok(FnArgOrVariadic::FnArg(item::FnArg::Typed(patt::Type {
        attrs,
        pat,
        colon,
        ty: input.parse()?,
    })))
}
impl Parse for item::Receiver {
    fn parse(x: ParseStream) -> Result<Self> {
        let reference = if x.peek(Token![&]) {
            let ampersand: Token![&] = x.parse()?;
            let lifetime: Option<Lifetime> = x.parse()?;
            Some((ampersand, lifetime))
        } else {
            None
        };
        let mut_: Option<Token![mut]> = x.parse()?;
        let self_: Token![self] = x.parse()?;
        let colon: Option<Token![:]> = if reference.is_some() { None } else { x.parse()? };
        let typ: ty::Type = if colon.is_some() {
            x.parse()?
        } else {
            let mut ty = ty::Type::Path(ty::Path {
                qself: None,
                path: Path::from(Ident::new("Self", self_.span)),
            });
            if let Some((ampersand, lifetime)) = reference.as_ref() {
                ty = ty::Type::Ref(ty::Ref {
                    and: Token![&](ampersand.span),
                    lifetime: lifetime.clone(),
                    mut_: mut_.as_ref().map(|x| Token![mut](x.span)),
                    elem: Box::new(ty),
                });
            }
            ty
        };
        Ok(item::Receiver {
            attrs: Vec::new(),
            reference,
            mut_,
            self_,
            colon,
            typ: Box::new(typ),
        })
    }
}
fn parse_fn_args(input: ParseStream) -> Result<(Punctuated<item::FnArg, Token![,]>, Option<item::Variadic>)> {
    let mut args = Punctuated::new();
    let mut vari = None;
    let mut has_receiver = false;
    while !input.is_empty() {
        let attrs = input.call(Attribute::parse_outer)?;
        if let Some(dots) = input.parse::<Option<Token![...]>>()? {
            vari = Some(item::Variadic {
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
                vari = Some(item::Variadic {
                    comma: if input.is_empty() { None } else { Some(input.parse()?) },
                    ..arg
                });
                break;
            },
        };
        match &arg {
            item::FnArg::Receiver(receiver) if has_receiver => {
                return Err(Err::new(receiver.self_.span, "unexpected second method receiver"));
            },
            item::FnArg::Receiver(receiver) if !args.is_empty() => {
                return Err(Err::new(receiver.self_.span, "unexpected method receiver"));
            },
            item::FnArg::Receiver(_) => has_receiver = true,
            item::FnArg::Typed(_) => {},
        }
        args.push_value(arg);
        if input.is_empty() {
            break;
        }
        let comma: Token![,] = input.parse()?;
        args.push_punct(comma);
    }
    Ok((args, vari))
}
impl Parse for item::Mod {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let mod_: Token![mod] = input.parse()?;
        let ident: Ident = if input.peek(Token![try]) {
            input.call(Ident::parse_any)
        } else {
            input.parse()
        }?;
        let lookahead = input.lookahead1();
        if lookahead.peek(Token![;]) {
            Ok(item::Mod {
                attrs,
                vis,
                unsafe_: unsafety,
                mod_,
                ident,
                gist: None,
                semi: Some(input.parse()?),
            })
        } else if lookahead.peek(tok::Brace) {
            let content;
            let brace = braced!(content in input);
            parse_inner(&content, &mut attrs)?;
            let mut items = Vec::new();
            while !content.is_empty() {
                items.push(content.parse()?);
            }
            Ok(item::Mod {
                attrs,
                vis,
                unsafe_: unsafety,
                mod_,
                ident,
                gist: Some((brace, items)),
                semi: None,
            })
        } else {
            Err(lookahead.error())
        }
    }
}
impl Parse for item::Foreign {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let abi: Abi = input.parse()?;
        let content;
        let brace = braced!(content in input);
        parse_inner(&content, &mut attrs)?;
        let mut items = Vec::new();
        while !content.is_empty() {
            items.push(content.parse()?);
        }
        Ok(item::Foreign {
            attrs,
            unsafe_: unsafety,
            abi,
            brace,
            items,
        })
    }
}
impl Parse for item::Foreign::Item {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let mut attrs = input.call(Attribute::parse_outer)?;
        let ahead = input.fork();
        let vis: Visibility = ahead.parse()?;
        let lookahead = ahead.lookahead1();
        let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
            let vis: Visibility = input.parse()?;
            let sig: item::Sig = input.parse()?;
            if input.peek(tok::Brace) {
                let content;
                braced!(content in input);
                content.call(Attribute::parse_inner)?;
                content.call(Block::parse_within)?;
                Ok(item::Foreign::Item::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(item::Foreign::Item::Fn(item::Foreign::Fn {
                    attrs: Vec::new(),
                    vis,
                    sig,
                    semi: input.parse()?,
                }))
            }
        } else if lookahead.peek(Token![static]) {
            let vis = input.parse()?;
            let static_ = input.parse()?;
            let mutability = input.parse()?;
            let ident = input.parse()?;
            let colon = input.parse()?;
            let ty = input.parse()?;
            if input.peek(Token![=]) {
                input.parse::<Token![=]>()?;
                input.parse::<Expr>()?;
                input.parse::<Token![;]>()?;
                Ok(item::Foreign::Item::Verbatim(verbatim_between(&begin, input)))
            } else {
                Ok(item::Foreign::Item::Static(item::Foreign::Static {
                    attrs: Vec::new(),
                    vis,
                    static_,
                    mut_: mutability,
                    ident,
                    colon,
                    typ: ty,
                    semi: input.parse()?,
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
            input.parse().map(item::Foreign::Item::Macro)
        } else {
            Err(lookahead.error())
        }?;
        let item_attrs = match &mut item {
            item::Foreign::Item::Fn(item) => &mut item.attrs,
            item::Foreign::Item::Static(item) => &mut item.attrs,
            item::Foreign::Item::Type(item) => &mut item.attrs,
            item::Foreign::Item::Macro(item) => &mut item.attrs,
            item::Foreign::Item::Verbatim(_) => return Ok(item),
        };
        attrs.append(item_attrs);
        *item_attrs = attrs;
        Ok(item)
    }
}
impl Parse for item::Foreign::Fn {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let sig: item::Sig = input.parse()?;
        let semi: Token![;] = input.parse()?;
        Ok(item::Foreign::Fn { attrs, vis, sig, semi })
    }
}
impl Parse for item::Foreign::Static {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Foreign::Static {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            static_: input.parse()?,
            mut_: input.parse()?,
            ident: input.parse()?,
            colon: input.parse()?,
            typ: input.parse()?,
            semi: input.parse()?,
        })
    }
}
impl Parse for item::Foreign::Type {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Foreign::Type {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            type_: input.parse()?,
            ident: input.parse()?,
            gens: {
                let mut gens: Generics = input.parse()?;
                gens.where_ = input.parse()?;
                gens
            },
            semi: input.parse()?,
        })
    }
}
fn parse_foreign_item_type(begin: ParseBuffer, input: ParseStream) -> Result<item::Foreign::Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        ty,
        semi,
    } = FlexibleItemTy::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::Both)?;
    if colon.is_some() || ty.is_some() {
        Ok(item::Foreign::Item::Verbatim(verbatim_between(&begin, input)))
    } else {
        Ok(item::Foreign::Item::Type(item::Foreign::Type {
            attrs: Vec::new(),
            vis,
            type_,
            ident,
            gens,
            semi,
        }))
    }
}
impl Parse for item::Foreign::Mac {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(item::Foreign::Mac { attrs, mac, semi })
    }
}
impl Parse for item::Type {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Type {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            type_: input.parse()?,
            ident: input.parse()?,
            gens: {
                let mut gens: Generics = input.parse()?;
                gens.where_ = input.parse()?;
                gens
            },
            eq: input.parse()?,
            typ: input.parse()?,
            semi: input.parse()?,
        })
    }
}
fn parse_item_type(begin: ParseBuffer, input: ParseStream) -> Result<Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        ty,
        semi,
    } = FlexibleItemTy::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::BeforeEq)?;
    let (eq, ty) = match ty {
        Some(ty) if colon.is_none() => ty,
        _ => return Ok(Item::Verbatim(verbatim_between(&begin, input))),
    };
    Ok(Item::Type(item::Type {
        attrs: Vec::new(),
        vis,
        type_,
        ident,
        gens,
        eq,
        typ: Box::new(ty),
        semi,
    }))
}
impl Parse for item::Struct {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let struct_ = input.parse::<Token![struct]>()?;
        let ident = input.parse::<Ident>()?;
        let gens = input.parse::<Generics>()?;
        let (where_clause, fields, semi) = data_struct(input)?;
        Ok(item::Struct {
            attrs,
            vis,
            struct_,
            ident,
            gens: Generics {
                where_: where_clause,
                ..gens
            },
            fields,
            semi,
        })
    }
}
impl Parse for item::Enum {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let enum_ = input.parse::<Token![enum]>()?;
        let ident = input.parse::<Ident>()?;
        let gens = input.parse::<Generics>()?;
        let (where_clause, brace, variants) = data_enum(input)?;
        Ok(item::Enum {
            attrs,
            vis,
            enum_,
            ident,
            gens: Generics {
                where_: where_clause,
                ..gens
            },
            brace,
            elems: variants,
        })
    }
}
impl Parse for item::Union {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse::<Visibility>()?;
        let union_ = input.parse::<Token![union]>()?;
        let ident = input.parse::<Ident>()?;
        let gens = input.parse::<Generics>()?;
        let (where_clause, fields) = data_union(input)?;
        Ok(item::Union {
            attrs,
            vis,
            union_,
            ident,
            gens: Generics {
                where_: where_clause,
                ..gens
            },
            fields,
        })
    }
}
fn parse_trait_or_trait_alias(input: ParseStream) -> Result<Item> {
    let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(input)?;
    let lookahead = input.lookahead1();
    if lookahead.peek(tok::Brace) || lookahead.peek(Token![:]) || lookahead.peek(Token![where]) {
        let unsafety = None;
        let auto_ = None;
        parse_rest_of_trait(input, attrs, vis, unsafety, auto_, trait_, ident, gens).map(Item::Trait)
    } else if lookahead.peek(Token![=]) {
        parse_rest_of_trait_alias(input, attrs, vis, trait_, ident, gens).map(Item::TraitAlias)
    } else {
        Err(lookahead.error())
    }
}
impl Parse for item::Trait {
    fn parse(input: ParseStream) -> Result<Self> {
        let outer_attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let unsafety: Option<Token![unsafe]> = input.parse()?;
        let auto_: Option<Token![auto]> = input.parse()?;
        let trait_: Token![trait] = input.parse()?;
        let ident: Ident = input.parse()?;
        let gens: Generics = input.parse()?;
        parse_rest_of_trait(input, outer_attrs, vis, unsafety, auto_, trait_, ident, gens)
    }
}
fn parse_rest_of_trait(
    input: ParseStream,
    mut attrs: Vec<Attribute>,
    vis: Visibility,
    unsafety: Option<Token![unsafe]>,
    auto_: Option<Token![auto]>,
    trait_: Token![trait],
    ident: Ident,
    mut gens: Generics,
) -> Result<item::Trait> {
    let colon: Option<Token![:]> = input.parse()?;
    let mut supertraits = Punctuated::new();
    if colon.is_some() {
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
    gens.where_ = input.parse()?;
    let content;
    let brace = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let mut items = Vec::new();
    while !content.is_empty() {
        items.push(content.parse()?);
    }
    Ok(item::Trait {
        attrs,
        vis,
        unsafe_: unsafety,
        auto_,
        restriction: None,
        trait_,
        ident,
        gens,
        colon,
        supers: supertraits,
        brace,
        items,
    })
}
impl Parse for item::TraitAlias {
    fn parse(input: ParseStream) -> Result<Self> {
        let (attrs, vis, trait_, ident, gens) = parse_start_of_trait_alias(input)?;
        parse_rest_of_trait_alias(input, attrs, vis, trait_, ident, gens)
    }
}
fn parse_start_of_trait_alias(
    input: ParseStream,
) -> Result<(Vec<Attribute>, Visibility, Token![trait], Ident, Generics)> {
    let attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let trait_: Token![trait] = input.parse()?;
    let ident: Ident = input.parse()?;
    let gens: Generics = input.parse()?;
    Ok((attrs, vis, trait_, ident, gens))
}
fn parse_rest_of_trait_alias(
    input: ParseStream,
    attrs: Vec<Attribute>,
    vis: Visibility,
    trait_: Token![trait],
    ident: Ident,
    mut gens: Generics,
) -> Result<item::TraitAlias> {
    let eq: Token![=] = input.parse()?;
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
    gens.where_ = input.parse()?;
    let semi: Token![;] = input.parse()?;
    Ok(item::TraitAlias {
        attrs,
        vis,
        trait_,
        ident,
        gens,
        eq,
        bounds,
        semi,
    })
}
impl Parse for item::Trait::Item {
    fn parse(input: ParseStream) -> Result<Self> {
        let begin = input.fork();
        let mut attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        let default_: Option<Token![default]> = input.parse()?;
        let ahead = input.fork();
        let lookahead = ahead.lookahead1();
        let mut item = if lookahead.peek(Token![fn]) || peek_signature(&ahead) {
            input.parse().map(item::Trait::Item::Fn)
        } else if lookahead.peek(Token![const]) {
            ahead.parse::<Token![const]>()?;
            let lookahead = ahead.lookahead1();
            if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                input.parse().map(item::Trait::Item::Const)
            } else if lookahead.peek(Token![async])
                || lookahead.peek(Token![unsafe])
                || lookahead.peek(Token![extern])
                || lookahead.peek(Token![fn])
            {
                input.parse().map(item::Trait::Item::Fn)
            } else {
                Err(lookahead.error())
            }
        } else if lookahead.peek(Token![type]) {
            parse_trait_item_type(begin.fork(), input)
        } else if vis.is_inherited()
            && default_.is_none()
            && (lookahead.peek(Ident)
                || lookahead.peek(Token![self])
                || lookahead.peek(Token![super])
                || lookahead.peek(Token![crate])
                || lookahead.peek(Token![::]))
        {
            input.parse().map(item::Trait::Item::Macro)
        } else {
            Err(lookahead.error())
        }?;
        match (vis, default_) {
            (Visibility::Inherited, None) => {},
            _ => return Ok(item::Trait::Item::Verbatim(verbatim_between(&begin, input))),
        }
        let item_attrs = match &mut item {
            item::Trait::Item::Const(item) => &mut item.attrs,
            item::Trait::Item::Fn(item) => &mut item.attrs,
            item::Trait::Item::Type(item) => &mut item.attrs,
            item::Trait::Item::Macro(item) => &mut item.attrs,
            item::Trait::Item::Verbatim(_) => unreachable!(),
        };
        attrs.append(item_attrs);
        *item_attrs = attrs;
        Ok(item)
    }
}
impl Parse for item::Trait::Const {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Trait::Const {
            attrs: input.call(Attribute::parse_outer)?,
            const_: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            gens: Generics::default(),
            colon: input.parse()?,
            typ: input.parse()?,
            default: {
                if input.peek(Token![=]) {
                    let eq: Token![=] = input.parse()?;
                    let default: Expr = input.parse()?;
                    Some((eq, default))
                } else {
                    None
                }
            },
            semi: input.parse()?,
        })
    }
}
impl Parse for item::Trait::Fn {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let sig: item::Sig = input.parse()?;
        let lookahead = input.lookahead1();
        let (brace, stmts, semi) = if lookahead.peek(tok::Brace) {
            let content;
            let brace = braced!(content in input);
            parse_inner(&content, &mut attrs)?;
            let stmts = content.call(Block::parse_within)?;
            (Some(brace), stmts, None)
        } else if lookahead.peek(Token![;]) {
            let semi: Token![;] = input.parse()?;
            (None, Vec::new(), Some(semi))
        } else {
            return Err(lookahead.error());
        };
        Ok(item::Trait::Fn {
            attrs,
            sig,
            default: brace.map(|brace| Block { brace, stmts }),
            semi,
        })
    }
}
impl Parse for item::Trait::Type {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let type_: Token![type] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut gens: Generics = input.parse()?;
        let (colon, bounds) = FlexibleItemTy::parse_optional_bounds(input)?;
        let default = FlexibleItemTy::parse_optional_definition(input)?;
        gens.where_ = input.parse()?;
        let semi: Token![;] = input.parse()?;
        Ok(item::Trait::Type {
            attrs,
            type_,
            ident,
            gens,
            colon,
            bounds,
            default,
            semi,
        })
    }
}
fn parse_trait_item_type(begin: ParseBuffer, input: ParseStream) -> Result<item::Trait::Item> {
    let FlexibleItemType {
        vis,
        default_: _,
        type_,
        ident,
        gens,
        colon,
        bounds,
        ty,
        semi,
    } = FlexibleItemTy::parse(input, TypeDefaultness::Disallowed, WhereClauseLocation::AfterEq)?;
    if vis.is_some() {
        Ok(item::Trait::Item::Verbatim(verbatim_between(&begin, input)))
    } else {
        Ok(item::Trait::Item::Type(item::Trait::Type {
            attrs: Vec::new(),
            type_,
            ident,
            gens,
            colon,
            bounds,
            default: ty,
            semi,
        }))
    }
}
impl Parse for item::Trait::Mac {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(item::Trait::Mac { attrs, mac, semi })
    }
}
impl Parse for item::Impl {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_verbatim_impl = false;
        parse_impl(input, allow_verbatim_impl).map(Option::unwrap)
    }
}
fn parse_impl(input: ParseStream, allow_verbatim_impl: bool) -> Result<Option<item::Impl>> {
    let mut attrs = input.call(Attribute::parse_outer)?;
    let has_visibility = allow_verbatim_impl && input.parse::<Visibility>()?.is_some();
    let default_: Option<Token![default]> = input.parse()?;
    let unsafety: Option<Token![unsafe]> = input.parse()?;
    let impl_: Token![impl] = input.parse()?;
    let has_generics = input.peek(Token![<])
        && (input.peek2(Token![>])
            || input.peek2(Token![#])
            || (input.peek2(Ident) || input.peek2(Lifetime))
                && (input.peek3(Token![:])
                    || input.peek3(Token![,])
                    || input.peek3(Token![>])
                    || input.peek3(Token![=]))
            || input.peek2(Token![const]));
    let mut gens: Generics = if has_generics {
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
    let mut first_ty: ty::Type = input.parse()?;
    let self_ty: ty::Type;
    let trait_;
    let is_impl_for = input.peek(Token![for]);
    if is_impl_for {
        let for_: Token![for] = input.parse()?;
        let mut first_ty_ref = &first_ty;
        while let ty::Type::Group(ty) = first_ty_ref {
            first_ty_ref = &ty.elem;
        }
        if let ty::Type::Path(ty::Path { qself: None, .. }) = first_ty_ref {
            while let ty::Type::Group(ty) = first_ty {
                first_ty = *ty.elem;
            }
            if let ty::Type::Path(ty::Path { qself: None, path }) = first_ty {
                trait_ = Some((polarity, path, for_));
            } else {
                unreachable!();
            }
        } else if !allow_verbatim_impl {
            return Err(Err::new_spanned(first_ty_ref, "expected trait path"));
        } else {
            trait_ = None;
        }
        self_ty = input.parse()?;
    } else {
        trait_ = None;
        self_ty = if polarity.is_none() {
            first_ty
        } else {
            ty::Type::Verbatim(verbatim_between(&begin, input))
        };
    }
    gens.where_ = input.parse()?;
    let content;
    let brace = braced!(content in input);
    parse_inner(&content, &mut attrs)?;
    let mut items = Vec::new();
    while !content.is_empty() {
        items.push(content.parse()?);
    }
    if has_visibility || is_const_impl || is_impl_for && trait_.is_none() {
        Ok(None)
    } else {
        Ok(Some(item::Impl {
            attrs,
            default_,
            unsafe_: unsafety,
            impl_,
            gens,
            trait_,
            typ: Box::new(self_ty),
            brace,
            items,
        }))
    }
}
impl Parse for item::Impl::Item {
    fn parse(x: ParseStream) -> Result<Self> {
        let begin = x.fork();
        let mut attrs = x.call(Attribute::parse_outer)?;
        let ahead = x.fork();
        let vis: Visibility = ahead.parse()?;
        let mut look = ahead.lookahead1();
        let default_ = if look.peek(Token![default]) && !ahead.peek2(Token![!]) {
            let default_: Token![default] = ahead.parse()?;
            look = ahead.lookahead1();
            Some(default_)
        } else {
            None
        };
        let mut item = if look.peek(Token![fn]) || peek_signature(&ahead) {
            let allow_omitted_body = true;
            if let Some(item) = parse_impl_item_fn(x, allow_omitted_body)? {
                Ok(item::Impl::Item::Fn(item))
            } else {
                Ok(item::Impl::Item::Verbatim(verbatim_between(&begin, x)))
            }
        } else if look.peek(Token![const]) {
            x.advance_to(&ahead);
            let const_: Token![const] = x.parse()?;
            let lookahead = x.lookahead1();
            let ident = if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                x.call(Ident::parse_any)?
            } else {
                return Err(lookahead.error());
            };
            let colon: Token![:] = x.parse()?;
            let typ: ty::Type = x.parse()?;
            if let Some(eq) = x.parse()? {
                return Ok(item::Impl::Item::Const(item::Impl::Const {
                    attrs,
                    vis,
                    default_,
                    const_,
                    ident,
                    gens: Generics::default(),
                    colon,
                    typ,
                    eq,
                    expr: x.parse()?,
                    semi: x.parse()?,
                }));
            } else {
                x.parse::<Token![;]>()?;
                return Ok(item::Impl::Item::Verbatim(verbatim_between(&begin, x)));
            }
        } else if look.peek(Token![type]) {
            parse_impl_item_type(begin, x)
        } else if vis.is_inherited()
            && default_.is_none()
            && (look.peek(Ident)
                || look.peek(Token![self])
                || look.peek(Token![super])
                || look.peek(Token![crate])
                || look.peek(Token![::]))
        {
            x.parse().map(item::Impl::Item::Macro)
        } else {
            Err(look.error())
        }?;
        {
            let item_attrs = match &mut item {
                item::Impl::Item::Const(item) => &mut item.attrs,
                item::Impl::Item::Fn(item) => &mut item.attrs,
                item::Impl::Item::Type(item) => &mut item.attrs,
                item::Impl::Item::Macro(item) => &mut item.attrs,
                item::Impl::Item::Verbatim(_) => return Ok(item),
            };
            attrs.append(item_attrs);
            *item_attrs = attrs;
        }
        Ok(item)
    }
}
impl Parse for item::Impl::Const {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(item::Impl::Const {
            attrs: input.call(Attribute::parse_outer)?,
            vis: input.parse()?,
            default_: input.parse()?,
            const_: input.parse()?,
            ident: {
                let lookahead = input.lookahead1();
                if lookahead.peek(Ident) || lookahead.peek(Token![_]) {
                    input.call(Ident::parse_any)?
                } else {
                    return Err(lookahead.error());
                }
            },
            gens: Generics::default(),
            colon: input.parse()?,
            typ: input.parse()?,
            eq: input.parse()?,
            expr: input.parse()?,
            semi: input.parse()?,
        })
    }
}
impl Parse for item::Impl::Fn {
    fn parse(input: ParseStream) -> Result<Self> {
        let allow_omitted_body = false;
        parse_impl_item_fn(input, allow_omitted_body).map(Option::unwrap)
    }
}
fn parse_impl_item_fn(input: ParseStream, allow_omitted_body: bool) -> Result<Option<item::Impl::Fn>> {
    let mut attrs = input.call(Attribute::parse_outer)?;
    let vis: Visibility = input.parse()?;
    let default_: Option<Token![default]> = input.parse()?;
    let sig: item::Sig = input.parse()?;
    if allow_omitted_body && input.parse::<Option<Token![;]>>()?.is_some() {
        return Ok(None);
    }
    let content;
    let brace = braced!(content in input);
    attrs.extend(content.call(Attribute::parse_inner)?);
    let block = Block {
        brace,
        stmts: content.call(Block::parse_within)?,
    };
    Ok(Some(item::Impl::Fn {
        attrs,
        vis,
        default_,
        sig,
        block,
    }))
}
impl Parse for item::Impl::Type {
    fn parse(x: ParseStream) -> Result<Self> {
        let attrs = x.call(Attribute::parse_outer)?;
        let vis: Visibility = x.parse()?;
        let default_: Option<Token![default]> = x.parse()?;
        let type_: Token![type] = x.parse()?;
        let ident: Ident = x.parse()?;
        let mut gens: Generics = x.parse()?;
        let eq: Token![=] = x.parse()?;
        let typ: ty::Type = x.parse()?;
        gens.where_ = x.parse()?;
        let semi: Token![;] = x.parse()?;
        Ok(item::Impl::Type {
            attrs,
            vis,
            default_,
            type_,
            ident,
            gens,
            eq,
            typ,
            semi,
        })
    }
}
fn parse_impl_item_type(begin: ParseBuffer, input: ParseStream) -> Result<item::Impl::Item> {
    let FlexibleItemType {
        vis,
        default_,
        type_,
        ident,
        gens,
        colon,
        bounds: _,
        ty,
        semi,
    } = FlexibleItemTy::parse(input, TypeDefaultness::Optional, WhereClauseLocation::AfterEq)?;
    let (eq, ty) = match ty {
        Some(ty) if colon.is_none() => ty,
        _ => return Ok(item::Impl::Item::Verbatim(verbatim_between(&begin, input))),
    };
    Ok(item::Impl::Item::Type(item::Impl::Type {
        attrs: Vec::new(),
        vis,
        default_,
        type_,
        ident,
        gens,
        eq,
        typ: ty,
        semi,
    }))
}
impl Parse for item::Impl::Mac {
    fn parse(input: ParseStream) -> Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let mac: Macro = input.parse()?;
        let semi: Option<Token![;]> = if mac.delim.is_brace() {
            None
        } else {
            Some(input.parse()?)
        };
        Ok(item::Impl::Mac { attrs, mac, semi })
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
impl MacroDelim {
    pub fn is_brace(&self) -> bool {
        match self {
            MacroDelim::Brace(_) => true,
            MacroDelim::Paren(_) | MacroDelim::Bracket(_) => false,
        }
    }
}
impl Parse for StaticMut {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut_token: Option<Token![mut]> = input.parse()?;
        Ok(mut_token.map_or(StaticMut::None, StaticMut::Mut))
    }
}

mod parsing {
    struct AllowNoSemi(bool);
    impl Block {
        pub fn parse_within(x: ParseStream) -> Result<Vec<stmt::Stmt>> {
            let mut ys = Vec::new();
            loop {
                while let semi @ Some(_) = x.parse()? {
                    ys.push(stmt::Stmt::Expr(Expr::Verbatim(TokenStream::new()), semi));
                }
                if x.is_empty() {
                    break;
                }
                let stmt = parse_stmt(x, AllowNoSemi(true))?;
                let requires_semicolon = match &stmt {
                    stmt::Stmt::Expr(x, None) => expr::requires_terminator(x),
                    stmt::Stmt::Macro(x) => x.semi.is_none() && !x.mac.delimiter.is_brace(),
                    stmt::Stmt::stmt::Local(_) | stmt::Stmt::Item(_) | stmt::Stmt::Expr(_, Some(_)) => false,
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
                brace: braced!(content in x),
                stmts: content.call(Block::parse_within)?,
            })
        }
    }
    impl Parse for stmt::Stmt {
        fn parse(x: ParseStream) -> Result<Self> {
            let allow_nosemi = AllowNoSemi(false);
            parse_stmt(x, allow_nosemi)
        }
    }
    fn parse_stmt(x: ParseStream, allow_nosemi: AllowNoSemi) -> Result<stmt::Stmt> {
        let begin = x.fork();
        let attrs = x.call(Attribute::parse_outer)?;
        let ahead = x.fork();
        let mut is_item_macro = false;
        if let Ok(path) = ahead.call(Path::parse_mod_style) {
            if ahead.peek(Token![!]) {
                if ahead.peek2(Ident) || ahead.peek2(Token![try]) {
                    is_item_macro = true;
                } else if ahead.peek2(tok::Brace) && !(ahead.peek3(Token![.]) || ahead.peek3(Token![?])) {
                    x.advance_to(&ahead);
                    return stmt_mac(x, attrs, path).map(stmt::Stmt::Macro);
                }
            }
        }
        if x.peek(Token![let]) {
            stmt_local(x, attrs).map(stmt::Stmt::stmt::Local)
        } else if x.peek(Token![pub])
            || x.peek(Token![crate]) && !x.peek2(Token![::])
            || x.peek(Token![extern])
            || x.peek(Token![use])
            || x.peek(Token![static])
                && (x.peek2(Token![mut])
                    || x.peek2(Ident) && !(x.peek2(Token![async]) && (x.peek3(Token![move]) || x.peek3(Token![|]))))
            || x.peek(Token![const])
                && !(x.peek2(tok::Brace)
                    || x.peek2(Token![static])
                    || x.peek2(Token![async])
                        && !(x.peek3(Token![unsafe]) || x.peek3(Token![extern]) || x.peek3(Token![fn]))
                    || x.peek2(Token![move])
                    || x.peek2(Token![|]))
            || x.peek(Token![unsafe]) && !x.peek2(tok::Brace)
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
            Ok(stmt::Stmt::Item(item))
        } else {
            stmt_expr(x, allow_nosemi, attrs)
        }
    }
    fn stmt_mac(x: ParseStream, attrs: Vec<Attribute>, path: Path) -> Result<stmt::Mac> {
        let bang: Token![!] = x.parse()?;
        let (delimiter, tokens) = mac_parse_delimiter(x)?;
        let semi: Option<Token![;]> = x.parse()?;
        Ok(stmt::Mac {
            attrs,
            mac: Macro {
                path,
                bang,
                delimiter,
                tokens,
            },
            semi,
        })
    }
    fn stmt_local(x: ParseStream, attrs: Vec<Attribute>) -> Result<stmt::Local> {
        let let_: Token![let] = x.parse()?;
        let mut pat = patt::Patt::parse_single(x)?;
        if x.peek(Token![:]) {
            let colon: Token![:] = x.parse()?;
            let ty: Type = x.parse()?;
            pat = patt::Patt::Type(patt::Type {
                attrs: Vec::new(),
                pat: Box::new(pat),
                colon,
                ty: Box::new(ty),
            });
        }
        let init = if let Some(eq) = x.parse()? {
            let eq: Token![=] = eq;
            let expr: Expr = x.parse()?;
            let diverge = if let Some(else_token) = x.parse()? {
                let else_token: Token![else] = else_token;
                let diverge = expr::Block {
                    attrs: Vec::new(),
                    label: None,
                    block: x.parse()?,
                };
                Some((else_token, Box::new(Expr::Block(diverge))))
            } else {
                None
            };
            Some(stmt::LocalInit {
                eq,
                expr: Box::new(expr),
                diverge,
            })
        } else {
            None
        };
        let semi: Token![;] = x.parse()?;
        Ok(stmt::Local {
            attrs,
            let_,
            pat,
            init,
            semi,
        })
    }
    fn stmt_expr(x: ParseStream, allow_nosemi: AllowNoSemi, mut attrs: Vec<Attribute>) -> Result<stmt::Stmt> {
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
        let semi: Option<Token![;]> = x.parse()?;
        match e {
            Expr::Macro(expr::Mac { attrs, mac }) if semi.is_some() || mac.delimiter.is_brace() => {
                return Ok(stmt::Stmt::Macro(stmt::Mac { attrs, mac, semi }));
            },
            _ => {},
        }
        if semi.is_some() {
            Ok(stmt::Stmt::Expr(e, semi))
        } else if allow_nosemi.0 || !expr::requires_terminator(&e) {
            Ok(stmt::Stmt::Expr(e, None))
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
            let fields = if input.peek(tok::Brace) {
                Fields::Named(input.parse()?)
            } else if input.peek(tok::Paren) {
                Fields::Unnamed(input.parse()?)
            } else {
                Fields::Unit
            };
            let discriminant = if input.peek(Token![=]) {
                let eq: Token![=] = input.parse()?;
                let discriminant: Expr = input.parse()?;
                Some((eq, discriminant))
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
                brace: braced!(content in input),
                named: content.parse_terminated(Field::parse_named, Token![,])?,
            })
        }
    }
    impl Parse for FieldsUnnamed {
        fn parse(input: ParseStream) -> Result<Self> {
            let content;
            Ok(FieldsUnnamed {
                paren: parenthesized!(content in input),
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
                colon: Some(input.parse()?),
                ty: input.parse()?,
            })
        }
        pub fn parse_unnamed(input: ParseStream) -> Result<Self> {
            Ok(Field {
                attrs: input.call(Attribute::parse_outer)?,
                vis: input.parse()?,
                mutability: FieldMutability::None,
                ident: None,
                colon: None,
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
                let struct_ = input.parse::<Token![struct]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<Generics>()?;
                let (where_clause, fields, semi) = data_struct(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: Generics { where_clause, ..gens },
                    data: Data::Struct(DataStruct { struct_, fields, semi }),
                })
            } else if lookahead.peek(Token![enum]) {
                let enum_ = input.parse::<Token![enum]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<Generics>()?;
                let (where_clause, brace, variants) = data_enum(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: Generics { where_clause, ..gens },
                    data: Data::Enum(DataEnum { enum_, brace, variants }),
                })
            } else if lookahead.peek(Token![union]) {
                let union_ = input.parse::<Token![union]>()?;
                let ident = input.parse::<Ident>()?;
                let gens = input.parse::<Generics>()?;
                let (where_clause, fields) = data_union(input)?;
                Ok(DeriveInput {
                    attrs,
                    vis,
                    ident,
                    gens: Generics { where_clause, ..gens },
                    data: Data::Union(DataUnion { union_, fields }),
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
        if where_clause.is_none() && lookahead.peek(tok::Paren) {
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
        } else if lookahead.peek(tok::Brace) {
            let fields = input.parse()?;
            Ok((where_clause, Fields::Named(fields), None))
        } else if lookahead.peek(Token![;]) {
            let semi = input.parse()?;
            Ok((where_clause, Fields::Unit, Some(semi)))
        } else {
            Err(lookahead.error())
        }
    }
    pub fn data_enum(input: ParseStream) -> Result<(Option<WhereClause>, tok::Brace, Punctuated<Variant, Token![,]>)> {
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
                bang: input.parse()?,
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
            if input.peek(tok::Group) {
                let ahead = input.fork();
                let group = super::parse_group(&ahead)?;
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
            let pub_ = input.parse::<Token![pub]>()?;
            if input.peek(tok::Paren) {
                let ahead = input.fork();
                let content;
                let paren = parenthesized!(content in ahead);
                if content.peek(Token![crate]) || content.peek(Token![self]) || content.peek(Token![super]) {
                    let path = content.call(Ident::parse_any)?;
                    if content.is_empty() {
                        input.advance_to(&ahead);
                        return Ok(Visibility::Restricted(VisRestricted {
                            pub_,
                            paren,
                            in_: None,
                            path: Box::new(Path::from(path)),
                        }));
                    }
                } else if content.peek(Token![in]) {
                    let in_: Token![in] = content.parse()?;
                    let path = content.call(Path::parse_mod_style)?;
                    input.advance_to(&ahead);
                    return Ok(Visibility::Restricted(VisRestricted {
                        pub_,
                        paren,
                        in_: Some(in_),
                        path: Box::new(path),
                    }));
                }
            }
            Ok(Visibility::Public(pub_))
        }
        pub fn is_some(&self) -> bool {
            match self {
                Visibility::Inherited => false,
                _ => true,
            }
        }
    }
}

pub mod lit {
    use crate::lit::*;
    impl Parse for Lit {
        fn parse(input: ParseStream) -> Result<Self> {
            input.step(|cursor| {
                if let Some((lit, rest)) = cursor.literal() {
                    return Ok((Lit::new(lit), rest));
                }
                if let Some((ident, rest)) = cursor.ident() {
                    let value = ident == "true";
                    if value || ident == "false" {
                        let lit_bool = lit::Bool {
                            value,
                            span: ident.span(),
                        };
                        return Ok((Lit::Bool(lit_bool), rest));
                    }
                }
                if let Some((punct, rest)) = cursor.punct() {
                    if punct.as_char() == '-' {
                        if let Some((lit, rest)) = parse_negative(punct, rest) {
                            return Ok((lit, rest));
                        }
                    }
                }
                Err(cursor.error("expected literal"))
            })
        }
    }
    fn parse_negative(neg: Punct, cursor: Cursor) -> Option<(Lit, Cursor)> {
        let (lit, rest) = cursor.literal()?;
        let mut span = neg.span();
        span = span.join(lit.span()).unwrap_or(span);
        let mut repr = lit.to_string();
        repr.insert(0, '-');
        if let Some((digits, suffix)) = super::lit::parse_int(&repr) {
            let mut token: Literal = repr.parse().unwrap();
            token.set_span(span);
            return Some((
                Lit::Int(lit::Int {
                    repr: Box::new(lit::IntRepr { token, digits, suffix }),
                }),
                rest,
            ));
        }
        let (digits, suffix) = super::lit::parse_float(&repr)?;
        let mut token: Literal = repr.parse().unwrap();
        token.set_span(span);
        Some((
            Lit::Float(lit::Float {
                repr: Box::new(lit::FloatRepr { token, digits, suffix }),
            }),
            rest,
        ))
    }
    impl Parse for Str {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Str(x)) => Ok(x),
                _ => Err(head.error("expected string literal")),
            }
        }
    }
    impl Parse for ByteStr {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::ByteStr(x)) => Ok(x),
                _ => Err(head.error("expected byte string literal")),
            }
        }
    }
    impl Parse for Byte {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Byte(x)) => Ok(x),
                _ => Err(head.error("expected byte literal")),
            }
        }
    }
    impl Parse for Char {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Char(x)) => Ok(x),
                _ => Err(head.error("expected character literal")),
            }
        }
    }
    impl Parse for Int {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Int(x)) => Ok(x),
                _ => Err(head.error("expected integer literal")),
            }
        }
    }
    impl Parse for Float {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Float(x)) => Ok(x),
                _ => Err(head.error("expected floating point literal")),
            }
        }
    }
    impl Parse for Bool {
        fn parse(x: ParseStream) -> Result<Self> {
            let head = x.fork();
            match x.parse() {
                Ok(Lit::Bool(x)) => Ok(x),
                _ => Err(head.error("expected boolean literal")),
            }
        }
    }
}

pub mod patt {
    use crate::patt::*;
    impl Patt {
        pub fn parse_single(x: ParseStream) -> Result<Self> {
            let begin = x.fork();
            let look = x.lookahead1();
            if look.peek(Ident)
                && (x.peek2(Token![::])
                    || x.peek2(Token![!])
                    || x.peek2(tok::Brace)
                    || x.peek2(tok::Paren)
                    || x.peek2(Token![..]))
                || x.peek(Token![self]) && x.peek2(Token![::])
                || look.peek(Token![::])
                || look.peek(Token![<])
                || x.peek(Token![Self])
                || x.peek(Token![super])
                || x.peek(Token![crate])
            {
                path_or_mac_or_struct_or_range(x)
            } else if look.peek(Token![_]) {
                x.call(wild).map(Patt::Wild)
            } else if x.peek(Token![box]) {
                verbatim(begin, x)
            } else if x.peek(Token![-]) || look.peek(Lit) || look.peek(Token![const]) {
                lit_or_range(x)
            } else if look.peek(Token![ref]) || look.peek(Token![mut]) || x.peek(Token![self]) || x.peek(Ident) {
                x.call(ident).map(Patt::Ident)
            } else if look.peek(Token![&]) {
                x.call(ref_).map(Patt::Ref)
            } else if look.peek(tok::Paren) {
                x.call(paren_or_tuple)
            } else if look.peek(tok::Bracket) {
                x.call(slice).map(Patt::Slice)
            } else if look.peek(Token![..]) && !x.peek(Token![...]) {
                range_half_open(x)
            } else if look.peek(Token![const]) {
                x.call(const_).map(Patt::Verbatim)
            } else {
                Err(look.error())
            }
        }
        pub fn parse_multi(x: ParseStream) -> Result<Self> {
            multi_impl(x, None)
        }
        pub fn parse_with_vert(x: ParseStream) -> Result<Self> {
            let vert: Option<Token![|]> = x.parse()?;
            multi_impl(x, vert)
        }
    }
    fn multi_impl(x: ParseStream, vert: Option<Token![|]>) -> Result<Patt> {
        let mut y = Patt::parse_single(x)?;
        if vert.is_some() || x.peek(Token![|]) && !x.peek(Token![||]) && !x.peek(Token![|=]) {
            let mut cases = Punctuated::new();
            cases.push_value(y);
            while x.peek(Token![|]) && !x.peek(Token![||]) && !x.peek(Token![|=]) {
                let punct = x.parse()?;
                cases.push_punct(punct);
                let pat = Patt::parse_single(x)?;
                cases.push_value(pat);
            }
            y = Patt::Or(Or {
                attrs: Vec::new(),
                vert,
                cases,
            });
        }
        Ok(y)
    }
    fn path_or_mac_or_struct_or_range(x: ParseStream) -> Result<Patt> {
        let (qself, path) = qpath(x, true)?;
        if qself.is_none() && x.peek(Token![!]) && !x.peek(Token![!=]) && path.is_mod_style() {
            let bang: Token![!] = x.parse()?;
            let (delimiter, tokens) = mac_parse_delimiter(x)?;
            return Ok(Patt::Macro(expr::Mac {
                attrs: Vec::new(),
                mac: Macro {
                    path,
                    bang,
                    delimiter,
                    tokens,
                },
            }));
        }
        if x.peek(tok::Brace) {
            struct_(x, qself, path).map(Patt::Struct)
        } else if x.peek(tok::Paren) {
            tuple_struct(x, qself, path).map(Patt::TupleStruct)
        } else if x.peek(Token![..]) {
            range(x, qself, path)
        } else {
            Ok(Patt::Path(expr::Path {
                attrs: Vec::new(),
                qself,
                path,
            }))
        }
    }
    fn wild(x: ParseStream) -> Result<Wild> {
        Ok(Wild {
            attrs: Vec::new(),
            underscore: x.parse()?,
        })
    }
    fn verbatim(beg: ParseBuffer, x: ParseStream) -> Result<Patt> {
        x.parse::<Token![box]>()?;
        Patt::parse_single(x)?;
        Ok(Patt::Verbatim(verbatim_between(&beg, x)))
    }
    fn ident(x: ParseStream) -> Result<Ident> {
        Ok(Ident {
            attrs: Vec::new(),
            ref_: x.parse()?,
            mut_: x.parse()?,
            ident: x.call(Ident::parse_any)?,
            sub: {
                if x.peek(Token![@]) {
                    let at_: Token![@] = x.parse()?;
                    let sub = Patt::parse_single(x)?;
                    Some((at_, Box::new(sub)))
                } else {
                    None
                }
            },
        })
    }
    fn tuple_struct(x: ParseStream, qself: Option<QSelf>, path: Path) -> Result<TupleStruct> {
        let gist;
        let paren = parenthesized!(gist in x);
        let mut patts = Punctuated::new();
        while !gist.is_empty() {
            let value = Patt::parse_multi(&gist)?;
            patts.push_value(value);
            if gist.is_empty() {
                break;
            }
            let punct = gist.parse()?;
            patts.push_punct(punct);
        }
        Ok(TupleStruct {
            attrs: Vec::new(),
            qself,
            path,
            paren,
            elems: patts,
        })
    }
    fn struct_(x: ParseStream, qself: Option<QSelf>, path: Path) -> Result<Struct> {
        let gist;
        let brace = braced!(gist in x);
        let mut fields = Punctuated::new();
        let mut rest = None;
        while !gist.is_empty() {
            let attrs = gist.call(Attribute::parse_outer)?;
            if gist.peek(Token![..]) {
                rest = Some(Rest {
                    attrs,
                    dot2: gist.parse()?,
                });
                break;
            }
            let mut y = gist.call(field)?;
            y.attrs = attrs;
            fields.push_value(y);
            if gist.is_empty() {
                break;
            }
            let y: Token![,] = gist.parse()?;
            fields.push_punct(y);
        }
        Ok(Struct {
            attrs: Vec::new(),
            qself,
            path,
            brace,
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
    fn field(x: ParseStream) -> Result<Field> {
        let beg = x.fork();
        let box_: Option<Token![box]> = x.parse()?;
        let ref_: Option<Token![ref]> = x.parse()?;
        let mut_: Option<Token![mut]> = x.parse()?;
        let member = if box_.is_some() || ref_.is_some() || mut_.is_some() {
            x.parse().map(Member::Named)
        } else {
            x.parse()
        }?;
        if box_.is_none() && ref_.is_none() && mut_.is_none() && x.peek(Token![:]) || member.is_unnamed() {
            return Ok(Field {
                attrs: Vec::new(),
                member,
                colon: Some(x.parse()?),
                patt: Box::new(Patt::parse_multi(x)?),
            });
        }
        let ident = match member {
            Member::Named(ident) => ident,
            Member::Unnamed(_) => unreachable!(),
        };
        let pat = if box_.is_some() {
            Patt::Verbatim(verbatim_between(&beg, x))
        } else {
            Patt::Ident(Ident {
                attrs: Vec::new(),
                ref_,
                mut_,
                ident: ident.clone(),
                sub: None,
            })
        };
        Ok(Field {
            attrs: Vec::new(),
            member: Member::Named(ident),
            colon: None,
            patt: Box::new(pat),
        })
    }
    fn range(x: ParseStream, qself: Option<QSelf>, path: Path) -> Result<Patt> {
        let limits = RangeLimits::parse_obsolete(x)?;
        let end = x.call(range_bound)?;
        if let (RangeLimits::Closed(_), None) = (&limits, &end) {
            return Err(x.error("expected range upper bound"));
        }
        Ok(Patt::Range(expr::Range {
            attrs: Vec::new(),
            start: Some(Box::new(Expr::Path(expr::Path {
                attrs: Vec::new(),
                qself,
                path,
            }))),
            limits,
            end: end.map(RangeBound::into_expr),
        }))
    }
    fn range_half_open(x: ParseStream) -> Result<Patt> {
        let limits: RangeLimits = x.parse()?;
        let end = x.call(range_bound)?;
        if end.is_some() {
            Ok(Patt::Range(expr::Range {
                attrs: Vec::new(),
                start: None,
                limits,
                end: end.map(RangeBound::into_expr),
            }))
        } else {
            match limits {
                RangeLimits::HalfOpen(dot2) => Ok(Patt::Rest(Rest {
                    attrs: Vec::new(),
                    dot2,
                })),
                RangeLimits::Closed(_) => Err(x.error("expected range upper bound")),
            }
        }
    }
    fn paren_or_tuple(x: ParseStream) -> Result<Patt> {
        let gist;
        let paren = parenthesized!(gist in x);
        let mut patts = Punctuated::new();
        while !gist.is_empty() {
            let x = Patt::parse_multi(&gist)?;
            if gist.is_empty() {
                if patts.is_empty() && !matches!(x, Patt::Rest(_)) {
                    return Ok(Patt::Paren(Paren {
                        attrs: Vec::new(),
                        paren,
                        patt: Box::new(x),
                    }));
                }
                patts.push_value(x);
                break;
            }
            patts.push_value(x);
            let punct = gist.parse()?;
            patts.push_punct(punct);
        }
        Ok(Patt::Tuple(Tuple {
            attrs: Vec::new(),
            paren,
            elems: patts,
        }))
    }
    fn ref_(x: ParseStream) -> Result<Ref> {
        Ok(Ref {
            attrs: Vec::new(),
            and: x.parse()?,
            mut_: x.parse()?,
            patt: Box::new(Patt::parse_single(x)?),
        })
    }
    fn lit_or_range(x: ParseStream) -> Result<Patt> {
        let beg = x.call(range_bound)?.unwrap();
        if x.peek(Token![..]) {
            let limits = RangeLimits::parse_obsolete(x)?;
            let end = x.call(range_bound)?;
            if let (RangeLimits::Closed(_), None) = (&limits, &end) {
                return Err(x.error("expected range upper bound"));
            }
            Ok(Patt::Range(expr::Range {
                attrs: Vec::new(),
                start: Some(beg.into_expr()),
                limits,
                end: end.map(RangeBound::into_expr),
            }))
        } else {
            Ok(beg.into_pat())
        }
    }
    enum RangeBound {
        Const(expr::Const),
        Lit(expr::Lit),
        Path(expr::Path),
    }
    impl RangeBound {
        fn into_expr(self) -> Box<Expr> {
            Box::new(match self {
                RangeBound::Const(x) => Expr::Const(x),
                RangeBound::Lit(x) => Expr::Lit(x),
                RangeBound::Path(x) => Expr::Path(x),
            })
        }
        fn into_pat(self) -> Patt {
            match self {
                RangeBound::Const(x) => Patt::Const(x),
                RangeBound::Lit(x) => Patt::Lit(x),
                RangeBound::Path(x) => Patt::Path(x),
            }
        }
    }
    fn range_bound(x: ParseStream) -> Result<Option<RangeBound>> {
        if x.is_empty()
            || x.peek(Token![|])
            || x.peek(Token![=])
            || x.peek(Token![:]) && !x.peek(Token![::])
            || x.peek(Token![,])
            || x.peek(Token![;])
            || x.peek(Token![if])
        {
            return Ok(None);
        }
        let look = x.lookahead1();
        let y = if look.peek(Lit) {
            RangeBound::Lit(x.parse()?)
        } else if look.peek(Ident)
            || look.peek(Token![::])
            || look.peek(Token![<])
            || look.peek(Token![self])
            || look.peek(Token![Self])
            || look.peek(Token![super])
            || look.peek(Token![crate])
        {
            RangeBound::Path(x.parse()?)
        } else if look.peek(Token![const]) {
            RangeBound::Const(x.parse()?)
        } else {
            return Err(look.error());
        };
        Ok(Some(y))
    }
    fn slice(x: ParseStream) -> Result<Slice> {
        let gist;
        let bracket = bracketed!(gist in x);
        let mut patts = Punctuated::new();
        while !gist.is_empty() {
            let y = Patt::parse_multi(&gist)?;
            match y {
                Patt::Range(x) if x.beg.is_none() || x.end.is_none() => {
                    let (start, end) = match x.limits {
                        RangeLimits::HalfOpen(x) => (x.spans[0], x.spans[1]),
                        RangeLimits::Closed(x) => (x.spans[0], x.spans[2]),
                    };
                    let m = "range pattern is not allowed unparenthesized inside slice pattern";
                    return Err(err::new2(start, end, m));
                },
                _ => {},
            }
            patts.push_value(y);
            if gist.is_empty() {
                break;
            }
            let y = gist.parse()?;
            patts.push_punct(y);
        }
        Ok(Slice {
            attrs: Vec::new(),
            bracket,
            elems: patts,
        })
    }
    fn const_(x: ParseStream) -> Result<TokenStream> {
        let beg = x.fork();
        x.parse::<Token![const]>()?;
        let gist;
        braced!(gist in x);
        gist.call(Attribute::parse_inner)?;
        gist.call(Block::parse_within)?;
        Ok(verbatim_between(&beg, x))
    }
}

pub mod path {
    use crate::path::*;
    impl Path {
        pub fn parse_mod_style(x: ParseStream) -> Result<Self> {
            Ok(Path {
                colon: x.parse()?,
                segs: {
                    let mut ys = Punctuated::new();
                    loop {
                        if !x.peek(Ident)
                            && !x.peek(Token![super])
                            && !x.peek(Token![self])
                            && !x.peek(Token![Self])
                            && !x.peek(Token![crate])
                        {
                            break;
                        }
                        let ident = Ident::parse_any(x)?;
                        ys.push_value(Segment::from(ident));
                        if !x.peek(Token![::]) {
                            break;
                        }
                        let punct = x.parse()?;
                        ys.push_punct(punct);
                    }
                    if ys.is_empty() {
                        return Err(x.parse::<Ident>().unwrap_err());
                    } else if ys.trailing_punct() {
                        return Err(x.error("expected path segment after `::`"));
                    }
                    ys
                },
            })
        }
        pub fn parse_helper(x: ParseStream, expr_style: bool) -> Result<Self> {
            let mut path = Path {
                colon: x.parse()?,
                segs: {
                    let mut ys = Punctuated::new();
                    let value = Segment::parse_helper(x, expr_style)?;
                    ys.push_value(value);
                    ys
                },
            };
            Path::parse_rest(x, &mut path, expr_style)?;
            Ok(path)
        }
        pub fn parse_rest(x: ParseStream, path: &mut Self, expr_style: bool) -> Result<()> {
            while x.peek(Token![::]) && !x.peek3(tok::Paren) {
                let punct: Token![::] = x.parse()?;
                path.segs.push_punct(punct);
                let value = Segment::parse_helper(x, expr_style)?;
                path.segs.push_value(value);
            }
            Ok(())
        }
        pub fn is_mod_style(&self) -> bool {
            self.segs.iter().all(|x| x.arguments.is_none())
        }
    }
    impl Parse for Path {
        fn parse(x: ParseStream) -> Result<Self> {
            Self::parse_helper(x, false)
        }
    }
    impl Segment {
        fn parse_helper(x: ParseStream, expr_style: bool) -> Result<Self> {
            if x.peek(Token![super])
                || x.peek(Token![self])
                || x.peek(Token![crate])
                || cfg!(feature = "full") && x.peek(Token![try])
            {
                let ident = x.call(Ident::parse_any)?;
                return Ok(Segment::from(ident));
            }
            let ident = if x.peek(Token![Self]) {
                x.call(Ident::parse_any)?
            } else {
                x.parse()?
            };
            if !expr_style && x.peek(Token![<]) && !x.peek(Token![<=]) || x.peek(Token![::]) && x.peek3(Token![<]) {
                Ok(Segment {
                    ident,
                    args: Args::Angled(x.parse()?),
                })
            } else {
                Ok(Segment::from(ident))
            }
        }
    }
    impl Parse for Segment {
        fn parse(x: ParseStream) -> Result<Self> {
            Self::parse_helper(x, false)
        }
    }
    impl AngledArgs {
        pub fn parse_turbofish(x: ParseStream) -> Result<Self> {
            let y: Token![::] = x.parse()?;
            Self::do_parse(Some(y), x)
        }
        fn do_parse(colon2: Option<Token![::]>, x: ParseStream) -> Result<Self> {
            Ok(AngledArgs {
                colon2,
                lt: x.parse()?,
                args: {
                    let mut ys = Punctuated::new();
                    loop {
                        if x.peek(Token![>]) {
                            break;
                        }
                        let y: Arg = x.parse()?;
                        ys.push_value(y);
                        if x.peek(Token![>]) {
                            break;
                        }
                        let y: Token![,] = x.parse()?;
                        ys.push_punct(y);
                    }
                    ys
                },
                gt: x.parse()?,
            })
        }
    }
    impl Parse for AngledArgs {
        fn parse(x: ParseStream) -> Result<Self> {
            let y: Option<Token![::]> = x.parse()?;
            Self::do_parse(y, x)
        }
    }
    impl Parse for ParenthesizedArgs {
        fn parse(x: ParseStream) -> Result<Self> {
            let gist;
            Ok(ParenthesizedArgs {
                paren: parenthesized!(gist in x),
                args: gist.parse_terminated(ty::Type::parse, Token![,])?,
                ret: x.call(ty::Ret::without_plus)?,
            })
        }
    }
    impl Parse for Arg {
        fn parse(x: ParseStream) -> Result<Self> {
            if x.peek(Lifetime) && !x.peek2(Token![+]) {
                return Ok(Arg::Lifetime(x.parse()?));
            }
            if x.peek(Lit) || x.peek(tok::Brace) {
                return const_argument(x).map(Arg::Const);
            }
            let mut y: Type = x.parse()?;
            match y {
                ty::Type::Path(mut ty)
                    if ty.qself.is_none()
                        && ty.path.colon.is_none()
                        && ty.path.segs.len() == 1
                        && match &ty.path.segments[0].arguments {
                            Args::None | Args::AngleBracketed(_) => true,
                            Args::Parenthesized(_) => false,
                        } =>
                {
                    if let Some(eq) = x.parse::<Option<Token![=]>>()? {
                        let seg = ty.path.segs.pop().unwrap().into_value();
                        let ident = seg.ident;
                        let gnrs = match seg.args {
                            Args::None => None,
                            Args::Angled(x) => Some(x),
                            Args::Parenthesized(_) => unreachable!(),
                        };
                        return if x.peek(Lit) || x.peek(tok::Brace) {
                            Ok(Arg::AssocConst(AssocConst {
                                ident,
                                args: gnrs,
                                eq,
                                val: const_argument(x)?,
                            }))
                        } else {
                            Ok(Arg::AssocType(AssocType {
                                ident,
                                args: gnrs,
                                eq,
                                typ: x.parse()?,
                            }))
                        };
                    }
                    if let Some(colon) = x.parse::<Option<Token![:]>>()? {
                        let seg = ty.path.segs.pop().unwrap().into_value();
                        return Ok(Arg::Constraint(Constraint {
                            ident: seg.ident,
                            args: match seg.args {
                                Args::None => None,
                                Args::Angled(x) => Some(x),
                                Args::Parenthesized(_) => unreachable!(),
                            },
                            colon,
                            bounds: {
                                let mut ys = Punctuated::new();
                                loop {
                                    if x.peek(Token![,]) || x.peek(Token![>]) {
                                        break;
                                    }
                                    let y: TypeParamBound = x.parse()?;
                                    ys.push_value(y);
                                    if !x.peek(Token![+]) {
                                        break;
                                    }
                                    let y: Token![+] = x.parse()?;
                                    ys.push_punct(y);
                                }
                                ys
                            },
                        }));
                    }
                    y = ty::Type::Path(ty);
                },
                _ => {},
            }
            Ok(Arg::Type(y))
        }
    }
    pub fn const_argument(x: ParseStream) -> Result<Expr> {
        let y = x.lookahead1();
        if x.peek(Lit) {
            let y = x.parse()?;
            return Ok(Expr::Lit(y));
        }
        if x.peek(Ident) {
            let y: Ident = x.parse()?;
            return Ok(Expr::Path(expr::Path {
                attrs: Vec::new(),
                qself: None,
                path: Path::from(y),
            }));
        }
        if x.peek(tok::Brace) {
            let y: expr::Block = x.parse()?;
            return Ok(Expr::Block(y));
        }
        Err(y.error())
    }
    pub fn qpath(x: ParseStream, expr_style: bool) -> Result<(Option<QSelf>, Path)> {
        if x.peek(Token![<]) {
            let lt: Token![<] = x.parse()?;
            let this: Type = x.parse()?;
            let y = if x.peek(Token![as]) {
                let as_: Token![as] = x.parse()?;
                let y: Path = x.parse()?;
                Some((as_, y))
            } else {
                None
            };
            let gt: Token![>] = x.parse()?;
            let colon2: Token![::] = x.parse()?;
            let mut rest = Punctuated::new();
            loop {
                let path = Segment::parse_helper(x, expr_style)?;
                rest.push_value(path);
                if !x.peek(Token![::]) {
                    break;
                }
                let punct: Token![::] = x.parse()?;
                rest.push_punct(punct);
            }
            let (pos, as_, y) = match y {
                Some((as_, mut y)) => {
                    let pos = y.segs.len();
                    y.segs.push_punct(colon2);
                    y.segs.extend(rest.into_pairs());
                    (pos, Some(as_), y)
                },
                None => {
                    let y = Path {
                        colon: Some(colon2),
                        segs: rest,
                    };
                    (0, None, y)
                },
            };
            let qself = QSelf {
                lt,
                typ: Box::new(this),
                pos,
                as_,
                gt,
            };
            Ok((Some(qself), y))
        } else {
            let y = Path::parse_helper(x, expr_style)?;
            Ok((None, y))
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

pub mod ty {
    use crate::{tok, ty::*};
    impl Parse for Type {
        fn parse(x: ParseStream) -> Result<Self> {
            let plus = true;
            let group_gen = true;
            ambig_ty(x, plus, group_gen)
        }
    }
    impl Type {
        pub fn without_plus(x: ParseStream) -> Result<Self> {
            let plus = false;
            let group_gen = true;
            ambig_ty(x, plus, group_gen)
        }
    }
    pub fn ambig_ty(x: ParseStream, allow_plus: bool, group_gen: bool) -> Result<Type> {
        let begin = x.fork();
        if x.peek(tok::Group) {
            let mut y: Group = x.parse()?;
            if x.peek(Token![::]) && x.peek3(Ident::peek_any) {
                if let Type::Path(mut ty) = *y.elem {
                    Path::parse_rest(x, &mut ty.path, false)?;
                    return Ok(Type::Path(ty));
                } else {
                    return Ok(Type::Path(Path {
                        qself: Some(QSelf {
                            lt: Token![<](y.group.span),
                            position: 0,
                            as_: None,
                            gt: Token![>](y.group.span),
                            ty: y.elem,
                        }),
                        path: Path::parse_helper(x, false)?,
                    }));
                }
            } else if x.peek(Token![<]) && group_gen || x.peek(Token![::]) && x.peek3(Token![<]) {
                if let Type::Path(mut ty) = *y.elem {
                    let args = &mut ty.path.segs.last_mut().unwrap().args;
                    if args.is_none() {
                        *args = path::Args::Angled(x.parse()?);
                        Path::parse_rest(x, &mut ty.path, false)?;
                        return Ok(Type::Path(ty));
                    } else {
                        y.elem = Box::new(Type::Path(ty));
                    }
                }
            }
            return Ok(Type::Group(y));
        }
        let mut lifes = None::<BoundLifetimes>;
        let mut look = x.lookahead1();
        if look.peek(Token![for]) {
            lifes = x.parse()?;
            look = x.lookahead1();
            if !look.peek(Ident)
                && !look.peek(Token![fn])
                && !look.peek(Token![unsafe])
                && !look.peek(Token![extern])
                && !look.peek(Token![super])
                && !look.peek(Token![self])
                && !look.peek(Token![Self])
                && !look.peek(Token![crate])
                || x.peek(Token![dyn])
            {
                return Err(look.error());
            }
        }
        if look.peek(tok::Paren) {
            let gist;
            let paren = parenthesized!(gist in x);
            if gist.is_empty() {
                return Ok(Type::Tuple(Tuple {
                    paren,
                    elems: Punctuated::new(),
                }));
            }
            if gist.peek(Lifetime) {
                return Ok(Type::Paren(Paren {
                    paren,
                    elem: Box::new(Type::TraitObject(gist.parse()?)),
                }));
            }
            if gist.peek(Token![?]) {
                return Ok(Type::TraitObject(TraitObj {
                    dyn_: None,
                    bounds: {
                        let mut ys = Punctuated::new();
                        ys.push_value(TypeParamBound::Trait(TraitBound {
                            paren: Some(paren),
                            ..gist.parse()?
                        }));
                        while let Some(plus) = x.parse()? {
                            ys.push_punct(plus);
                            ys.push_value(x.parse()?);
                        }
                        ys
                    },
                }));
            }
            let mut first: Type = gist.parse()?;
            if gist.peek(Token![,]) {
                return Ok(Type::Tuple(Tuple {
                    paren,
                    elems: {
                        let mut ys = Punctuated::new();
                        ys.push_value(first);
                        ys.push_punct(gist.parse()?);
                        while !gist.is_empty() {
                            ys.push_value(gist.parse()?);
                            if gist.is_empty() {
                                break;
                            }
                            ys.push_punct(gist.parse()?);
                        }
                        ys
                    },
                }));
            }
            if allow_plus && x.peek(Token![+]) {
                loop {
                    let first = match first {
                        Type::Path(Path { qself: None, path }) => TypeParamBound::Trait(TraitBound {
                            paren: Some(paren),
                            modifier: TraitBoundModifier::None,
                            lifetimes: None,
                            path,
                        }),
                        Type::TraitObject(TraitObj { dyn_: None, bounds }) => {
                            if bounds.len() > 1 || bounds.trailing_punct() {
                                first = Type::TraitObject(TraitObj { dyn_: None, bounds });
                                break;
                            }
                            match bounds.into_iter().next().unwrap() {
                                TypeParamBound::Trait(trait_bound) => TypeParamBound::Trait(TraitBound {
                                    paren: Some(paren),
                                    ..trait_bound
                                }),
                                other @ (TypeParamBound::Lifetime(_) | TypeParamBound::Verbatim(_)) => other,
                            }
                        },
                        _ => break,
                    };
                    return Ok(Type::TraitObject(TraitObj {
                        dyn_: None,
                        bounds: {
                            let mut ys = Punctuated::new();
                            ys.push_value(first);
                            while let Some(plus) = x.parse()? {
                                ys.push_punct(plus);
                                ys.push_value(x.parse()?);
                            }
                            ys
                        },
                    }));
                }
            }
            Ok(Type::Paren(Paren {
                paren,
                elem: Box::new(first),
            }))
        } else if look.peek(Token![fn]) || look.peek(Token![unsafe]) || look.peek(Token![extern]) {
            let mut y: Fn = x.parse()?;
            y.lifes = lifes;
            Ok(Type::Fn(y))
        } else if look.peek(Ident)
            || x.peek(Token![super])
            || x.peek(Token![self])
            || x.peek(Token![Self])
            || x.peek(Token![crate])
            || look.peek(Token![::])
            || look.peek(Token![<])
        {
            let ty: Path = x.parse()?;
            if ty.qself.is_some() {
                return Ok(Type::Path(ty));
            }
            if x.peek(Token![!]) && !x.peek(Token![!=]) && ty.path.is_mod_style() {
                let bang: Token![!] = x.parse()?;
                let (delimiter, tokens) = mac_parse_delimiter(x)?;
                return Ok(Type::Mac(Mac {
                    mac: Macro {
                        path: ty.path,
                        bang,
                        delimiter,
                        tokens,
                    },
                }));
            }
            if lifes.is_some() || allow_plus && x.peek(Token![+]) {
                let mut bounds = Punctuated::new();
                bounds.push_value(TypeParamBound::Trait(TraitBound {
                    paren: None,
                    modifier: TraitBoundModifier::None,
                    lifetimes: lifes,
                    path: ty.path,
                }));
                if allow_plus {
                    while x.peek(Token![+]) {
                        bounds.push_punct(x.parse()?);
                        if !(x.peek(Ident::peek_any)
                            || x.peek(Token![::])
                            || x.peek(Token![?])
                            || x.peek(Lifetime)
                            || x.peek(tok::Paren))
                        {
                            break;
                        }
                        bounds.push_value(x.parse()?);
                    }
                }
                return Ok(Type::TraitObject(TraitObj { dyn_: None, bounds }));
            }
            Ok(Type::Path(ty))
        } else if look.peek(Token![dyn]) {
            let dyn_: Token![dyn] = x.parse()?;
            let dyn_span = dyn_.span;
            let star: Option<Token![*]> = x.parse()?;
            let bounds = TraitObj::parse_bounds(dyn_span, x, allow_plus)?;
            return Ok(if star.is_some() {
                Type::Verbatim(verbatim_between(&begin, x))
            } else {
                Type::TraitObject(TraitObj {
                    dyn_: Some(dyn_),
                    bounds,
                })
            });
        } else if look.peek(tok::Bracket) {
            let gist;
            let bracket = bracketed!(gist in x);
            let elem: Type = gist.parse()?;
            if gist.peek(Token![;]) {
                Ok(Type::Array(Array {
                    bracket,
                    elem: Box::new(elem),
                    semi: gist.parse()?,
                    len: gist.parse()?,
                }))
            } else {
                Ok(Type::Slice(Slice {
                    bracket,
                    elem: Box::new(elem),
                }))
            }
        } else if look.peek(Token![*]) {
            x.parse().map(Type::Ptr)
        } else if look.peek(Token![&]) {
            x.parse().map(Type::Reference)
        } else if look.peek(Token![!]) && !x.peek(Token![=]) {
            x.parse().map(Type::Never)
        } else if look.peek(Token![impl]) {
            Impl::parse(x, allow_plus).map(Type::ImplTrait)
        } else if look.peek(Token![_]) {
            x.parse().map(Type::Infer)
        } else if look.peek(Lifetime) {
            x.parse().map(Type::TraitObject)
        } else {
            Err(look.error())
        }
    }
    impl Parse for Slice {
        fn parse(x: ParseStream) -> Result<Self> {
            let gist;
            Ok(Slice {
                bracket: bracketed!(gist in x),
                elem: gist.parse()?,
            })
        }
    }
    impl Parse for Array {
        fn parse(x: ParseStream) -> Result<Self> {
            let gist;
            Ok(Array {
                bracket: bracketed!(gist in x),
                elem: gist.parse()?,
                semi: gist.parse()?,
                len: gist.parse()?,
            })
        }
    }
    impl Parse for Ptr {
        fn parse(x: ParseStream) -> Result<Self> {
            let star: Token![*] = x.parse()?;
            let look = x.lookahead1();
            let (const_, mut_) = if look.peek(Token![const]) {
                (Some(x.parse()?), None)
            } else if look.peek(Token![mut]) {
                (None, Some(x.parse()?))
            } else {
                return Err(look.error());
            };
            Ok(Ptr {
                star,
                const_,
                mut_,
                elem: Box::new(x.call(Type::without_plus)?),
            })
        }
    }
    impl Parse for Ref {
        fn parse(x: ParseStream) -> Result<Self> {
            Ok(Ref {
                and: x.parse()?,
                life: x.parse()?,
                mut_: x.parse()?,
                elem: Box::new(x.call(Type::without_plus)?),
            })
        }
    }
    impl Parse for Fn {
        fn parse(x: ParseStream) -> Result<Self> {
            let args;
            let mut vari = None;
            Ok(Fn {
                lifes: x.parse()?,
                unsafe_: x.parse()?,
                abi: x.parse()?,
                fn_: x.parse()?,
                paren: parenthesized!(args in x),
                args: {
                    let mut ys = Punctuated::new();
                    while !args.is_empty() {
                        let attrs = args.call(Attribute::parse_outer)?;
                        if ys.empty_or_trailing()
                            && (args.peek(Token![...])
                                || args.peek(Ident) && args.peek2(Token![:]) && args.peek3(Token![...]))
                        {
                            vari = Some(parse_bare_variadic(&args, attrs)?);
                            break;
                        }
                        let allow_self = ys.is_empty();
                        let arg = parse_bare_fn_arg(&args, allow_self)?;
                        ys.push_value(FnArg { attrs, ..arg });
                        if args.is_empty() {
                            break;
                        }
                        let comma = args.parse()?;
                        ys.push_punct(comma);
                    }
                    ys
                },
                vari,
                ret: x.call(Ret::without_plus)?,
            })
        }
    }
    impl Parse for Never {
        fn parse(x: ParseStream) -> Result<Self> {
            Ok(Never { bang: x.parse()? })
        }
    }
    impl Parse for Infer {
        fn parse(x: ParseStream) -> Result<Self> {
            Ok(Infer { underscore: x.parse()? })
        }
    }
    impl Parse for Tuple {
        fn parse(x: ParseStream) -> Result<Self> {
            let gist;
            let paren = parenthesized!(gist in x);
            if gist.is_empty() {
                return Ok(Tuple {
                    paren,
                    elems: Punctuated::new(),
                });
            }
            let first: Type = gist.parse()?;
            Ok(Tuple {
                paren,
                elems: {
                    let mut ys = Punctuated::new();
                    ys.push_value(first);
                    ys.push_punct(gist.parse()?);
                    while !gist.is_empty() {
                        ys.push_value(gist.parse()?);
                        if gist.is_empty() {
                            break;
                        }
                        ys.push_punct(gist.parse()?);
                    }
                    ys
                },
            })
        }
    }
    impl Parse for Mac {
        fn parse(x: ParseStream) -> Result<Self> {
            Ok(Mac { mac: x.parse()? })
        }
    }
    impl Parse for Path {
        fn parse(x: ParseStream) -> Result<Self> {
            let expr_style = false;
            let (qself, path) = qpath(x, expr_style)?;
            Ok(Path { qself, path })
        }
    }
    impl Ret {
        pub fn without_plus(x: ParseStream) -> Result<Self> {
            let plus = false;
            Self::parse(x, plus)
        }
        pub fn parse(x: ParseStream, plus: bool) -> Result<Self> {
            if x.peek(Token![->]) {
                let arrow = x.parse()?;
                let group_gen = true;
                let ty = ambig_ty(x, plus, group_gen)?;
                Ok(Ret::Type(arrow, Box::new(ty)))
            } else {
                Ok(Ret::Default)
            }
        }
    }
    impl Parse for Ret {
        fn parse(x: ParseStream) -> Result<Self> {
            let plus = true;
            Self::parse(x, plus)
        }
    }
    impl Parse for TraitObj {
        fn parse(x: ParseStream) -> Result<Self> {
            let plus = true;
            Self::parse(x, plus)
        }
    }
    impl TraitObj {
        pub fn without_plus(x: ParseStream) -> Result<Self> {
            let plus = false;
            Self::parse(x, plus)
        }
        pub fn parse(x: ParseStream, plus: bool) -> Result<Self> {
            let dyn_: Option<Token![dyn]> = x.parse()?;
            let span = match &dyn_ {
                Some(x) => x.span,
                None => x.span(),
            };
            let bounds = Self::parse_bounds(span, x, plus)?;
            Ok(TraitObj { dyn_, bounds })
        }
        fn parse_bounds(s: Span, x: ParseStream, plus: bool) -> Result<Punctuated<TypeParamBound, Token![+]>> {
            let ys = TypeParamBound::parse_multiple(x, plus)?;
            let mut last = None;
            let mut one = false;
            for y in &ys {
                match y {
                    TypeParamBound::Trait(_) | TypeParamBound::Verbatim(_) => {
                        one = true;
                        break;
                    },
                    TypeParamBound::Lifetime(x) => {
                        last = Some(x.ident.span());
                    },
                }
            }
            if !one {
                let msg = "at least one trait is required for an object type";
                return Err(err::new2(s, last.unwrap(), msg));
            }
            Ok(ys)
        }
    }
    impl Parse for Impl {
        fn parse(x: ParseStream) -> Result<Self> {
            let plus = true;
            Self::parse(x, plus)
        }
    }
    impl Impl {
        pub fn without_plus(x: ParseStream) -> Result<Self> {
            let plus = false;
            Self::parse(x, plus)
        }
        pub fn parse(x: ParseStream, plus: bool) -> Result<Self> {
            let impl_: Token![impl] = x.parse()?;
            let bounds = TypeParamBound::parse_multiple(x, plus)?;
            let mut last = None;
            let mut one = false;
            for x in &bounds {
                match x {
                    TypeParamBound::Trait(_) | TypeParamBound::Verbatim(_) => {
                        one = true;
                        break;
                    },
                    TypeParamBound::Lifetime(x) => {
                        last = Some(x.ident.span());
                    },
                }
            }
            if !one {
                let msg = "at least one trait must be specified";
                return Err(err::new2(impl_.span, last.unwrap(), msg));
            }
            Ok(Impl { impl_, bounds })
        }
    }
    impl Parse for Group {
        fn parse(x: ParseStream) -> Result<Self> {
            let y = super::parse_group(x)?;
            Ok(Group {
                group: y.token,
                elem: y.gist.parse()?,
            })
        }
    }
    impl Parse for Paren {
        fn parse(x: ParseStream) -> Result<Self> {
            let plus = false;
            Self::parse(x, plus)
        }
    }
    impl Paren {
        fn parse(x: ParseStream, plus: bool) -> Result<Self> {
            let gist;
            Ok(Paren {
                paren: parenthesized!(gist in x),
                elem: Box::new({
                    let group_gen = true;
                    ambig_ty(&gist, plus, group_gen)?
                }),
            })
        }
    }
    impl Parse for FnArg {
        fn parse(x: ParseStream) -> Result<Self> {
            let self_ = false;
            parse_bare_fn_arg(x, self_)
        }
    }
    fn parse_bare_fn_arg(x: ParseStream, self_: bool) -> Result<FnArg> {
        let attrs = x.call(Attribute::parse_outer)?;
        let beg = x.fork();
        let has_mut_self = self_ && x.peek(Token![mut]) && x.peek2(Token![self]);
        if has_mut_self {
            x.parse::<Token![mut]>()?;
        }
        let mut has_self = false;
        let mut name = if (x.peek(Ident) || x.peek(Token![_]) || {
            has_self = self_ && x.peek(Token![self]);
            has_self
        }) && x.peek2(Token![:])
            && !x.peek2(Token![::])
        {
            let name = x.call(Ident::parse_any)?;
            let colon: Token![:] = x.parse()?;
            Some((name, colon))
        } else {
            has_self = false;
            None
        };
        let ty = if self_ && !has_self && x.peek(Token![mut]) && x.peek2(Token![self]) {
            x.parse::<Token![mut]>()?;
            x.parse::<Token![self]>()?;
            None
        } else if has_mut_self && name.is_none() {
            x.parse::<Token![self]>()?;
            None
        } else {
            Some(x.parse()?)
        };
        let ty = match ty {
            Some(ty) if !has_mut_self => ty,
            _ => {
                name = None;
                Type::Verbatim(verbatim_between(&beg, x))
            },
        };
        Ok(FnArg { attrs, name, ty })
    }
    fn parse_bare_variadic(x: ParseStream, attrs: Vec<Attribute>) -> Result<Variadic> {
        Ok(Variadic {
            attrs,
            name: if x.peek(Ident) || x.peek(Token![_]) {
                let name = x.call(Ident::parse_any)?;
                let colon: Token![:] = x.parse()?;
                Some((name, colon))
            } else {
                None
            },
            dots: x.parse()?,
            comma: x.parse()?,
        })
    }
    impl Parse for Abi {
        fn parse(x: ParseStream) -> Result<Self> {
            Ok(Abi {
                extern_: x.parse()?,
                name: x.parse()?,
            })
        }
    }
    impl Parse for Option<Abi> {
        fn parse(x: ParseStream) -> Result<Self> {
            if x.peek(Token![extern]) {
                x.parse().map(Some)
            } else {
                Ok(None)
            }
        }
    }
}
