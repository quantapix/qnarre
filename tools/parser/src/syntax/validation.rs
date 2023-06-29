use crate::{
    lexer::unescape::{self, unesc_lit, EscErr, Mode},
    match_ast,
    syntax::{
        self, algo,
        ast::{self, HasAttrs, HasVisibility, IsString},
        core::Direction,
        SyntaxErr, TextSize, T,
    },
    SyntaxKind::{self, CONST, FN, INT_NUMBER, TYPE_ALIAS},
};

pub fn validate(root: &syntax::Node) -> Vec<SyntaxErr> {
    let mut y = Vec::new();
    for x in root.descendants() {
        use ast::*;
        match_ast! {
            match x {
                Literal(x) => literal(x, &mut y),
                Const(x) => validate_const(x, &mut y),
                BlockExpr(x) => block_expr(x, &mut y),
                FieldExpr(x) => num_name(x.name_ref(), &mut y),
                RecordExprField(x) => num_name(x.name_ref(), &mut y),
                Visibility(x) => visibility(x, &mut y),
                RangeExpr(x) => range_expr(x, &mut y),
                PathSegment(x) => validate_path_keywords(x, &mut y),
                RefType(x) => validate_trait_object_ref_ty(x, &mut y),
                PtrType(x) => validate_trait_object_ptr_ty(x, &mut y),
                FnPtrType(x) => validate_trait_object_fn_ptr_ret_ty(x, &mut y),
                MacroRules(x) => mac_rules(x, &mut y),
                LetExpr(x) => let_expr(x, &mut y),
                _ => (),
            }
        }
    }
    y
}

pub fn block_expr(x: ast::BlockExpr, y: &mut Vec<SyntaxErr>) {
    if let Some(x) = x.syntax().parent() {
        use SyntaxKind::*;
        match x.kind() {
            FN | EXPR_STMT | STMT_LIST => return,
            _ => {},
        }
    }
    if let Some(x) = x.stmt_list() {
        y.extend(x.attrs().filter(|x| x.kind().is_inner()).map(|x| {
            SyntaxErr::new(
                "A block in this position cannot accept inner attributes",
                x.syntax().text_range(),
            )
        }));
    }
}

fn unescape_err_to_string(x: EscErr) -> (&'static str, bool) {
    use EscErr::*;
    #[rustfmt::skip]
    let y = match x {
        ZeroChars => {
            "Literal must not be empty"
        }
        ManyChars => {
            "Literal must be one character long"
        }
        OneSlash => {
            "Character must be escaped: `\\`"
        }
        InvalidEsc => {
            "Invalid escape"
        }
        BareCarriageReturn | CarriageReturnInRaw => {
            "Character must be escaped: `\r`"
        }
        EscOnlyChar => {
            "Escape character `\\` must be escaped itself"
        }
        ShortHexEsc => {
            "ASCII hex escape code must have exactly two digits"
        }
        InvalidInHexEsc => {
            "ASCII hex escape code must contain only hex characters"
        }
        OutOfRangeHexEsc => {
            "ASCII hex escape code must be at most 0x7F"
        }
        NoBraceInUniEsc => {
            "Missing `{` to begin the unicode escape"
        }
        InvalidInUniEsc => {
            "Unicode escape must contain only hex characters and underscores"
        }
        EmptyUniEsc => {
            "Unicode escape must not be empty"
        }
        UnclosedUniEsc => {
            "Missing `}` to terminate the unicode escape"
        }
        UnderscoreUniEsc => {
            "Unicode escape code must not begin with an underscore"
        }
        LongUniEsc => {
            "Unicode escape code must have at most 6 digits"
        }
        LoneSurrogateUniEsc => {
            "Unicode escape code must not be a surrogate"
        }
        OutOfRangeUniEsc => {
            "Unicode escape code must be at most 0x10FFFF"
        }
        UniEscInByte => {
            "Byte literals must not contain unicode escapes"
        }
        NonAsciiInByte  => {
            "Byte literals must not contain non-ASCII characters"
        }
        UnskippedWhitespace => "Whitespace after this escape is not skipped",
        ManySkippedLines => "Multiple lines are skipped by this escape",
    };
    (y, x.is_fatal())
}
fn literal(x: ast::Literal, y: &mut Vec<SyntaxErr>) {
    fn unquote(text: &str, beg: usize, delim: char) -> Option<&str> {
        text.rfind(delim).and_then(|end| text.get(beg..end))
    }
    let token = x.token();
    let text = token.text();
    let mut push_err = |pre_len, off, err: EscErr| {
        let off = token.text_range().start() + TextSize::try_from(off + pre_len).unwrap();
        let (msg, is_err) = unescape_err_to_string(err);
        if is_err {
            y.push(SyntaxErr::new_at_offset(msg, off));
        }
    };
    use ast::LiteralKind::*;
    match x.kind() {
        String(x) => {
            if !x.is_raw() {
                if let Some(x) = unquote(text, 1, '"') {
                    unesc_lit(x, Mode::Str, &mut |range, x| {
                        if let Err(x) = x {
                            push_err(1, range.start, x);
                        }
                    });
                }
            }
        },
        ByteString(x) => {
            if !x.is_raw() {
                if let Some(x) = unquote(text, 2, '"') {
                    unesc_lit(x, Mode::ByteStr, &mut |range, x| {
                        if let Err(x) = x {
                            push_err(1, range.start, x);
                        }
                    });
                }
            }
        },
        CString(x) => {
            if !x.is_raw() {
                if let Some(x) = unquote(text, 2, '"') {
                    unesc_lit(x, Mode::ByteStr, &mut |range, x| {
                        if let Err(x) = x {
                            push_err(1, range.start, x);
                        }
                    });
                }
            }
        },
        Char(_) => {
            if let Some(x) = unquote(text, 1, '\'') {
                unesc_lit(x, Mode::Char, &mut |range, x| {
                    if let Err(x) = x {
                        push_err(1, range.start, x);
                    }
                });
            }
        },
        Byte(_) => {
            if let Some(x) = unquote(text, 2, '\'') {
                unesc_lit(x, Mode::Byte, &mut |range, x| {
                    if let Err(x) = x {
                        push_err(2, range.start, x);
                    }
                });
            }
        },
        IntNumber(_) | FloatNumber(_) | Bool(_) => {},
    }
}
pub fn block_structure(root: &syntax::Node) {
    let mut stack = Vec::new();
    for node in root.descendants_with_tokens() {
        match node.kind() {
            T!['{'] => stack.push(node),
            T!['}'] => {
                if let Some(pair) = stack.pop() {
                    assert_eq!(
                        node.parent(),
                        pair.parent(),
                        "\nunpaired curlies:\n{}\n{:#?}\n",
                        root.text(),
                        root,
                    );
                    assert!(
                        node.next_sibling_or_token().is_none() && pair.prev_sibling_or_token().is_none(),
                        "\nfloating curlies at {:?}\nfile:\n{}\nerror:\n{}\n",
                        node,
                        root.text(),
                        node,
                    );
                }
            },
            _ => (),
        }
    }
}
fn num_name(x: Option<ast::NameRef>, y: &mut Vec<SyntaxErr>) {
    if let Some(x) = int_token(x) {
        if x.text().chars().any(|x| !x.is_ascii_digit()) {
            y.push(SyntaxErr::new(
                "Tuple (struct) field access is only allowed through \
                decimal integers with no underscores or suffix",
                x.text_range(),
            ));
        }
    }
    fn int_token(name_ref: Option<ast::NameRef>) -> Option<syntax::Token> {
        name_ref?
            .syntax()
            .first_child_or_token()?
            .into_token()
            .filter(|x| x.kind() == INT_NUMBER)
    }
}
fn visibility(x: ast::Visibility, y: &mut Vec<SyntaxErr>) {
    let path_without_in_token = x.in_token().is_none()
        && x.path()
            .and_then(|p| p.as_single_name_ref())
            .and_then(|n| n.ident_token())
            .is_some();
    if path_without_in_token {
        y.push(SyntaxErr::new(
            "incorrect visibility restriction",
            x.syntax.text_range(),
        ));
    }
    let parent = match x.syntax().parent() {
        Some(it) => it,
        None => return,
    };
    match parent.kind() {
        FN | CONST | TYPE_ALIAS => (),
        _ => return,
    }
    let impl_def = match parent.parent().and_then(|x| x.parent()).and_then(ast::Impl::cast) {
        Some(it) => it,
        None => return,
    };
    if impl_def.trait_().is_some() && impl_def.attrs().next().is_none() {
        y.push(SyntaxErr::new(
            "Unnecessary visibility qualifier",
            x.syntax.text_range(),
        ));
    }
}
fn range_expr(x: ast::RangeExpr, y: &mut Vec<SyntaxErr>) {
    if x.op_kind() == Some(ast::RangeOp::Inclusive) && x.end().is_none() {
        y.push(SyntaxErr::new(
            "An inclusive range must have an end expression",
            x.syntax().text_range(),
        ));
    }
}
fn validate_path_keywords(segment: ast::PathSegment, errors: &mut Vec<SyntaxErr>) {
    let path = segment.parent_path();
    let is_path_start = segment.coloncolon_token().is_none() && path.qualifier().is_none();
    if let Some(token) = segment.self_token() {
        if !is_path_start {
            errors.push(SyntaxErr::new(
                "The `self` keyword is only allowed as the first segment of a path",
                token.text_range(),
            ));
        }
    } else if let Some(token) = segment.crate_token() {
        if !is_path_start || use_prefix(path).is_some() {
            errors.push(SyntaxErr::new(
                "The `crate` keyword is only allowed as the first segment of a path",
                token.text_range(),
            ));
        }
    }
    fn use_prefix(mut path: ast::Path) -> Option<ast::Path> {
        for node in path.syntax().ancestors().skip(1) {
            match_ast! {
                match node {
                    ast::UseTree(it) => if let Some(tree_path) = it.path() {
                        if tree_path != path {
                            return Some(tree_path);
                        }
                    },
                    ast::UseTreeList(_) => continue,
                    ast::Path(parent) => path = parent,
                    _ => return None,
                }
            };
        }
        None
    }
}
fn validate_trait_object_ref_ty(ty: ast::RefType, errors: &mut Vec<SyntaxErr>) {
    if let Some(ast::Type::DynTraitType(ty)) = ty.ty() {
        if let Some(err) = trait_obj_ty(ty) {
            errors.push(err);
        }
    }
}
fn validate_trait_object_ptr_ty(ty: ast::PtrType, errors: &mut Vec<SyntaxErr>) {
    if let Some(ast::Type::DynTraitType(ty)) = ty.ty() {
        if let Some(err) = trait_obj_ty(ty) {
            errors.push(err);
        }
    }
}
fn validate_trait_object_fn_ptr_ret_ty(ty: ast::FnPtrType, errors: &mut Vec<SyntaxErr>) {
    if let Some(ast::Type::DynTraitType(ty)) = ty.ret_type().and_then(|ty| ty.ty()) {
        if let Some(err) = trait_obj_ty(ty) {
            errors.push(err);
        }
    }
}
fn trait_obj_ty(x: ast::DynTraitType) -> Option<SyntaxErr> {
    let tbl = x.type_bound_list()?;
    if tbl.bounds().count() > 1 {
        let dyn_token = x.dyn_token()?;
        let potential_parenthesis = algo::skip_trivia_token(dyn_token.prev_token()?, Direction::Prev)?;
        let kind = potential_parenthesis.kind();
        if !matches!(kind, T!['('] | T![<] | T![=]) {
            return Some(SyntaxErr::new("ambiguous `+` in a type", x.syntax().text_range()));
        }
    }
    None
}
fn mac_rules(x: ast::MacroRules, y: &mut Vec<SyntaxErr>) {
    if let Some(x) = x.visibility() {
        y.push(SyntaxErr::new(
            "visibilities are not allowed on `macro_rules!` items",
            x.syntax().text_range(),
        ));
    }
}
fn validate_const(x: ast::Const, y: &mut Vec<SyntaxErr>) {
    if let Some(x) = x
        .const_token()
        .and_then(|x| x.next_token())
        .and_then(|x| algo::skip_trivia_token(x, Direction::Next))
        .filter(|x| x.kind() == T![mut])
    {
        y.push(SyntaxErr::new("const globals cannot be mutable", x.text_range()));
    }
}
fn let_expr(x: ast::LetExpr, y: &mut Vec<SyntaxErr>) {
    let mut tok = x.syntax().clone();
    loop {
        use ast::*;
        tok = match tok.parent() {
            Some(x) => x,
            None => break,
        };
        if ParenExpr::can_cast(tok.kind()) {
            continue;
        } else if let Some(x) = BinExpr::cast(tok.clone()) {
            if x.op_kind() == Some(BinaryOp::LogicOp(LogicOp::And)) {
                continue;
            }
        } else if IfExpr::can_cast(tok.kind()) || WhileExpr::can_cast(tok.kind()) || MatchGuard::can_cast(tok.kind()) {
            return;
        }
        break;
    }
    y.push(SyntaxErr::new(
        "`let` expressions are not supported here",
        x.syntax().text_range(),
    ));
}
