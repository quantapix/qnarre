use crate::{
    algo,
    ast::{self, HasAttrs, HasVisibility, IsString},
    core::{api, Direction, TextSize},
    match_ast, SyntaxErr,
    SyntaxKind::{CONST, FN, INT_NUMBER, TYPE_ALIAS},
    T,
};
use rustc_lexer::unescape::{self, unescape_literal, Mode};
mod block {
    use crate::{
        ast::{self, HasAttrs},
        SyntaxErr,
        SyntaxKind::*,
    };
    pub fn validate_block_expr(block: ast::BlockExpr, errors: &mut Vec<SyntaxErr>) {
        if let Some(parent) = block.syntax().parent() {
            match parent.kind() {
                FN | EXPR_STMT | STMT_LIST => return,
                _ => {},
            }
        }
        if let Some(stmt_list) = block.stmt_list() {
            errors.extend(stmt_list.attrs().filter(|attr| attr.kind().is_inner()).map(|attr| {
                SyntaxErr::new(
                    "A block in this position cannot accept inner attributes",
                    attr.syntax().text_range(),
                )
            }));
        }
    }
}
pub fn validate(root: &api::Node) -> Vec<SyntaxErr> {
    let mut errors = Vec::new();
    for node in root.descendants() {
        match_ast! {
            match node {
                ast::Literal(it) => validate_literal(it, &mut errors),
                ast::Const(it) => validate_const(it, &mut errors),
                ast::BlockExpr(it) => block::validate_block_expr(it, &mut errors),
                ast::FieldExpr(it) => validate_numeric_name(it.name_ref(), &mut errors),
                ast::RecordExprField(it) => validate_numeric_name(it.name_ref(), &mut errors),
                ast::Visibility(it) => validate_visibility(it, &mut errors),
                ast::RangeExpr(it) => validate_range_expr(it, &mut errors),
                ast::PathSegment(it) => validate_path_keywords(it, &mut errors),
                ast::RefType(it) => validate_trait_object_ref_ty(it, &mut errors),
                ast::PtrType(it) => validate_trait_object_ptr_ty(it, &mut errors),
                ast::FnPtrType(it) => validate_trait_object_fn_ptr_ret_ty(it, &mut errors),
                ast::MacroRules(it) => validate_macro_rules(it, &mut errors),
                ast::LetExpr(it) => validate_let_expr(it, &mut errors),
                _ => (),
            }
        }
    }
    errors
}
fn rustc_unescape_error_to_string(err: unescape::EscapeError) -> (&'static str, bool) {
    use unescape::EscapeError as EE;
    #[rustfmt::skip]
    let err_message = match err {
        EE::ZeroChars => {
            "Literal must not be empty"
        }
        EE::MoreThanOneChar => {
            "Literal must be one character long"
        }
        EE::LoneSlash => {
            "Character must be escaped: `\\`"
        }
        EE::InvalidEscape => {
            "Invalid escape"
        }
        EE::BareCarriageReturn | EE::BareCarriageReturnInRawString => {
            "Character must be escaped: `\r`"
        }
        EE::EscapeOnlyChar => {
            "Escape character `\\` must be escaped itself"
        }
        EE::TooShortHexEscape => {
            "ASCII hex escape code must have exactly two digits"
        }
        EE::InvalidCharInHexEscape => {
            "ASCII hex escape code must contain only hex characters"
        }
        EE::OutOfRangeHexEscape => {
            "ASCII hex escape code must be at most 0x7F"
        }
        EE::NoBraceInUnicodeEscape => {
            "Missing `{` to begin the unicode escape"
        }
        EE::InvalidCharInUnicodeEscape => {
            "Unicode escape must contain only hex characters and underscores"
        }
        EE::EmptyUnicodeEscape => {
            "Unicode escape must not be empty"
        }
        EE::UnclosedUnicodeEscape => {
            "Missing `}` to terminate the unicode escape"
        }
        EE::LeadingUnderscoreUnicodeEscape => {
            "Unicode escape code must not begin with an underscore"
        }
        EE::OverlongUnicodeEscape => {
            "Unicode escape code must have at most 6 digits"
        }
        EE::LoneSurrogateUnicodeEscape => {
            "Unicode escape code must not be a surrogate"
        }
        EE::OutOfRangeUnicodeEscape => {
            "Unicode escape code must be at most 0x10FFFF"
        }
        EE::UnicodeEscapeInByte => {
            "Byte literals must not contain unicode escapes"
        }
        EE::NonAsciiCharInByte  => {
            "Byte literals must not contain non-ASCII characters"
        }
        EE::UnskippedWhitespaceWarning => "Whitespace after this escape is not skipped",
        EE::MultipleSkippedLinesWarning => "Multiple lines are skipped by this escape",
    };
    (err_message, err.is_fatal())
}
fn validate_literal(literal: ast::Literal, acc: &mut Vec<SyntaxErr>) {
    fn unquote(text: &str, prefix_len: usize, end_delimiter: char) -> Option<&str> {
        text.rfind(end_delimiter).and_then(|end| text.get(prefix_len..end))
    }
    let token = literal.token();
    let text = token.text();
    let mut push_err = |prefix_len, off, err: unescape::EscapeError| {
        let off = token.text_range().start() + TextSize::try_from(off + prefix_len).unwrap();
        let (message, is_err) = rustc_unescape_error_to_string(err);
        if is_err {
            acc.push(SyntaxErr::new_at_offset(message, off));
        }
    };
    match literal.kind() {
        ast::LiteralKind::String(s) => {
            if !s.is_raw() {
                if let Some(without_quotes) = unquote(text, 1, '"') {
                    unescape_literal(without_quotes, Mode::Str, &mut |range, char| {
                        if let Err(err) = char {
                            push_err(1, range.start, err);
                        }
                    });
                }
            }
        },
        ast::LiteralKind::ByteString(s) => {
            if !s.is_raw() {
                if let Some(without_quotes) = unquote(text, 2, '"') {
                    unescape_literal(without_quotes, Mode::ByteStr, &mut |range, char| {
                        if let Err(err) = char {
                            push_err(1, range.start, err);
                        }
                    });
                }
            }
        },
        ast::LiteralKind::CString(s) => {
            if !s.is_raw() {
                if let Some(without_quotes) = unquote(text, 2, '"') {
                    unescape_literal(without_quotes, Mode::ByteStr, &mut |range, char| {
                        if let Err(err) = char {
                            push_err(1, range.start, err);
                        }
                    });
                }
            }
        },
        ast::LiteralKind::Char(_) => {
            if let Some(without_quotes) = unquote(text, 1, '\'') {
                unescape_literal(without_quotes, Mode::Char, &mut |range, char| {
                    if let Err(err) = char {
                        push_err(1, range.start, err);
                    }
                });
            }
        },
        ast::LiteralKind::Byte(_) => {
            if let Some(without_quotes) = unquote(text, 2, '\'') {
                unescape_literal(without_quotes, Mode::Byte, &mut |range, char| {
                    if let Err(err) = char {
                        push_err(2, range.start, err);
                    }
                });
            }
        },
        ast::LiteralKind::IntNumber(_) | ast::LiteralKind::FloatNumber(_) | ast::LiteralKind::Bool(_) => {},
    }
}
pub fn validate_block_structure(root: &api::Node) {
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
fn validate_numeric_name(name_ref: Option<ast::NameRef>, errors: &mut Vec<SyntaxErr>) {
    if let Some(int_token) = int_token(name_ref) {
        if int_token.text().chars().any(|c| !c.is_ascii_digit()) {
            errors.push(SyntaxErr::new(
                "Tuple (struct) field access is only allowed through \
                decimal integers with no underscores or suffix",
                int_token.text_range(),
            ));
        }
    }
    fn int_token(name_ref: Option<ast::NameRef>) -> Option<api::Token> {
        name_ref?
            .syntax()
            .first_child_or_token()?
            .into_token()
            .filter(|it| it.kind() == INT_NUMBER)
    }
}
fn validate_visibility(vis: ast::Visibility, errors: &mut Vec<SyntaxErr>) {
    let path_without_in_token = vis.in_token().is_none()
        && vis
            .path()
            .and_then(|p| p.as_single_name_ref())
            .and_then(|n| n.ident_token())
            .is_some();
    if path_without_in_token {
        errors.push(SyntaxErr::new(
            "incorrect visibility restriction",
            vis.syntax.text_range(),
        ));
    }
    let parent = match vis.syntax().parent() {
        Some(it) => it,
        None => return,
    };
    match parent.kind() {
        FN | CONST | TYPE_ALIAS => (),
        _ => return,
    }
    let impl_def = match parent.parent().and_then(|it| it.parent()).and_then(ast::Impl::cast) {
        Some(it) => it,
        None => return,
    };
    if impl_def.trait_().is_some() && impl_def.attrs().next().is_none() {
        errors.push(SyntaxErr::new(
            "Unnecessary visibility qualifier",
            vis.syntax.text_range(),
        ));
    }
}
fn validate_range_expr(expr: ast::RangeExpr, errors: &mut Vec<SyntaxErr>) {
    if expr.op_kind() == Some(ast::RangeOp::Inclusive) && expr.end().is_none() {
        errors.push(SyntaxErr::new(
            "An inclusive range must have an end expression",
            expr.syntax().text_range(),
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
        if let Some(err) = validate_trait_object_ty(ty) {
            errors.push(err);
        }
    }
}
fn validate_trait_object_ptr_ty(ty: ast::PtrType, errors: &mut Vec<SyntaxErr>) {
    if let Some(ast::Type::DynTraitType(ty)) = ty.ty() {
        if let Some(err) = validate_trait_object_ty(ty) {
            errors.push(err);
        }
    }
}
fn validate_trait_object_fn_ptr_ret_ty(ty: ast::FnPtrType, errors: &mut Vec<SyntaxErr>) {
    if let Some(ast::Type::DynTraitType(ty)) = ty.ret_type().and_then(|ty| ty.ty()) {
        if let Some(err) = validate_trait_object_ty(ty) {
            errors.push(err);
        }
    }
}
fn validate_trait_object_ty(ty: ast::DynTraitType) -> Option<SyntaxErr> {
    let tbl = ty.type_bound_list()?;
    if tbl.bounds().count() > 1 {
        let dyn_token = ty.dyn_token()?;
        let potential_parenthesis = algo::skip_trivia_token(dyn_token.prev_token()?, Direction::Prev)?;
        let kind = potential_parenthesis.kind();
        if !matches!(kind, T!['('] | T![<] | T![=]) {
            return Some(SyntaxErr::new("ambiguous `+` in a type", ty.syntax().text_range()));
        }
    }
    None
}
fn validate_macro_rules(mac: ast::MacroRules, errors: &mut Vec<SyntaxErr>) {
    if let Some(vis) = mac.visibility() {
        errors.push(SyntaxErr::new(
            "visibilities are not allowed on `macro_rules!` items",
            vis.syntax().text_range(),
        ));
    }
}
fn validate_const(const_: ast::Const, errors: &mut Vec<SyntaxErr>) {
    if let Some(mut_token) = const_
        .const_token()
        .and_then(|t| t.next_token())
        .and_then(|t| algo::skip_trivia_token(t, Direction::Next))
        .filter(|t| t.kind() == T![mut])
    {
        errors.push(SyntaxErr::new(
            "const globals cannot be mutable",
            mut_token.text_range(),
        ));
    }
}
fn validate_let_expr(let_: ast::LetExpr, errors: &mut Vec<SyntaxErr>) {
    let mut token = let_.syntax().clone();
    loop {
        token = match token.parent() {
            Some(it) => it,
            None => break,
        };
        if ast::ParenExpr::can_cast(token.kind()) {
            continue;
        } else if let Some(it) = ast::BinExpr::cast(token.clone()) {
            if it.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And)) {
                continue;
            }
        } else if ast::IfExpr::can_cast(token.kind())
            || ast::WhileExpr::can_cast(token.kind())
            || ast::MatchGuard::can_cast(token.kind())
        {
            return;
        }
        break;
    }
    errors.push(SyntaxErr::new(
        "`let` expressions are not supported here",
        let_.syntax().text_range(),
    ));
}
