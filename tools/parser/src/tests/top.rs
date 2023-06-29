use expect_test::expect;

use crate::TopEntry;

#[test]
fn source_file() {
    check(
        TopEntry::SourceFile,
        "",
        expect![[r#"
        SOURCE_FILE
    "#]],
    );

    check(
        TopEntry::SourceFile,
        "struct S;",
        expect![[r#"
        SOURCE_FILE
          STRUCT
            STRUCT_KW "struct"
            WHITESPACE " "
            NAME
              IDENT "S"
            SEMICOLON ";"
    "#]],
    );

    check(
        TopEntry::SourceFile,
        "@error@",
        expect![[r#"
        SOURCE_FILE
          ERROR
            AT "@"
          MACRO_CALL
            PATH
              PATH_SEGMENT
                NAME_REF
                  IDENT "error"
          ERROR
            AT "@"
        error 0: expected an item
        error 6: expected BANG
        error 6: expected `{`, `[`, `(`
        error 6: expected SEMICOLON
        error 6: expected an item
    "#]],
    );
}

#[test]
fn macro_stmt() {
    check(
        TopEntry::MacroStmts,
        "",
        expect![[r#"
            MACRO_STMTS
        "#]],
    );
    check(
        TopEntry::MacroStmts,
        "#!/usr/bin/rust",
        expect![[r##"
            MACRO_STMTS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected expression, item or let statement
        "##]],
    );
    check(
        TopEntry::MacroStmts,
        "let x = 1 2 struct S;",
        expect![[r#"
            MACRO_STMTS
              LET_STMT
                LET_KW "let"
                WHITESPACE " "
                IDENT_PAT
                  NAME
                    IDENT "x"
                WHITESPACE " "
                EQ "="
                WHITESPACE " "
                LITERAL
                  INT_NUMBER "1"
              WHITESPACE " "
              EXPR_STMT
                LITERAL
                  INT_NUMBER "2"
              WHITESPACE " "
              STRUCT
                STRUCT_KW "struct"
                WHITESPACE " "
                NAME
                  IDENT "S"
                SEMICOLON ";"
        "#]],
    );
}

#[test]
fn macro_items() {
    check(
        TopEntry::MacroItems,
        "",
        expect![[r#"
            MACRO_ITEMS
        "#]],
    );
    check(
        TopEntry::MacroItems,
        "#!/usr/bin/rust",
        expect![[r##"
            MACRO_ITEMS
              ERROR
                SHEBANG "#!/usr/bin/rust"
            error 0: expected an item
        "##]],
    );
    check(
        TopEntry::MacroItems,
        "struct S; foo!{}",
        expect![[r#"
            MACRO_ITEMS
              STRUCT
                STRUCT_KW "struct"
                WHITESPACE " "
                NAME
                  IDENT "S"
                SEMICOLON ";"
              WHITESPACE " "
              MACRO_CALL
                PATH
                  PATH_SEGMENT
                    NAME_REF
                      IDENT "foo"
                BANG "!"
                TOKEN_TREE
                  L_CURLY "{"
                  R_CURLY "}"
        "#]],
    );
}

#[test]
fn macro_pattern() {
    check(
        TopEntry::Pattern,
        "",
        expect![[r#"
            ERROR
            error 0: expected pattern
        "#]],
    );
    check(
        TopEntry::Pattern,
        "Some(_)",
        expect![[r#"
            TUPLE_STRUCT_PAT
              PATH
                PATH_SEGMENT
                  NAME_REF
                    IDENT "Some"
              L_PAREN "("
              WILDCARD_PAT
                UNDERSCORE "_"
              R_PAREN ")"
        "#]],
    );

    check(
        TopEntry::Pattern,
        "None leftover tokens",
        expect![[r#"
            ERROR
              IDENT_PAT
                NAME
                  IDENT "None"
              WHITESPACE " "
              IDENT "leftover"
              WHITESPACE " "
              IDENT "tokens"
        "#]],
    );

    check(
        TopEntry::Pattern,
        "@err",
        expect![[r#"
            ERROR
              ERROR
                AT "@"
              IDENT "err"
            error 0: expected pattern
        "#]],
    );
}

#[test]
fn ty() {
    check(
        TopEntry::Type,
        "",
        expect![[r#"
            ERROR
            error 0: expected type
        "#]],
    );

    check(
        TopEntry::Type,
        "Option<!>",
        expect![[r#"
            PATH_TYPE
              PATH
                PATH_SEGMENT
                  NAME_REF
                    IDENT "Option"
                  GENERIC_ARG_LIST
                    L_ANGLE "<"
                    TYPE_ARG
                      NEVER_TYPE
                        BANG "!"
                    R_ANGLE ">"
        "#]],
    );
    check(
        TopEntry::Type,
        "() () ()",
        expect![[r#"
            ERROR
              TUPLE_TYPE
                L_PAREN "("
                R_PAREN ")"
              WHITESPACE " "
              L_PAREN "("
              R_PAREN ")"
              WHITESPACE " "
              L_PAREN "("
              R_PAREN ")"
        "#]],
    );
    check(
        TopEntry::Type,
        "$$$",
        expect![[r#"
            ERROR
              ERROR
                DOLLAR "$"
              DOLLAR "$"
              DOLLAR "$"
            error 0: expected type
        "#]],
    );
}

#[test]
fn expr() {
    check(
        TopEntry::Expr,
        "",
        expect![[r#"
            ERROR
            error 0: expected expression
        "#]],
    );
    check(
        TopEntry::Expr,
        "2 + 2 == 5",
        expect![[r#"
        BIN_EXPR
          BIN_EXPR
            LITERAL
              INT_NUMBER "2"
            WHITESPACE " "
            PLUS "+"
            WHITESPACE " "
            LITERAL
              INT_NUMBER "2"
          WHITESPACE " "
          EQ2 "=="
          WHITESPACE " "
          LITERAL
            INT_NUMBER "5"
    "#]],
    );
    check(
        TopEntry::Expr,
        "let _ = 0;",
        expect![[r#"
            ERROR
              LET_EXPR
                LET_KW "let"
                WHITESPACE " "
                WILDCARD_PAT
                  UNDERSCORE "_"
                WHITESPACE " "
                EQ "="
                WHITESPACE " "
                LITERAL
                  INT_NUMBER "0"
              SEMICOLON ";"
        "#]],
    );
}

#[track_caller]
fn check(entry: TopEntry, input: &str, expect: expect_test::Expect) {
    let (parsed, _errors) = super::parse(entry, input);
    expect.assert_eq(&parsed)
}
