#![allow(clippy::uninlined_format_args)]
#[macro_use]
mod macros;
#[test]
fn test_basic() {
    let content = "#!/usr/bin/env rustx\nfn main() {}";
    let file = syn::parse_file(content).unwrap();
    snapshot!(file, @r###"
    File {
        shebang: Some("#!/usr/bin/env rustx"),
        items: [
            Item::Fn {
                vis: data::Visibility::Inherited,
                sig: item::Sig {
                    ident: "main",
                    gens: gen::Gens,
                    ret: typ::Ret::Default,
                },
                block: Block,
            },
        ],
    }
    "###);
}
#[test]
fn test_comment() {
    let content = "#!//am/i/a/comment\n[allow(dead_code)] fn main() {}";
    let file = syn::parse_file(content).unwrap();
    snapshot!(file, @r###"
    File {
        attrs: [
            attr::Attr {
                style: attr::Style::Inner,
                meta: meta::Meta::List {
                    path: Path {
                        segments: [
                            path::Segment {
                                ident: "allow",
                            },
                        ],
                    },
                    delimiter: MacroDelimiter::Paren,
                    tokens: pm2::Stream(`dead_code`),
                },
            },
        ],
        items: [
            Item::Fn {
                vis: data::Visibility::Inherited,
                sig: item::Sig {
                    ident: "main",
                    gens: gen::Gens,
                    ret: typ::Ret::Default,
                },
                block: Block,
            },
        ],
    }
    "###);
}
