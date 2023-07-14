#![allow(clippy::non_ascii_literal)]
use syn::*;
#[test]
#[should_panic(expected = "Fork was not derived from the advancing parse stream")]
fn smuggled_speculative_cursor_between_sources() {
    struct BreakRules;
    impl Parse for BreakRules {
        fn parse(input1: Stream) -> Res<Self> {
            let nested = |input2: Stream| {
                input1.advance_to(input2);
                Ok(Self)
            };
            nested.parse::parse_str("")
        }
    }
    syn::parse::parse_str::<BreakRules>("").unwrap();
}
#[test]
#[should_panic(expected = "Fork was not derived from the advancing parse stream")]
fn smuggled_speculative_cursor_between_brackets() {
    struct BreakRules;
    impl Parse for BreakRules {
        fn parse(x: Stream) -> Res<Self> {
            let a;
            let b;
            parenthesized!(a in x);
            parenthesized!(b in x);
            a.advance_to(&b);
            Ok(Self)
        }
    }
    syn::parse::parse_str::<BreakRules>("()()").unwrap();
}
#[test]
#[should_panic(expected = "Fork was not derived from the advancing parse stream")]
fn smuggled_speculative_cursor_into_brackets() {
    struct BreakRules;
    impl Parse for BreakRules {
        fn parse(x: Stream) -> Res<Self> {
            let a;
            parenthesized!(a in x);
            x.advance_to(&a);
            Ok(Self)
        }
    }
    syn::parse::parse_str::<BreakRules>("()").unwrap();
}
#[test]
fn trailing_empty_none_group() {
    fn parse(x: Stream) -> Res<()> {
        x.parse::<Token![+]>()?;
        let content;
        parenthesized!(content in x);
        content.parse::<Token![+]>()?;
        Ok(())
    }
    let tokens = pm2::Stream::from_iter(vec![
        pm2::Tree::Punct(Punct::new('+', pm2::Spacing::Alone)),
        pm2::Tree::Group(Group::new(
            pm2::Delim::Parenthesis,
            pm2::Stream::from_iter(vec![
                pm2::Tree::Punct(Punct::new('+', pm2::Spacing::Alone)),
                pm2::Tree::Group(Group::new(pm2::Delim::None, pm2::Stream::new())),
            ]),
        )),
        pm2::Tree::Group(Group::new(pm2::Delim::None, pm2::Stream::new())),
        pm2::Tree::Group(Group::new(
            pm2::Delim::None,
            pm2::Stream::from_iter(vec![pm2::Tree::Group(Group::new(pm2::Delim::None, pm2::Stream::new()))]),
        )),
    ]);
    parse.parse2(tokens).unwrap();
}
