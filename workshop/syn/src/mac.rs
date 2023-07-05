use super::*;
use crate::parse::{Parse, ParseStream, Parser, Result};
use crate::token::{Brace, Bracket, Paren};
use proc_macro2::extra::DelimSpan;
use proc_macro2::Delimiter;
use proc_macro2::TokenStream;
use proc_macro2::TokenTree;
ast_struct! {
    pub struct Macro {
        pub path: Path,
        pub bang_token: Token![!],
        pub delimiter: MacroDelimiter,
        pub tokens: TokenStream,
    }
}
ast_enum! {
    pub enum MacroDelimiter {
        Paren(Paren),
        Brace(Brace),
        Bracket(Bracket),
    }
}
impl MacroDelimiter {
    pub fn span(&self) -> &DelimSpan {
        match self {
            MacroDelimiter::Paren(token) => &token.span,
            MacroDelimiter::Brace(token) => &token.span,
            MacroDelimiter::Bracket(token) => &token.span,
        }
    }
}
impl Macro {
    pub fn parse_body<T: Parse>(&self) -> Result<T> {
        self.parse_body_with(T::parse)
    }

    pub fn parse_body_with<F: Parser>(&self, parser: F) -> Result<F::Output> {
        let scope = self.delimiter.span().close();
        crate::parse::parse_scoped(parser, scope, self.tokens.clone())
    }
}
pub(crate) fn parse_delimiter(input: ParseStream) -> Result<(MacroDelimiter, TokenStream)> {
    input.step(|cursor| {
        if let Some((TokenTree::Group(g), rest)) = cursor.token_tree() {
            let span = g.delim_span();
            let delimiter = match g.delimiter() {
                Delimiter::Parenthesis => MacroDelimiter::Paren(Paren(span)),
                Delimiter::Brace => MacroDelimiter::Brace(Brace(span)),
                Delimiter::Bracket => MacroDelimiter::Bracket(Bracket(span)),
                Delimiter::None => {
                    return Err(cursor.error("expected delimiter"));
                },
            };
            Ok(((delimiter, g.stream()), rest))
        } else {
            Err(cursor.error("expected delimiter"))
        }
    })
}
pub(crate) mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};

    impl Parse for Macro {
        fn parse(input: ParseStream) -> Result<Self> {
            let tokens;
            Ok(Macro {
                path: input.call(Path::parse_mod_style)?,
                bang_token: input.parse()?,
                delimiter: {
                    let (delimiter, content) = parse_delimiter(input)?;
                    tokens = content;
                    delimiter
                },
                tokens,
            })
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    impl MacroDelimiter {
        pub(crate) fn surround(&self, tokens: &mut TokenStream, inner: TokenStream) {
            let (delim, span) = match self {
                MacroDelimiter::Paren(paren) => (Delimiter::Parenthesis, paren.span),
                MacroDelimiter::Brace(brace) => (Delimiter::Brace, brace.span),
                MacroDelimiter::Bracket(bracket) => (Delimiter::Bracket, bracket.span),
            };
            token::printing::delim(delim, span.join(), tokens, inner);
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for Macro {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.path.to_tokens(tokens);
            self.bang_token.to_tokens(tokens);
            self.delimiter.surround(tokens, self.tokens.clone());
        }
    }
}
