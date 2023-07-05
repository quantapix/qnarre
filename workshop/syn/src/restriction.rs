use super::*;
ast_enum! {
    pub enum Visibility {
        Public(Token![pub]),
        Restricted(VisRestricted),
        Inherited,
    }
}
ast_struct! {
    pub struct VisRestricted {
        pub pub_token: Token![pub],
        pub paren_token: token::Paren,
        pub in_token: Option<Token![in]>,
        pub path: Box<Path>,
    }
}
ast_enum! {
    #[non_exhaustive]
    pub enum FieldMutability {
        None,
    }
}
pub(crate) mod parsing {
    use super::*;
    use crate::ext::IdentExt;
    use crate::parse::discouraged::Speculative;
    use crate::parse::{Parse, ParseStream, Result};

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
        pub(crate) fn is_some(&self) -> bool {
            match self {
                Visibility::Inherited => false,
                _ => true,
            }
        }
    }
}
mod printing {
    use super::*;
    use proc_macro2::TokenStream;
    use quote::ToTokens;
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for Visibility {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Visibility::Public(pub_token) => pub_token.to_tokens(tokens),
                Visibility::Restricted(vis_restricted) => vis_restricted.to_tokens(tokens),
                Visibility::Inherited => {},
            }
        }
    }
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for VisRestricted {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            self.pub_token.to_tokens(tokens);
            self.paren_token.surround(tokens, |tokens| {
                self.in_token.to_tokens(tokens);
                self.path.to_tokens(tokens);
            });
        }
    }
}
