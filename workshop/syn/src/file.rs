use super::*;
ast_struct! {

    pub struct File {
        pub shebang: Option<String>,
        pub attrs: Vec<Attribute>,
        pub items: Vec<Item>,
    }
}
pub(crate) mod parsing {
    use super::*;
    use crate::parse::{Parse, ParseStream, Result};

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
mod printing {
    use super::*;
    use crate::attr::FilterAttrs;
    use proc_macro2::TokenStream;
    use quote::{ToTokens, TokenStreamExt};
    #[cfg_attr(doc_cfg, doc(cfg(feature = "printing")))]
    impl ToTokens for File {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            tokens.append_all(self.attrs.inner());
            tokens.append_all(&self.items);
        }
    }
}
