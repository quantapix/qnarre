use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{parse2, File};

use crate::Opts;

mod merge {
    use syn::{
        visit_mut::{visit_file_mut, visit_item_mod_mut, VisitMut},
        File, Item, ItemForeignMod, ItemMod,
    };

    pub fn extern_blocks(x: &mut File) {
        Visitor.visit_file_mut(x)
    }

    struct Visitor;

    impl VisitMut for Visitor {
        fn visit_file_mut(&mut self, x: &mut File) {
            visit_items(&mut x.items);
            visit_file_mut(self, x)
        }

        fn visit_item_mod_mut(&mut self, x: &mut ItemMod) {
            if let Some((_, ref mut xs)) = x.content {
                visit_items(xs);
            }
            visit_item_mod_mut(self, x)
        }
    }

    fn visit_items(xs: &mut Vec<Item>) {
        let mut ys = Vec::<ItemForeignMod>::new();
        for x in std::mem::take(xs) {
            if let Item::ForeignMod(ItemForeignMod {
                attrs,
                abi,
                brace_token,
                unsafety,
                items,
            }) = x
            {
                let mut exists = false;
                for y in &mut ys {
                    if y.attrs == attrs && y.abi == abi {
                        y.items.extend_from_slice(&items);
                        exists = true;
                        break;
                    }
                }
                if !exists {
                    ys.push(ItemForeignMod {
                        attrs,
                        abi,
                        brace_token,
                        unsafety,
                        items,
                    });
                }
            } else {
                xs.push(x);
            }
        }
        for y in ys {
            xs.push(Item::ForeignMod(y));
        }
    }
}
mod sort {
    use syn::{
        visit_mut::{visit_file_mut, visit_item_mod_mut, VisitMut},
        File, Item, ItemMod,
    };

    pub fn semantically(x: &mut File) {
        Visitor.visit_file_mut(x)
    }

    struct Visitor;

    impl VisitMut for Visitor {
        fn visit_file_mut(&mut self, x: &mut File) {
            visit_items(&mut x.items);
            visit_file_mut(self, x)
        }

        fn visit_item_mod_mut(&mut self, x: &mut ItemMod) {
            if let Some((_, ref mut xs)) = x.content {
                visit_items(xs);
            }
            visit_item_mod_mut(self, x)
        }
    }

    fn visit_items(xs: &mut [Item]) {
        xs.sort_by_key(|x| match x {
            Item::Type(_) => 0,
            Item::Struct(_) => 1,
            Item::Const(_) => 2,
            Item::Fn(_) => 3,
            Item::Enum(_) => 4,
            Item::Union(_) => 5,
            Item::Static(_) => 6,
            Item::Trait(_) => 7,
            Item::TraitAlias(_) => 8,
            Item::Impl(_) => 9,
            Item::Mod(_) => 10,
            Item::Use(_) => 11,
            Item::Verbatim(_) => 12,
            Item::ExternCrate(_) => 13,
            Item::ForeignMod(_) => 14,
            Item::Macro(_) => 15,
            _ => 18,
        });
    }
}

use merge::extern_blocks as merge_extern_blocks;
use sort::semantically as sort_semantically;

struct PostProcessingPass {
    should_run: fn(&Opts) -> bool,
    run: fn(&mut File),
}

macro_rules! pass {
    ($pass:ident) => {
        PostProcessingPass {
            should_run: |x| x.$pass,
            run: |x| $pass(x),
        }
    };
}

const PASSES: &[PostProcessingPass] = &[pass!(merge_extern_blocks), pass!(sort_semantically)];

pub(crate) fn postproc(xs: Vec<TokenStream>, opts: &Opts) -> TokenStream {
    let xs = xs.into_iter().collect();
    let syn = PASSES.iter().any(|x| (x.should_run)(opts));
    if !syn {
        return xs;
    }
    let mut y = parse2::<File>(xs).unwrap();
    for x in PASSES {
        if (x.should_run)(opts) {
            (x.run)(&mut y);
        }
    }
    y.into_token_stream()
}
