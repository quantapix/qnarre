use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{parse2, File};

use crate::Opts;

mod merge;
mod sort;

use merge::merge_extern_blocks;
use sort::sort_semantically;

struct PostProcessingPass {
    should_run: fn(&Opts) -> bool,
    run: fn(&mut File),
}

macro_rules! pass {
    ($pass:ident) => {
        PostProcessingPass {
            should_run: |options| options.$pass,
            run: |file| $pass(file),
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
