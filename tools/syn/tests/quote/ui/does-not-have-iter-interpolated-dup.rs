use quote::quote;

fn main() {
    let nonrep = "";

    quote!(#(#nonrep #nonrep)*);
}
