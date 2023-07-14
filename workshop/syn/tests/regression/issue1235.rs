#[test]
fn main() {
    let tokens = quote! {
        pub static FOO: usize;
        pub static BAR: usize;
    };
    let file = syn::parse2::<syn::item::File>(tokens).unwrap();
    println!("{:#?}", file);
    let inner = Group::new(
        pm2::Delim::None,
        quote!(static FOO: usize = 0; pub static BAR: usize = 0),
    );
    let tokens = quote!(pub #inner;);
    let file = syn::parse2::<syn::item::File>(tokens).unwrap();
    println!("{:#?}", file);
    let inner = Group::new(pm2::Delim::None, quote!(static FOO: usize; pub static BAR: usize));
    let tokens = quote!(pub #inner;);
    let file = syn::parse2::<syn::item::File>(tokens).unwrap();
    println!("{:#?}", file);
}
