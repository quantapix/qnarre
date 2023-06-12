use proc_macro::TokenStream;

#[proc_macro]
pub fn do_thrice(args: TokenStream) -> TokenStream {
    let mut stream = TokenStream::default();
    stream.extend(args.clone());
    stream.extend(args.clone());
    stream.extend(args);
    stream
}

#[allow(dead_code)]
fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
