use proc_macro2::Span;
use quote::spanned::Spanned as ToTokens;
pub trait Spanned: private::Sealed {
    fn span(&self) -> Span;
}
impl<T: ?Sized + ToTokens> Spanned for T {
    fn span(&self) -> Span {
        self.__span()
    }
}
mod private {
    use super::*;
    pub trait Sealed {}
    impl<T: ?Sized + ToTokens> Sealed for T {}
}
