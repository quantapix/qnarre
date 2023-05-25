use super::context::{BindgenContext, ItemId};
use super::function::FunctionSig;
use super::item::Item;
use super::traversal::{Trace, Tracer};
use super::ty::TypeKind;
use crate::clang;
use clang::CXChildVisit_Continue;
use clang::CXCursor_ObjCCategoryDecl;
use clang::CXCursor_ObjCClassMethodDecl;
use clang::CXCursor_ObjCClassRef;
use clang::CXCursor_ObjCInstanceMethodDecl;
use clang::CXCursor_ObjCProtocolDecl;
use clang::CXCursor_ObjCProtocolRef;
use clang::CXCursor_ObjCSuperClassRef;
use clang::CXCursor_TemplateTypeParameter;
use proc_macro2::{Ident, Span, TokenStream};

#[derive(Debug)]
pub(crate) struct ObjCInterface {
    name: String,

    category: Option<String>,

    is_protocol: bool,

    pub(crate) template_names: Vec<String>,

    pub(crate) conforms_to: Vec<ItemId>,

    pub(crate) parent_class: Option<ItemId>,

    methods: Vec<ObjCMethod>,

    class_methods: Vec<ObjCMethod>,
}

#[derive(Debug)]
pub(crate) struct ObjCMethod {
    name: String,

    rust_name: String,

    signature: FunctionSig,

    is_class_method: bool,
}

impl ObjCInterface {
    fn new(name: &str) -> ObjCInterface {
        ObjCInterface {
            name: name.to_owned(),
            category: None,
            is_protocol: false,
            template_names: Vec::new(),
            parent_class: None,
            conforms_to: Vec::new(),
            methods: Vec::new(),
            class_methods: Vec::new(),
        }
    }

    pub(crate) fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub(crate) fn rust_name(&self) -> String {
        if let Some(ref cat) = self.category {
            format!("{}_{}", self.name(), cat)
        } else if self.is_protocol {
            format!("P{}", self.name())
        } else {
            format!("I{}", self.name().to_owned())
        }
    }

    pub(crate) fn is_template(&self) -> bool {
        !self.template_names.is_empty()
    }

    pub(crate) fn methods(&self) -> &Vec<ObjCMethod> {
        &self.methods
    }

    pub(crate) fn is_protocol(&self) -> bool {
        self.is_protocol
    }

    pub(crate) fn is_category(&self) -> bool {
        self.category.is_some()
    }

    pub(crate) fn class_methods(&self) -> &Vec<ObjCMethod> {
        &self.class_methods
    }

    pub(crate) fn from_ty(cursor: &clang::Cursor, ctx: &mut BindgenContext) -> Option<Self> {
        let name = cursor.spelling();
        let mut interface = Self::new(&name);

        if cursor.kind() == CXCursor_ObjCProtocolDecl {
            interface.is_protocol = true;
        }

        cursor.visit(|c| {
            match c.kind() {
                CXCursor_ObjCClassRef => {
                    if cursor.kind() == CXCursor_ObjCCategoryDecl {
                        // We are actually a category extension, and we found the reference
                        // to the original interface, so name this interface approriately
                        interface.name = c.spelling();
                        interface.category = Some(cursor.spelling());
                    }
                },
                CXCursor_ObjCProtocolRef => {
                    // Gather protocols this interface conforms to
                    let needle = format!("P{}", c.spelling());
                    let items_map = ctx.items();
                    debug!("Interface {} conforms to {}, find the item", interface.name, needle);

                    for (id, item) in items_map {
                        if let Some(ty) = item.as_type() {
                            if let TypeKind::ObjCInterface(ref protocol) = *ty.kind() {
                                if protocol.is_protocol {
                                    debug!("Checking protocol {}, ty.name {:?}", protocol.name, ty.name());
                                    if Some(needle.as_ref()) == ty.name() {
                                        debug!("Found conforming protocol {:?}", item);
                                        interface.conforms_to.push(id);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                },
                CXCursor_ObjCInstanceMethodDecl | CXCursor_ObjCClassMethodDecl => {
                    let name = c.spelling();
                    let signature = FunctionSig::from_ty(&c.cur_type(), &c, ctx).expect("Invalid function sig");
                    let is_class_method = c.kind() == CXCursor_ObjCClassMethodDecl;
                    let method = ObjCMethod::new(&name, signature, is_class_method);
                    interface.add_method(method);
                },
                CXCursor_TemplateTypeParameter => {
                    let name = c.spelling();
                    interface.template_names.push(name);
                },
                CXCursor_ObjCSuperClassRef => {
                    let item = Item::from_ty_or_ref(c.cur_type(), c, None, ctx);
                    interface.parent_class = Some(item.into());
                },
                _ => {},
            }
            CXChildVisit_Continue
        });
        Some(interface)
    }

    fn add_method(&mut self, method: ObjCMethod) {
        if method.is_class_method {
            self.class_methods.push(method);
        } else {
            self.methods.push(method);
        }
    }
}

impl ObjCMethod {
    fn new(name: &str, signature: FunctionSig, is_class_method: bool) -> ObjCMethod {
        let split_name: Vec<&str> = name.split(':').collect();

        let rust_name = split_name.join("_");

        ObjCMethod {
            name: name.to_owned(),
            rust_name,
            signature,
            is_class_method,
        }
    }

    pub(crate) fn rust_name(&self) -> &str {
        self.rust_name.as_ref()
    }

    pub(crate) fn signature(&self) -> &FunctionSig {
        &self.signature
    }

    pub(crate) fn is_class_method(&self) -> bool {
        self.is_class_method
    }

    pub(crate) fn format_method_call(&self, args: &[TokenStream]) -> TokenStream {
        let split_name: Vec<Option<Ident>> = self
            .name
            .split(':')
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    // Try to parse the current name as an identifier. This might fail if the
                    // name is a keyword so we try to prepend "r#" to it and parse again. If
                    // this also fails, we panic with the first error.
                    Some(
                        syn::parse_str::<Ident>(name)
                            .or_else(|err| syn::parse_str::<Ident>(&format!("r#{}", name)).map_err(|_| err))
                            .expect("Invalid identifier"),
                    )
                }
            })
            .collect();

        if args.is_empty() && split_name.len() == 1 {
            let name = &split_name[0];
            return quote! {
                #name
            };
        }

        assert!(
            args.len() == split_name.len() - 1,
            "Incorrect method name or arguments for objc method, {:?} vs {:?}",
            args,
            split_name
        );

        let mut args_without_types = vec![];
        for arg in args.iter() {
            let arg = arg.to_string();
            let name_and_sig: Vec<&str> = arg.split(' ').collect();
            let name = name_and_sig[0];
            args_without_types.push(Ident::new(name, Span::call_site()))
        }

        let args = split_name.into_iter().zip(args_without_types).map(|(arg, arg_val)| {
            if let Some(arg) = arg {
                quote! { #arg: #arg_val }
            } else {
                quote! { #arg_val: #arg_val }
            }
        });

        quote! {
            #( #args )*
        }
    }
}

impl Trace for ObjCInterface {
    type Extra = ();

    fn trace<T>(&self, context: &BindgenContext, tracer: &mut T, _: &())
    where
        T: Tracer,
    {
        for method in &self.methods {
            method.signature.trace(context, tracer, &());
        }

        for class_method in &self.class_methods {
            class_method.signature.trace(context, tracer, &());
        }

        for protocol in &self.conforms_to {
            tracer.visit(*protocol);
        }
    }
}
