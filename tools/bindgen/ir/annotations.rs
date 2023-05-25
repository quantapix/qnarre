use std::str::FromStr;

use crate::clang;

#[derive(Copy, PartialEq, Eq, Clone, Debug)]
pub enum FieldVisibilityKind {
    Private,
    PublicCrate,
    Public,
}

impl FromStr for FieldVisibilityKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "private" => Ok(Self::Private),
            "crate" => Ok(Self::PublicCrate),
            "public" => Ok(Self::Public),
            _ => Err(format!("Invalid visibility kind: `{}`", s)),
        }
    }
}

impl std::fmt::Display for FieldVisibilityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FieldVisibilityKind::Private => "private",
            FieldVisibilityKind::PublicCrate => "crate",
            FieldVisibilityKind::Public => "public",
        };

        s.fmt(f)
    }
}

impl Default for FieldVisibilityKind {
    fn default() -> Self {
        FieldVisibilityKind::Public
    }
}

#[derive(Copy, PartialEq, Eq, Clone, Debug)]
pub(crate) enum FieldAccessorKind {
    None,
    Regular,
    Unsafe,
    Immutable,
}

#[derive(Default, Clone, PartialEq, Eq, Debug)]
pub(crate) struct Annotations {
    opaque: bool,
    hide: bool,
    use_instead_of: Option<Vec<String>>,
    disallow_copy: bool,
    disallow_debug: bool,
    disallow_default: bool,
    must_use_type: bool,
    visibility_kind: Option<FieldVisibilityKind>,
    accessor_kind: Option<FieldAccessorKind>,
    constify_enum_variant: bool,
    derives: Vec<String>,
}

fn parse_accessor(s: &str) -> FieldAccessorKind {
    match s {
        "false" => FieldAccessorKind::None,
        "unsafe" => FieldAccessorKind::Unsafe,
        "immutable" => FieldAccessorKind::Immutable,
        _ => FieldAccessorKind::Regular,
    }
}

impl Annotations {
    pub(crate) fn new(cursor: &clang::Cursor) -> Option<Annotations> {
        let mut anno = Annotations::default();
        let mut matched_one = false;
        anno.parse(&cursor.comment(), &mut matched_one);

        if matched_one {
            Some(anno)
        } else {
            None
        }
    }

    pub(crate) fn hide(&self) -> bool {
        self.hide
    }

    pub(crate) fn opaque(&self) -> bool {
        self.opaque
    }

    pub(crate) fn use_instead_of(&self) -> Option<&[String]> {
        self.use_instead_of.as_deref()
    }

    pub(crate) fn derives(&self) -> &[String] {
        &self.derives
    }

    pub(crate) fn disallow_copy(&self) -> bool {
        self.disallow_copy
    }

    pub(crate) fn disallow_debug(&self) -> bool {
        self.disallow_debug
    }

    pub(crate) fn disallow_default(&self) -> bool {
        self.disallow_default
    }

    pub(crate) fn must_use_type(&self) -> bool {
        self.must_use_type
    }

    pub(crate) fn visibility_kind(&self) -> Option<FieldVisibilityKind> {
        self.visibility_kind
    }

    pub(crate) fn accessor_kind(&self) -> Option<FieldAccessorKind> {
        self.accessor_kind
    }

    fn parse(&mut self, comment: &clang::Comment, matched: &mut bool) {
        use clang::CXComment_HTMLStartTag;
        if comment.kind() == CXComment_HTMLStartTag
            && comment.get_tag_name() == "div"
            && comment
                .get_tag_attrs()
                .next()
                .map_or(false, |attr| attr.name == "rustbindgen")
        {
            *matched = true;
            for attr in comment.get_tag_attrs() {
                match attr.name.as_str() {
                    "opaque" => self.opaque = true,
                    "hide" => self.hide = true,
                    "nocopy" => self.disallow_copy = true,
                    "nodebug" => self.disallow_debug = true,
                    "nodefault" => self.disallow_default = true,
                    "mustusetype" => self.must_use_type = true,
                    "replaces" => self.use_instead_of = Some(attr.value.split("::").map(Into::into).collect()),
                    "derive" => self.derives.push(attr.value),
                    "private" => {
                        self.visibility_kind = if attr.value != "false" {
                            Some(FieldVisibilityKind::Private)
                        } else {
                            Some(FieldVisibilityKind::Public)
                        };
                    },
                    "accessor" => self.accessor_kind = Some(parse_accessor(&attr.value)),
                    "constant" => self.constify_enum_variant = true,
                    _ => {},
                }
            }
        }

        for child in comment.get_children() {
            self.parse(&child, matched);
        }
    }

    pub(crate) fn constify_enum_variant(&self) -> bool {
        self.constify_enum_variant
    }
}
