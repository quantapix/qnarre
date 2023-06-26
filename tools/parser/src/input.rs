use crate::SyntaxKind;

#[allow(non_camel_case_types)]
type bits = u64;

#[derive(Default)]
pub struct Input {
    kind: Vec<SyntaxKind>,
    joint: Vec<bits>,
    contextual_kind: Vec<SyntaxKind>,
}

impl Input {
    #[inline]
    pub fn push(&mut self, kind: SyntaxKind) {
        self.push_impl(kind, SyntaxKind::EOF)
    }
    #[inline]
    pub fn push_ident(&mut self, contextual_kind: SyntaxKind) {
        self.push_impl(SyntaxKind::IDENT, contextual_kind)
    }
    #[inline]
    pub fn was_joint(&mut self) {
        let n = self.len() - 1;
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] |= 1 << b_idx;
    }
    #[inline]
    fn push_impl(&mut self, kind: SyntaxKind, contextual_kind: SyntaxKind) {
        let idx = self.len();
        if idx % (bits::BITS as usize) == 0 {
            self.joint.push(0);
        }
        self.kind.push(kind);
        self.contextual_kind.push(contextual_kind);
    }
}

impl Input {
    pub(crate) fn kind(&self, idx: usize) -> SyntaxKind {
        self.kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
    }
    pub(crate) fn contextual_kind(&self, idx: usize) -> SyntaxKind {
        self.contextual_kind.get(idx).copied().unwrap_or(SyntaxKind::EOF)
    }
    pub(crate) fn is_joint(&self, n: usize) -> bool {
        let (idx, b_idx) = self.bit_index(n);
        self.joint[idx] & 1 << b_idx != 0
    }
}

impl Input {
    fn bit_index(&self, n: usize) -> (usize, usize) {
        let idx = n / (bits::BITS as usize);
        let b_idx = n % (bits::BITS as usize);
        (idx, b_idx)
    }
    fn len(&self) -> usize {
        self.kind.len()
    }
}
