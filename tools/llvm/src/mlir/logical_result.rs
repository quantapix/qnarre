use mlir_sys::MlirLogicalResult;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogicalResult {
    raw: MlirLogicalResult,
}

impl LogicalResult {
    pub fn success() -> Self {
        Self {
            raw: MlirLogicalResult { value: 1 },
        }
    }

    pub fn failure() -> Self {
        Self {
            raw: MlirLogicalResult { value: 0 },
        }
    }

    pub fn is_success(&self) -> bool {
        self.raw.value != 0
    }

    #[allow(dead_code)]
    pub fn is_failure(&self) -> bool {
        self.raw.value == 0
    }

    pub fn from_raw(result: MlirLogicalResult) -> Self {
        Self { raw: result }
    }

    pub fn to_raw(self) -> MlirLogicalResult {
        self.raw
    }
}

impl From<bool> for LogicalResult {
    fn from(ok: bool) -> Self {
        if ok {
            Self::success()
        } else {
            Self::failure()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success() {
        assert!(LogicalResult::success().is_success());
    }

    #[test]
    fn failure() {
        assert!(LogicalResult::failure().is_failure());
    }
}
