use mohu_error::{MohuError, MohuResult, ErrorCode, ErrorKind, ResultExt};

// ── construction ──────────────────────────────────────────────────────────────

#[test]
fn shape_mismatch_constructs() {
    let e = MohuError::ShapeMismatch { expected: vec![3, 4], got: vec![3, 5] };
    assert!(matches!(e, MohuError::ShapeMismatch { .. }));
}

#[test]
fn bug_constructs_internal() {
    let e = MohuError::bug("invariant violated");
    assert!(matches!(e, MohuError::Internal(_)));
}

#[test]
fn alloc_constructs_allocation_failed() {
    let e = MohuError::alloc(1024 * 1024);
    assert!(matches!(e, MohuError::AllocationFailed { bytes: 1048576, .. }));
}

#[test]
fn domain_error_constructs() {
    let e = MohuError::domain("log", "input must be positive");
    assert!(matches!(e, MohuError::DomainError { op: "log", .. }));
}

#[test]
fn matmul_shape_constructs() {
    let e = MohuError::matmul_shape("matmul", [2, 3], [4, 5]);
    assert!(matches!(e, MohuError::MatrixDimensionMismatch {
        lhs_rows: 2, lhs_cols: 3, rhs_rows: 4, rhs_cols: 5, ..
    }));
}

// ── error codes ───────────────────────────────────────────────────────────────

#[test]
fn error_codes_in_correct_domain() {
    assert_eq!(MohuError::ShapeMismatch { expected: vec![], got: vec![] }.code(),
               ErrorCode::ShapeMismatch);
    assert_eq!(MohuError::DLPackNullPointer.code(), ErrorCode::DLPackNullPointer);
    assert_eq!(MohuError::NonContiguous.code(), ErrorCode::NonContiguous);
    assert_eq!(MohuError::DivisionByZero.code(), ErrorCode::DivisionByZero);
}

#[test]
fn error_code_numeric_ranges() {
    let shape_code = MohuError::ShapeMismatch { expected: vec![], got: vec![] }.code() as u32;
    assert!((1000..2000).contains(&shape_code));

    let dtype_code = MohuError::DTypeMismatch {
        expected: "f32".into(), got: "i32".into()
    }.code() as u32;
    assert!((2000..3000).contains(&dtype_code));

    let dlpack_code = MohuError::DLPackNullPointer.code() as u32;
    assert!((7000..8000).contains(&dlpack_code));
}

// ── error kinds ───────────────────────────────────────────────────────────────

#[test]
fn shape_mismatch_is_usage_error() {
    let e = MohuError::ShapeMismatch { expected: vec![2], got: vec![3] };
    assert!(e.is_usage_error());
    assert_eq!(e.kind(), ErrorKind::Usage);
}

#[test]
fn internal_error_is_not_usage() {
    let e = MohuError::bug("test");
    assert!(!e.is_usage_error());
    assert_eq!(e.kind(), ErrorKind::Internal);
}

#[test]
fn allocation_failed_is_transient() {
    let e = MohuError::alloc(4096);
    assert!(e.is_transient());
}

#[test]
fn shape_mismatch_is_not_transient() {
    let e = MohuError::ShapeMismatch { expected: vec![], got: vec![] };
    assert!(!e.is_transient());
}

// ── Display ───────────────────────────────────────────────────────────────────

#[test]
fn display_includes_details() {
    let e = MohuError::ReshapeIncompatible {
        src_len: 6, dst_shape: vec![2, 4], dst_len: 8,
    };
    let s = format!("{e}");
    assert!(s.contains("6"));
    assert!(s.contains("8"));
}

#[test]
fn display_dlpack_null() {
    let s = format!("{}", MohuError::DLPackNullPointer);
    assert!(s.to_lowercase().contains("null") || s.to_lowercase().contains("dlpack"));
}

// ── context extension ─────────────────────────────────────────────────────────

#[test]
fn result_ext_context_wraps_error() {
    let result: MohuResult<i32> = Err(MohuError::DivisionByZero);
    let wrapped = result.context("computing inverse");
    let err = wrapped.unwrap_err();
    assert!(matches!(err, MohuError::Context { .. }));
    let s = format!("{err}");
    assert!(s.contains("computing inverse"));
}

#[test]
fn result_ext_ok_passes_through() {
    let result: MohuResult<i32> = Ok(42);
    assert_eq!(result.context("unused").unwrap(), 42);
}

// ── MohuResult propagation ────────────────────────────────────────────────────

#[test]
fn question_mark_propagates() {
    fn inner() -> MohuResult<()> {
        Err(MohuError::ZeroSliceStep)
    }
    fn outer() -> MohuResult<i32> {
        inner()?;
        Ok(0)
    }
    assert!(outer().is_err());
}

// ── Send + Sync ───────────────────────────────────────────────────────────────

#[test]
fn mohu_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MohuError>();
}
