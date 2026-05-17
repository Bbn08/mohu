//! Smoke tests for mohu-core re-exports.
//!
//! mohu-core is a thin re-export facade. These tests verify that the facade
//! compiles and that re-exported types are usable through the mohu_core path.

use mohu_core::{mohu_error, mohu_dtype, mohu_buffer};

// ── re-export accessibility ───────────────────────────────────────────────────

#[test]
fn error_types_reachable_via_core() {
    let e = mohu_error::MohuError::DivisionByZero;
    assert!(matches!(e, mohu_error::MohuError::DivisionByZero));
}

#[test]
fn dtype_reachable_via_core() {
    let dt = mohu_dtype::DType::F64;
    assert_eq!(dt.itemsize(), 8);
}

#[test]
fn buffer_reachable_via_core() {
    let buf = mohu_buffer::Buffer::zeros(mohu_dtype::DType::F32, &[4]).unwrap();
    assert_eq!(buf.len(), 4);
}

#[test]
fn mohu_result_type_alias_works() {
    let result: mohu_error::MohuResult<i32> = Ok(42);
    assert_eq!(result.unwrap(), 42);
}

// ── cross-type interaction through core ───────────────────────────────────────

#[test]
fn dtype_from_core_used_with_buffer_from_core() {
    let dtype = mohu_dtype::DType::I64;
    let buf   = mohu_buffer::Buffer::zeros(dtype, &[8]).unwrap();
    assert_eq!(buf.dtype(), dtype);
    assert_eq!(buf.itemsize(), dtype.itemsize());
}

#[test]
fn error_from_core_matches_code_from_core() {
    use mohu_error::{MohuError, ErrorCode};
    let e = MohuError::NonContiguous;
    assert_eq!(e.code(), ErrorCode::NonContiguous);
}

#[test]
fn promote_through_core() {
    use mohu_dtype::{DType, promote::promote};
    assert_eq!(promote(DType::F32, DType::F64), DType::F64);
}
