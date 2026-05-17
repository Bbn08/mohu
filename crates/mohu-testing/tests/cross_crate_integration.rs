//! Cross-crate integration tests.
//!
//! These tests exercise the public APIs of multiple crates together to catch
//! any interface mismatches that per-crate unit tests cannot see.

use mohu_buffer::Buffer;
use mohu_dtype::{DType, promote::CastMode};
use mohu_error::MohuResult;

// ── mohu-buffer + mohu-dtype ──────────────────────────────────────────────────

#[test]
fn dtype_itemsize_matches_buffer_itemsize() {
    for dtype in [DType::F32, DType::F64, DType::I32, DType::I64, DType::U8] {
        let buf = Buffer::zeros(dtype, &[1]).unwrap();
        assert_eq!(buf.itemsize(), dtype.itemsize(), "mismatch for {dtype:?}");
    }
}

#[test]
fn cast_f64_to_f32_preserves_values() -> MohuResult<()> {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let buf = Buffer::from_slice(&data)?;
    let casted = buf.cast(DType::F32, CastMode::Safe)?;

    assert_eq!(casted.dtype(), DType::F32);
    let got = casted.as_slice::<f32>()?;
    for (g, &e) in got.iter().zip(data.iter()) {
        assert!(((*g as f64) - e).abs() < 1e-5, "value mismatch: {g} vs {e}");
    }
    Ok(())
}

#[test]
fn cast_i32_to_f64() -> MohuResult<()> {
    let data: Vec<i32> = vec![10, 20, 30];
    let buf = Buffer::from_slice(&data)?;
    let casted = buf.cast(DType::F64, CastMode::Safe)?;
    let got = casted.as_slice::<f64>()?;
    assert_eq!(got, &[10.0_f64, 20.0, 30.0]);
    Ok(())
}

// ── mohu-buffer + mohu-error ──────────────────────────────────────────────────

#[test]
fn dtype_mismatch_returns_correct_error() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0]).unwrap();
    let result = buf.as_slice::<f32>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    // Error message should mention the type mismatch.
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("dtype") || msg.to_lowercase().contains("type"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn reshape_element_count_mismatch_is_error() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let result = buf.reshape(&[2, 2]); // 3 ≠ 4
    assert!(result.is_err());
}

// ── DLPack cross-crate ────────────────────────────────────────────────────────

#[test]
fn dlpack_dtype_code_matches_mohu_dtype() {
    // Verify that the DLPack dtype code round-trips through mohu-dtype correctly.
    for dtype in [DType::F32, DType::F64, DType::I32, DType::I64] {
        let buf = Buffer::zeros(dtype, &[4]).unwrap();
        let managed = buf.to_dlpack().unwrap();
        let imported = unsafe { Buffer::from_dlpack(managed).unwrap() };
        assert_eq!(imported.dtype(), dtype, "DLPack dtype mismatch for {dtype:?}");
    }
}
