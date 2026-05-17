//! mohu-array integration tests.
//!
//! NdArray<T> is not yet implemented (src/array.rs is a stub).
//! All tests are `#[ignore]`d until the implementation lands.
//! Un-ignore by running:
//!   cargo test -p mohu-array --test array_tests -- --include-ignored

// Import what's available now (dtype + buffer are foundations).
use mohu_dtype::DType;
use mohu_buffer::Buffer;

// Placeholder: once NdArray is implemented, import here.
// use mohu_array::NdArray;

// ── construction (blocked on NdArray impl) ────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented (crates/mohu-array/src/array.rs is a stub)"]
fn ndarray_zeros_f64() {
    // let arr: NdArray<f64> = NdArray::zeros(&[3, 4]).unwrap();
    // assert_eq!(arr.shape(), &[3, 4]);
    // assert_eq!(arr.dtype(), DType::F64);
}

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_from_slice() {
    // let arr = NdArray::from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    // assert_eq!(arr.len(), 3);
}

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_ones_i32() {
    // let arr: NdArray<i32> = NdArray::ones(&[5]).unwrap();
    // assert!(arr.as_slice().unwrap().iter().all(|&x| x == 1));
}

// ── reshape ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_reshape() {
    // let arr: NdArray<f32> = NdArray::from_slice(&(0..12).map(|x| x as f32).collect::<Vec<_>>()).unwrap();
    // let r = arr.reshape(&[3, 4]).unwrap();
    // assert_eq!(r.shape(), &[3, 4]);
}

// ── indexing ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_get_element() {
    // let arr: NdArray<f64> = NdArray::from_slice(&[10.0, 20.0, 30.0]).unwrap();
    // assert_eq!(arr[[0]], 10.0_f64);
    // assert_eq!(arr[[2]], 30.0_f64);
}

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_set_element() {
    // let mut arr: NdArray<f64> = NdArray::zeros(&[3]).unwrap();
    // arr[[1]] = 99.0;
    // assert_eq!(arr[[1]], 99.0_f64);
}

// ── transpose ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_transpose_2d() {
    // let arr: NdArray<f32> = NdArray::from_slice_2d(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    // let t = arr.transpose();
    // assert_eq!(t.shape(), &[2, 2]);
    // assert_eq!(t[[0, 1]], 3.0_f32);
}

// ── dtype cast ────────────────────────────────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_cast_f64_to_f32() {
    // let arr: NdArray<f64> = NdArray::from_slice(&[1.0, 2.0, 3.0]).unwrap();
    // let casted: NdArray<f32> = arr.cast().unwrap();
    // assert_eq!(casted.dtype(), DType::F32);
}

// ── iteration ─────────────────────────────────────────────────────────────────

#[test]
#[ignore = "NdArray not yet implemented"]
fn ndarray_iter_1d() {
    // let arr: NdArray<i32> = NdArray::from_slice(&[1, 2, 3, 4]).unwrap();
    // let sum: i32 = arr.iter().sum();
    // assert_eq!(sum, 10);
}

// ── buffer foundation smoke test (always passes) ──────────────────────────────

#[test]
fn buffer_foundation_accessible() {
    // Verify mohu-buffer (which mohu-array wraps) is reachable and functional.
    let buf = Buffer::zeros(DType::F64, &[4]).unwrap();
    assert_eq!(buf.dtype(), DType::F64);
    assert_eq!(buf.len(), 4);
}
