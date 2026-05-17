//! Arithmetic correctness tests.
//!
//! These tests are `#[ignore]`d until mohu-ops implements element-wise arithmetic
//! on Buffer. Un-ignore them by running:
//!   cargo test -p mohu-buffer --test arith_correctness -- --include-ignored

use mohu_buffer::Buffer;
use mohu_dtype::DType;

// Placeholder: once mohu-ops exposes buffer-level add/sub/mul/div, import here.
// use mohu_ops::arith::{add, sub, mul, div};

#[test]
#[ignore = "waiting for mohu-ops element-wise arithmetic (issue #9)"]
fn add_two_f64_buffers() {
    let a = Buffer::from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let b = Buffer::from_slice(&[4.0_f64, 5.0, 6.0]).unwrap();
    // let c = add(&a, &b).unwrap();
    // assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
    let _ = (a, b);
}

#[test]
#[ignore = "waiting for mohu-ops element-wise arithmetic (issue #9)"]
fn sub_two_f32_buffers() {
    let a = Buffer::from_slice(&[10.0_f32, 20.0, 30.0]).unwrap();
    let b = Buffer::from_slice(&[1.0_f32, 2.0, 3.0]).unwrap();
    // let c = sub(&a, &b).unwrap();
    // assert_eq!(c.as_slice::<f32>().unwrap(), &[9.0, 18.0, 27.0]);
    let _ = (a, b);
}

#[test]
#[ignore = "waiting for mohu-ops element-wise arithmetic (issue #9)"]
fn mul_broadcast_scalar() {
    // Broadcast [3] * [1] => [3]
    let a = Buffer::from_slice(&[2.0_f64, 4.0, 6.0]).unwrap();
    let scalar = Buffer::from_slice(&[0.5_f64]).unwrap();
    // let c = mul(&a, &scalar).unwrap();
    // assert_eq!(c.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
    let _ = (a, scalar);
}

#[test]
#[ignore = "waiting for mohu-ops element-wise arithmetic (issue #9)"]
fn div_by_nonzero() {
    let a = Buffer::from_slice(&[6.0_f64, 9.0, 12.0]).unwrap();
    let b = Buffer::from_slice(&[2.0_f64, 3.0, 4.0]).unwrap();
    // let c = div(&a, &b).unwrap();
    // assert_eq!(c.as_slice::<f64>().unwrap(), &[3.0, 3.0, 3.0]);
    let _ = (a, b);
}

#[test]
#[ignore = "waiting for mohu-ops reductions (issue #9)"]
fn sum_reduces_to_scalar() {
    let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    // let s = sum(&buf, None).unwrap();
    // assert_eq!(s.get::<f64>(&[]).unwrap(), 55.0);
    let _ = buf;
}

#[test]
#[ignore = "waiting for mohu-ops reductions (issue #9)"]
fn mean_1d() {
    let data: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
    let buf = Buffer::from_slice(&data).unwrap();
    // let m = mean(&buf, None).unwrap();
    // assert!((m.get::<f64>(&[]).unwrap() - 5.0).abs() < 1e-10);
    let _ = buf;
}

#[test]
#[ignore = "waiting for mohu-ops reductions (issue #9)"]
fn max_1d() {
    let data = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let buf = Buffer::from_slice(&data).unwrap();
    // let m = max(&buf, None).unwrap();
    // assert_eq!(m.get::<f32>(&[]).unwrap(), 9.0_f32);
    let _ = buf;
}

#[test]
#[ignore = "waiting for mohu-ops matmul (issue #13)"]
fn matmul_2x2() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Buffer::from_slice_2d::<f64>(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let b = Buffer::from_slice_2d::<f64>(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();
    // let c = matmul(&a, &b).unwrap();
    // assert_eq!(c.get::<f64>(&[0,0]).unwrap(), 19.0);
    // assert_eq!(c.get::<f64>(&[1,1]).unwrap(), 50.0);
    let _ = (a, b);
}
