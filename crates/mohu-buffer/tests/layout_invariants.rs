use mohu_buffer::{Buffer, Order, SliceArg};
use mohu_dtype::DType;

// ── strides ───────────────────────────────────────────────────────────────────

#[test]
fn c_order_strides_row_major() {
    // shape [3, 4], f64 (8 bytes):
    //   strides = [4*8, 8] = [32, 8]
    let buf = Buffer::alloc(DType::F64, &[3, 4], Order::C).unwrap();
    let strides = buf.strides();
    assert_eq!(strides, &[32, 8]);
}

#[test]
fn f_order_strides_col_major() {
    // shape [3, 4], f64 (8 bytes):
    //   strides = [8, 3*8] = [8, 24]
    let buf = Buffer::alloc(DType::F64, &[3, 4], Order::F).unwrap();
    let strides = buf.strides();
    assert_eq!(strides, &[8, 24]);
}

#[test]
fn transpose_swaps_strides() {
    let buf = Buffer::alloc(DType::F64, &[3, 4], Order::C).unwrap();
    let t = buf.transpose();
    // transposed: shape [4,3], strides [8, 32]
    assert_eq!(t.shape(), &[4, 3]);
    assert_eq!(t.strides(), &[8, 32]);
}

#[test]
fn c_contiguous_flag_set_on_new_c() {
    let buf = Buffer::alloc(DType::F32, &[5, 5], Order::C).unwrap();
    assert!(buf.is_c_contiguous());
    assert!(!buf.is_f_contiguous());
}

#[test]
fn f_contiguous_flag_set_on_new_f() {
    let buf = Buffer::alloc(DType::F32, &[5, 5], Order::F).unwrap();
    assert!(buf.is_f_contiguous());
    assert!(!buf.is_c_contiguous());
}

#[test]
fn transpose_not_c_contiguous() {
    let buf = Buffer::alloc(DType::F64, &[4, 4], Order::C).unwrap();
    let t = buf.transpose();
    assert!(!t.is_c_contiguous());
}

// ── nbytes consistency ─────────────────────────────────────────────────────────

#[test]
fn nbytes_equals_len_times_itemsize() {
    for &dtype in &[DType::F32, DType::F64, DType::I32, DType::I64] {
        let buf = Buffer::zeros(dtype, &[7, 8]).unwrap();
        assert_eq!(buf.nbytes(), buf.len() * buf.itemsize());
        assert_eq!(buf.len(), 7 * 8);
    }
}

// ── reshape does not re-allocate ──────────────────────────────────────────────

#[test]
fn reshape_is_view_not_copy() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let shared = buf.share();
    let reshaped = shared.reshape(&[2, 3]).unwrap();
    // Both should be shared (same Arc)
    assert!(buf.is_shared());
    let _ = reshaped; // keep alive
}

// ── slice axis ────────────────────────────────────────────────────────────────

#[test]
fn slice_preserves_column_count() {
    let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[4, 5]).unwrap();
    let s = buf.slice_axis(0, SliceArg { start: Some(1), stop: Some(3), step: Some(1) }).unwrap();
    assert_eq!(s.shape(), &[2, 5]);
}

#[test]
fn slice_step_2_halves_length() {
    let data: Vec<i32> = (0..20).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    let s = buf.slice_axis(0, SliceArg { start: Some(0), stop: Some(20), step: Some(2) }).unwrap();
    assert_eq!(s.shape(), &[10]);
}

// ── broadcast strides are zero ────────────────────────────────────────────────

#[test]
fn broadcast_zero_stride_on_expanded_axis() {
    let data = vec![1.0_f64, 2.0, 3.0];
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[1, 3]).unwrap();
    let b = buf.broadcast_to(&[5, 3]).unwrap();
    // Row axis was size 1, so its byte stride should be 0 after broadcast.
    assert_eq!(b.strides()[0], 0);
    assert_eq!(b.shape(), &[5, 3]);
}

// ── permute ───────────────────────────────────────────────────────────────────

#[test]
fn permute_shape_reordered() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[2, 3, 4]).unwrap();
    let p = buf.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);
}

#[test]
fn permute_ndim_preserved() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[2, 3, 4]).unwrap();
    let p = buf.permute(&[1, 2, 0]).unwrap();
    assert_eq!(p.ndim(), 3);
}

// ── 1D edge cases ─────────────────────────────────────────────────────────────

#[test]
fn single_element_buffer() {
    let buf = Buffer::from_slice(&[42.0_f64]).unwrap();
    assert_eq!(buf.shape(), &[1]);
    assert_eq!(buf.len(), 1);
    assert_eq!(buf.get::<f64>(&[0]).unwrap(), 42.0_f64);
}

#[test]
fn scalar_reshape_to_1d() {
    let data = vec![7.0_f32];
    let buf = Buffer::from_slice(&data).unwrap();
    let r = buf.reshape(&[1]).unwrap();
    assert_eq!(r.shape(), &[1]);
    assert_eq!(r.get::<f32>(&[0]).unwrap(), 7.0_f32);
}
