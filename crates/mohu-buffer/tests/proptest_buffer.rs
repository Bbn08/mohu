use mohu_buffer::{Buffer, Order};
use mohu_dtype::DType;
use proptest::prelude::*;

fn arb_shape_1d() -> impl Strategy<Value = Vec<usize>> {
    (1usize..=256).prop_map(|n| vec![n])
}

fn arb_shape_2d() -> impl Strategy<Value = Vec<usize>> {
    (1usize..=32, 1usize..=32).prop_map(|(r, c)| vec![r, c])
}

proptest! {
    #[test]
    fn prop_from_slice_roundtrip(data in prop::collection::vec(any::<f64>(), 1..=128)) {
        let buf = Buffer::from_slice(&data).unwrap();
        let got = buf.as_slice::<f64>().unwrap();
        prop_assert_eq!(got, data.as_slice());
    }

    #[test]
    fn prop_zeros_all_zero(shape in arb_shape_1d()) {
        let buf = Buffer::zeros(DType::F64, &shape).unwrap();
        let slice = buf.as_slice::<f64>().unwrap();
        prop_assert!(slice.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn prop_zeros_shape_preserved(shape in arb_shape_2d()) {
        let buf = Buffer::zeros(DType::I32, &shape).unwrap();
        prop_assert_eq!(buf.shape(), shape.as_slice());
    }

    #[test]
    fn prop_len_matches_shape_product(shape in arb_shape_2d()) {
        let buf = Buffer::zeros(DType::F32, &shape).unwrap();
        let expected: usize = shape.iter().product();
        prop_assert_eq!(buf.len(), expected);
    }

    #[test]
    fn prop_double_transpose_identity(data in prop::collection::vec(any::<f64>(), 4..=64)) {
        let n = (data.len() as f64).sqrt() as usize;
        // Use a square sub-slice so n*n == actual len
        let sq = n * n;
        if sq == 0 || sq > data.len() { return Ok(()); }
        let buf = Buffer::from_slice(&data[..sq]).unwrap()
            .reshape(&[n, n]).unwrap();
        let tt = buf.transpose().transpose();
        // Shape must be same as original
        prop_assert_eq!(tt.shape(), buf.shape());
        // Every element must match
        for i in 0..n {
            for j in 0..n {
                prop_assert_eq!(
                    tt.get::<f64>(&[i, j]).unwrap(),
                    buf.get::<f64>(&[i, j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn prop_reshape_preserves_data(data in prop::collection::vec(any::<i32>(), 1..=60)) {
        let len = data.len();
        let buf = Buffer::from_slice(&data).unwrap();
        // Find a valid 2d reshape
        for r in 1..=len {
            if len % r == 0 {
                let c = len / r;
                let reshaped = buf.reshape(&[r, c]).unwrap();
                prop_assert_eq!(reshaped.shape(), &[r, c]);
                // Flatten back and compare
                let flat = reshaped.reshape(&[len]).unwrap();
                prop_assert_eq!(flat.as_slice::<i32>().unwrap(), data.as_slice());
                break;
            }
        }
    }

    #[test]
    fn prop_nbytes_eq_len_times_itemsize(shape in arb_shape_2d()) {
        let buf = Buffer::zeros(DType::F64, &shape).unwrap();
        prop_assert_eq!(buf.nbytes(), buf.len() * buf.itemsize());
    }

    #[test]
    fn prop_c_order_is_c_contiguous(shape in arb_shape_2d()) {
        let buf = Buffer::alloc(DType::F32, &shape, Order::C).unwrap();
        prop_assert!(buf.is_c_contiguous());
    }

    #[test]
    fn prop_f_order_is_f_contiguous(shape in arb_shape_2d()) {
        let buf = Buffer::alloc(DType::F32, &shape, Order::F).unwrap();
        prop_assert!(buf.is_f_contiguous());
    }

    #[test]
    fn prop_aligned_allocation(shape in arb_shape_1d()) {
        let buf = Buffer::zeros(DType::F64, &shape).unwrap();
        prop_assert!(buf.is_aligned());
    }
}
