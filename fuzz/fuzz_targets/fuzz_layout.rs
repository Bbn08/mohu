#![no_main]

use libfuzzer_sys::fuzz_target;
use mohu_buffer::{Buffer, Order};
use mohu_dtype::DType;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 { return; }

    // Derive shape dimensions from input bytes (small, to avoid OOM).
    let ndim = (data[0] % 4) as usize + 1;
    if data.len() < ndim + 1 { return; }

    let shape: Vec<usize> = data[1..=ndim]
        .iter()
        .map(|&b| (b % 16) as usize + 1)
        .collect();

    let order = if data[0] & 1 == 0 { Order::C } else { Order::F };

    let buf = match Buffer::alloc(DType::F64, &shape, order) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Layout invariants must hold.
    assert_eq!(buf.ndim(), ndim);
    assert_eq!(buf.shape(), shape.as_slice());
    assert_eq!(buf.nbytes(), buf.len() * buf.itemsize());

    // transpose must produce valid shape.
    if ndim >= 2 {
        let t = buf.transpose();
        assert_eq!(t.ndim(), ndim);
        let mut expected_shape = shape.clone();
        expected_shape.reverse();
        assert_eq!(t.shape(), expected_shape.as_slice());
    }
});
