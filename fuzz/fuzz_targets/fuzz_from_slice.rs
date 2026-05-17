#![no_main]

use libfuzzer_sys::fuzz_target;
use mohu_buffer::Buffer;

fuzz_target!(|data: &[f32]| {
    if data.is_empty() { return; }

    let buf = match Buffer::from_slice(data) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Length must match input.
    assert_eq!(buf.len(), data.len());

    let got = buf.as_slice::<f32>().unwrap();
    for (g, e) in got.iter().zip(data.iter()) {
        // NaN-safe comparison: bit pattern must be identical.
        assert_eq!(g.to_bits(), e.to_bits());
    }

    // reshape to [n, 1] must always work.
    let n = data.len();
    let r = buf.reshape(&[n, 1]).unwrap();
    assert_eq!(r.shape(), &[n, 1]);
});
