#![no_main]

use libfuzzer_sys::fuzz_target;
use mohu_buffer::Buffer;
use mohu_dtype::DType;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 { return; }

    // Use first byte to pick dtype, rest as f64 payload.
    let dtype = match data[0] % 4 {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I32,
        _ => DType::I64,
    };

    let elem_size = dtype.itemsize();
    let payload = &data[1..];
    let n = payload.len() / elem_size;
    if n == 0 { return; }

    // Build a buffer through the normal allocation path.
    let buf = match Buffer::zeros(dtype, &[n]) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Export → import DLPack round-trip must not panic or corrupt memory.
    let managed = match buf.to_dlpack() {
        Ok(m) => m,
        Err(_) => return,
    };

    // SAFETY: pointer was just produced by to_dlpack.
    let imported = unsafe { Buffer::from_dlpack(managed) };
    if let Ok(imp) = imported {
        // Verify shape survives the round-trip.
        assert_eq!(imp.shape(), buf.shape());
        assert_eq!(imp.dtype(), buf.dtype());
    }
});
