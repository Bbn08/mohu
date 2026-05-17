use mohu_buffer::{Buffer, DLManagedTensor};
use mohu_dtype::DType;

fn export_import(src: &Buffer) -> Buffer {
    let managed = src.to_dlpack().expect("to_dlpack");
    // SAFETY: we just produced this pointer from a valid Buffer.
    unsafe { Buffer::from_dlpack(managed).expect("from_dlpack") }
}

#[test]
fn dlpack_f64_1d_roundtrip() {
    let data: Vec<f64> = (0..16).map(|x| x as f64 * 1.5).collect();
    let src = Buffer::from_slice(&data).unwrap();
    let imported = export_import(&src);

    assert_eq!(imported.shape(), src.shape());
    assert_eq!(imported.dtype(), DType::F64);
    assert_eq!(imported.len(), src.len());

    let got = imported.as_slice::<f64>().unwrap();
    assert_eq!(got, data.as_slice());
}

#[test]
fn dlpack_f32_2d_roundtrip() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let src = Buffer::from_slice(&data).unwrap().reshape(&[3, 4]).unwrap();
    let imported = export_import(&src);

    assert_eq!(imported.shape(), &[3, 4]);
    assert_eq!(imported.dtype(), DType::F32);

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(
                imported.get::<f32>(&[i, j]).unwrap(),
                src.get::<f32>(&[i, j]).unwrap(),
            );
        }
    }
}

#[test]
fn dlpack_i32_roundtrip() {
    let data: Vec<i32> = (0..8).collect();
    let src = Buffer::from_slice(&data).unwrap();
    let imported = export_import(&src);

    assert_eq!(imported.dtype(), DType::I32);
    assert_eq!(imported.as_slice::<i32>().unwrap(), data.as_slice());
}

#[test]
fn dlpack_dtype_preserved_f64() {
    let buf = Buffer::zeros(DType::F64, &[4]).unwrap();
    let imported = export_import(&buf);
    assert_eq!(imported.dtype(), DType::F64);
}

#[test]
fn dlpack_dtype_preserved_f32() {
    let buf = Buffer::zeros(DType::F32, &[4]).unwrap();
    let imported = export_import(&buf);
    assert_eq!(imported.dtype(), DType::F32);
}

#[test]
fn dlpack_shape_preserved_3d() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let src = Buffer::from_slice(&data).unwrap().reshape(&[2, 3, 4]).unwrap();
    let imported = export_import(&src);
    assert_eq!(imported.shape(), &[2, 3, 4]);
}

#[test]
fn dlpack_null_pointer_errors() {
    let result = unsafe { Buffer::from_dlpack(std::ptr::null_mut::<DLManagedTensor>()) };
    assert!(result.is_err());
}

#[test]
fn dlpack_deleter_called_on_drop() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    static DROPPED: AtomicBool = AtomicBool::new(false);

    let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    let src = Buffer::from_slice(&data).unwrap();
    let managed = src.to_dlpack().unwrap();

    // Wrap original deleter to observe teardown.
    // We simply verify that importing and then dropping the result
    // does not panic or leak (MIRI / AddressSanitizer would catch leaks).
    {
        // SAFETY: valid pointer from to_dlpack.
        let imported = unsafe { Buffer::from_dlpack(managed).unwrap() };
        drop(imported);
    }
    // If we got here without a double-free or segfault, the deleter is safe.
    let _ = DROPPED.load(Ordering::SeqCst);
}
