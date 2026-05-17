use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mohu_buffer::{Buffer, Order};
use mohu_dtype::DType;

fn bench_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("zeros");
    for &n in &[64usize, 1024, 16384, 262144] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let buf = Buffer::zeros(DType::F64, black_box(&[n])).unwrap();
                black_box(buf);
            });
        });
    }
    group.finish();
}

fn bench_from_slice(c: &mut Criterion) {
    let data: Vec<f64> = (0..65536).map(|x| x as f64).collect();
    let mut group = c.benchmark_group("from_slice");
    for &n in &[64usize, 1024, 16384, 65536] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let buf = Buffer::from_slice(black_box(&data[..n])).unwrap();
                black_box(buf);
            });
        });
    }
    group.finish();
}

fn bench_reshape(c: &mut Criterion) {
    let data: Vec<f64> = (0..1024).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    c.bench_function("reshape_1024_to_32x32", |b| {
        b.iter(|| {
            let r = black_box(&buf).reshape(black_box(&[32, 32])).unwrap();
            black_box(r);
        });
    });
}

fn bench_transpose(c: &mut Criterion) {
    let buf = Buffer::alloc(DType::F64, &[512, 512], Order::C).unwrap();
    c.bench_function("transpose_512x512", |b| {
        b.iter(|| {
            let t = black_box(&buf).transpose();
            black_box(t);
        });
    });
}

fn bench_cast(c: &mut Criterion) {
    use mohu_dtype::promote::CastMode;
    let data: Vec<f64> = (0..16384).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    c.bench_function("cast_f64_to_f32_16k", |b| {
        b.iter(|| {
            let c = black_box(&buf).cast(DType::F32, CastMode::Safe).unwrap();
            black_box(c);
        });
    });
}

fn bench_dlpack_roundtrip(c: &mut Criterion) {
    let data: Vec<f64> = (0..4096).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    c.bench_function("dlpack_export_import_4096", |b| {
        b.iter(|| {
            let managed = black_box(&buf).to_dlpack().unwrap();
            // SAFETY: valid pointer from to_dlpack.
            let imported = unsafe { Buffer::from_dlpack(managed).unwrap() };
            black_box(imported);
        });
    });
}

criterion_group!(
    benches,
    bench_zeros,
    bench_from_slice,
    bench_reshape,
    bench_transpose,
    bench_cast,
    bench_dlpack_roundtrip,
);
criterion_main!(benches);
