use mohu_dtype::{
    DType,
    dtype::ALL_DTYPES,
    promote::{promote, can_cast, CastMode},
};

// ── itemsize ──────────────────────────────────────────────────────────────────

#[test]
fn itemsize_all_variants() {
    assert_eq!(DType::Bool.itemsize(), 1);
    assert_eq!(DType::I8.itemsize(),   1);
    assert_eq!(DType::I16.itemsize(),  2);
    assert_eq!(DType::I32.itemsize(),  4);
    assert_eq!(DType::I64.itemsize(),  8);
    assert_eq!(DType::U8.itemsize(),   1);
    assert_eq!(DType::U16.itemsize(),  2);
    assert_eq!(DType::U32.itemsize(),  4);
    assert_eq!(DType::U64.itemsize(),  8);
    assert_eq!(DType::F16.itemsize(),  2);
    assert_eq!(DType::BF16.itemsize(), 2);
    assert_eq!(DType::F32.itemsize(),  4);
    assert_eq!(DType::F64.itemsize(),  8);
    assert_eq!(DType::C64.itemsize(),  8);
    assert_eq!(DType::C128.itemsize(), 16);
}

#[test]
fn bit_width_equals_itemsize_times_8() {
    for dtype in ALL_DTYPES {
        assert_eq!(dtype.bit_width(), (dtype.itemsize() * 8) as u32, "{dtype:?}");
    }
}

// ── classification ────────────────────────────────────────────────────────────

#[test]
fn is_float_correct() {
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F32.is_float());
    assert!(DType::F64.is_float());
    assert!(!DType::I32.is_float());
    assert!(!DType::C64.is_float());
}

#[test]
fn is_integer_correct() {
    for dtype in [DType::I8, DType::I16, DType::I32, DType::I64,
                  DType::U8, DType::U16, DType::U32, DType::U64] {
        assert!(dtype.is_integer(), "{dtype:?} should be integer");
    }
    assert!(!DType::F32.is_integer());
    assert!(!DType::Bool.is_integer());
}

#[test]
fn is_complex_correct() {
    assert!(DType::C64.is_complex());
    assert!(DType::C128.is_complex());
    assert!(!DType::F64.is_complex());
}

#[test]
fn is_numeric_excludes_bool() {
    assert!(!DType::Bool.is_numeric());
    for dtype in [DType::I32, DType::F32, DType::C64] {
        assert!(dtype.is_numeric(), "{dtype:?} should be numeric");
    }
}

#[test]
fn is_ordered_excludes_complex_and_bool() {
    assert!(!DType::C64.is_ordered());
    assert!(!DType::Bool.is_ordered());
    assert!(DType::I32.is_ordered());
    assert!(DType::F64.is_ordered());
}

// ── conversions ───────────────────────────────────────────────────────────────

#[test]
fn as_signed_unsigned_roundtrip() {
    for (u, s) in [(DType::U8,  DType::I8),
                   (DType::U16, DType::I16),
                   (DType::U32, DType::I32),
                   (DType::U64, DType::I64)] {
        assert_eq!(u.as_signed(), s);
        assert_eq!(s.as_unsigned(), u);
    }
}

#[test]
fn real_dtype_of_complex() {
    assert_eq!(DType::C64.real_dtype(),  DType::F32);
    assert_eq!(DType::C128.real_dtype(), DType::F64);
    assert_eq!(DType::F32.real_dtype(),  DType::F32);
}

#[test]
fn complex_dtype_of_float() {
    assert_eq!(DType::F32.complex_dtype(), DType::C64);
    assert_eq!(DType::F64.complex_dtype(), DType::C128);
    assert_eq!(DType::I32.complex_dtype(), DType::I32);
}

#[test]
fn widen_max_types_return_self() {
    assert_eq!(DType::F64.widen(),  DType::F64);
    assert_eq!(DType::I64.widen(),  DType::I64);
    assert_eq!(DType::U64.widen(),  DType::U64);
    assert_eq!(DType::C128.widen(), DType::C128);
}

#[test]
fn narrow_min_types_return_none() {
    assert!(DType::I8.narrow().is_none());
    assert!(DType::U8.narrow().is_none());
    assert!(DType::F16.narrow().is_none());
    assert!(DType::Bool.narrow().is_none());
}

// ── string parsing ────────────────────────────────────────────────────────────

#[test]
fn from_str_numpy_names() {
    assert_eq!(DType::from_str("float32").unwrap(),  DType::F32);
    assert_eq!(DType::from_str("float64").unwrap(),  DType::F64);
    assert_eq!(DType::from_str("int32").unwrap(),    DType::I32);
    assert_eq!(DType::from_str("int64").unwrap(),    DType::I64);
    assert_eq!(DType::from_str("uint8").unwrap(),    DType::U8);
    assert_eq!(DType::from_str("bool").unwrap(),     DType::Bool);
    assert_eq!(DType::from_str("complex64").unwrap(),DType::C64);
    assert_eq!(DType::from_str("bfloat16").unwrap(), DType::BF16);
}

#[test]
fn from_str_aliases() {
    assert_eq!(DType::from_str("f4").unwrap(),  DType::F32);
    assert_eq!(DType::from_str("f8").unwrap(),  DType::F64);
    assert_eq!(DType::from_str("i4").unwrap(),  DType::I32);
    assert_eq!(DType::from_str("i8").unwrap(),  DType::I64);
    assert_eq!(DType::from_str("u1").unwrap(),  DType::U8);
    assert_eq!(DType::from_str("double").unwrap(), DType::F64);
    assert_eq!(DType::from_str("half").unwrap(), DType::F16);
}

#[test]
fn from_str_case_insensitive() {
    assert_eq!(DType::from_str("FLOAT32").unwrap(), DType::F32);
    assert_eq!(DType::from_str("Int32").unwrap(),   DType::I32);
}

#[test]
fn from_str_unknown_errors() {
    assert!(DType::from_str("not_a_dtype").is_err());
    assert!(DType::from_str("").is_err());
    assert!(DType::from_str("f3").is_err());
}

#[test]
fn display_roundtrips_via_from_str() {
    for dtype in ALL_DTYPES {
        if dtype == DType::BF16 { continue; } // no standard numpy char, skip roundtrip
        let s = format!("{dtype}");
        let parsed = DType::from_str(&s).unwrap();
        assert_eq!(parsed, dtype, "display/parse roundtrip failed for {dtype:?}");
    }
}

// ── u8 code roundtrip ─────────────────────────────────────────────────────────

#[test]
fn from_u8_roundtrip() {
    for dtype in ALL_DTYPES {
        let code = dtype.as_u8();
        let recovered = DType::from_u8(code).unwrap();
        assert_eq!(recovered, dtype);
    }
}

#[test]
fn from_u8_invalid_errors() {
    assert!(DType::from_u8(15).is_err());
    assert!(DType::from_u8(255).is_err());
}

// ── type promotion ────────────────────────────────────────────────────────────

#[test]
fn promote_symmetric() {
    for &a in &ALL_DTYPES {
        for &b in &ALL_DTYPES {
            assert_eq!(promote(a, b), promote(b, a), "promote({a:?},{b:?}) not symmetric");
        }
    }
}

#[test]
fn promote_idempotent() {
    for dtype in ALL_DTYPES {
        assert_eq!(promote(dtype, dtype), dtype, "promote({dtype:?},{dtype:?}) not idempotent");
    }
}

#[test]
fn promote_key_cases() {
    assert_eq!(promote(DType::I32, DType::F32),  DType::F64);
    assert_eq!(promote(DType::F16, DType::F32),  DType::F32);
    assert_eq!(promote(DType::F32, DType::F64),  DType::F64);
    assert_eq!(promote(DType::C64, DType::F64),  DType::C128);
    assert_eq!(promote(DType::Bool, DType::I32), DType::I32);
}

// ── casting rules ─────────────────────────────────────────────────────────────

#[test]
fn safe_cast_lossless_only() {
    assert!(can_cast(DType::I8,  DType::I16, CastMode::Safe));
    assert!(can_cast(DType::F32, DType::F64, CastMode::Safe));
    assert!(can_cast(DType::U8,  DType::I16, CastMode::Safe));
    assert!(!can_cast(DType::F64, DType::F32, CastMode::Safe));
    assert!(!can_cast(DType::I64, DType::F32, CastMode::Safe));
}

#[test]
fn unsafe_cast_allows_all() {
    assert!(can_cast(DType::F64, DType::I32, CastMode::Unsafe));
    assert!(can_cast(DType::C64, DType::F32, CastMode::Unsafe));
    assert!(can_cast(DType::I64, DType::U8,  CastMode::Unsafe));
}

#[test]
fn same_dtype_always_safe_cast() {
    for dtype in ALL_DTYPES {
        assert!(can_cast(dtype, dtype, CastMode::Safe), "{dtype:?}→{dtype:?} should be safe");
    }
}

// ── numpy string ──────────────────────────────────────────────────────────────

#[test]
fn numpy_str_all_variants() {
    assert_eq!(DType::F32.numpy_str(),  "float32");
    assert_eq!(DType::F64.numpy_str(),  "float64");
    assert_eq!(DType::I32.numpy_str(),  "int32");
    assert_eq!(DType::Bool.numpy_str(), "bool");
    assert_eq!(DType::BF16.numpy_str(), "bfloat16");
    assert_eq!(DType::C128.numpy_str(), "complex128");
}

#[test]
fn numpy_char_bf16_is_none() {
    assert!(DType::BF16.numpy_char().is_none());
}

#[test]
fn kind_char_correct() {
    assert_eq!(DType::Bool.kind_char(), 'b');
    assert_eq!(DType::I32.kind_char(),  'i');
    assert_eq!(DType::U64.kind_char(),  'u');
    assert_eq!(DType::F32.kind_char(),  'f');
    assert_eq!(DType::C64.kind_char(),  'c');
}
