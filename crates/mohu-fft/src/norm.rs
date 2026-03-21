/// FFT normalization modes (matches NumPy's `norm` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Norm {
    /// No scaling on forward; scale by 1/n on backward. NumPy default.
    #[default]
    Backward,
    /// Scale by 1/sqrt(n) on both forward and backward (unitary).
    Ortho,
    /// Scale by 1/n on forward; no scaling on backward.
    Forward,
}
