# Contributing to mohu

Thanks for your interest in contributing! mohu is an early-stage project and
contributions at every level are welcome — from small cleanups to new crates.

---

## Finding an issue

Browse open issues and look for ones tagged `good first issue` if you are new
to the codebase. Issues tagged `help wanted` are higher-effort but equally
welcome.

**To claim an issue**, comment `/assign` on it. The bot will assign you
automatically. Each issue supports a maximum of 3 contributors at a time.

If an issue already has 3 assignees, look for another one — there are plenty.

---

## Prerequisites

- Rust stable (see `rust-toolchain.toml` for the pinned version)
- Python 3.10+ (only needed for `mohu-py` development)
- `cargo` tools: `rustfmt`, `clippy` (installed automatically with the toolchain)

---

## Development workflow

```sh
# 1. Fork and clone
git clone https://github.com/<your-username>/mohu.git
cd mohu

# 2. Create a branch
git checkout -b feat/your-feature

# 3. Make changes, then verify
make check   # runs fmt, clippy, and tests

# 4. Open a PR using the template
```

Open a **draft PR** early. This lets maintainers give feedback before you
invest time in a full implementation.

---

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(buffer): add Buffer::is_square() convenience method
fix(buffer): correct null check in BufferSource::DLPack Drop impl
perf(buffer): parallelize tril/triu with Rayon row iterator
refactor(dtype): replace inline matches! with DType::is_float()
doc(examples): add buffer_basics.rs walkthrough
test(buffer): add proptest round-trip for transpose
ci(workflows): pin cargo-deny-action config path
```

**Types:** `feat`, `fix`, `perf`, `refactor`, `doc`, `test`, `chore`, `ci`

**Breaking changes:** append `!` after the type:
```
feat(core)!: rename Array to NdArray
```

---

## Sign-off

All commits must include a sign-off line:

```
Signed-off-by: Your Name <your@email.com>
```

Add it automatically with:

```sh
git commit -s -m "feat(buffer): add Buffer::is_square()"
```

---

## Code style

- `cargo fmt --all` before every commit — no exceptions.
- `cargo clippy --workspace --all-targets -- -D warnings` must pass.
- No `#[allow(clippy::...)]` without a comment explaining why.
- No `unwrap()` in library code — use `MohuResult<T>` and `?`.
- Unsafe blocks must have a `// SAFETY:` comment.

---

## Crate responsibilities

| Crate | Owns |
|---|---|
| `mohu-error` | `MohuError`, `MohuResult<T>`, error codes, reporter |
| `mohu-dtype` | `DType` enum, scalar traits, type promotion, DLPack types |
| `mohu-buffer` | `Buffer`, `Layout`, strides, DLPack import/export |
| `mohu-array` | `NdArray<T>` typed wrapper |
| `mohu-simd` | AVX2, AVX-512, NEON kernel primitives |
| `mohu-ops` | Broadcasting, arithmetic, comparison, logical, reductions |
| `mohu-fft` | FFT, IFFT, RFFT, N-dimensional transforms |
| `mohu-random` | PRNG engines, statistical distributions |
| `mohu-special` | Special math functions (erf, gamma, beta, Bessel) |
| `mohu-stats` | Descriptive statistics, hypothesis tests |
| `mohu-sparse` | COO, CSR, CSC sparse matrix formats |
| `mohu-masked` | Masked arrays, null/invalid value propagation |
| `mohu-io` | .npy/.npz, CSV, Arrow IPC, memory-mapped arrays |
| `mohu-py` | Python module, PyO3 bindings, NumPy buffer protocol |
| `mohu-testing` | Test fixtures, proptest strategies, array comparison |

---

## Running checks

```sh
make test      # all workspace tests
make lint      # clippy (warnings = errors)
make bench     # benchmarks
cargo deny check  # dependency audit
```

---

## PR checklist

Before marking a PR as ready for review:

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] `cargo fmt --all` applied
- [ ] Commits are signed off (`git commit -s`)
- [ ] `CHANGELOG.md` updated (if user-facing change)
- [ ] Benchmarks added or updated (if touching a hot path)

---

## Questions

Open a discussion or leave a comment on the relevant issue. We prefer async
communication over DMs so answers benefit everyone.
