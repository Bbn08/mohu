use crate::MohuError;

/// An iterator that walks the full source chain of a `MohuError`.
///
/// The first item yielded is always the error itself. Subsequent items
/// are the `source` fields of any [`MohuError::Context`] wrappers,
/// unwrapped until a non-`Context` variant is reached.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, context::ResultExt, chain::ErrorChain};
/// let inner = MohuError::SingularMatrix;
/// let wrapped = Err::<(), _>(inner)
///     .context("computing inverse")
///     .context("normalising weights")
///     .unwrap_err();
///
/// for (depth, cause) in ErrorChain::new(&wrapped).enumerate() {
///     eprintln!("{:>2}: {cause}", depth);
/// }
/// // 0: normalising weights: computing inverse: singular matrix: ...
/// // 1: computing inverse: singular matrix: ...
/// // 2: singular matrix: ...
/// ```
pub struct ErrorChain<'a> {
    current: Option<&'a MohuError>,
}

impl<'a> ErrorChain<'a> {
    /// Creates a new chain iterator starting at `err`.
    pub fn new(err: &'a MohuError) -> Self {
        Self { current: Some(err) }
    }

    /// Returns the root cause of the error — the last (innermost) item
    /// in the chain.
    ///
    /// This is O(depth) but depth is almost always tiny (< 10).
    pub fn root(err: &'a MohuError) -> &'a MohuError {
        let mut cur = err;
        while let MohuError::Context { source, .. } = cur {
            cur = source.as_ref();
        }
        cur
    }

    /// Collects all context messages (the string portion of every
    /// `Context` wrapper) from outermost to innermost.
    ///
    /// The underlying non-`Context` error is **not** included.
    pub fn context_messages(err: &'a MohuError) -> Vec<&'a str> {
        let mut msgs = Vec::new();
        let mut cur = err;
        while let MohuError::Context { context, source } = cur {
            msgs.push(context.as_str());
            cur = source.as_ref();
        }
        msgs
    }

    /// Returns the depth of the context chain.
    ///
    /// An unwrapped error has depth 0. Each `.context()` call adds 1.
    pub fn depth(err: &'a MohuError) -> usize {
        let mut d = 0usize;
        let mut cur = err;
        while let MohuError::Context { source, .. } = cur {
            d += 1;
            cur = source.as_ref();
        }
        d
    }
}

impl<'a> Iterator for ErrorChain<'a> {
    type Item = &'a MohuError;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        // Advance to the inner source if we're sitting on a Context wrapper.
        self.current = if let MohuError::Context { source, .. } = current {
            Some(source.as_ref())
        } else {
            None
        };
        Some(current)
    }
}

impl std::iter::FusedIterator for ErrorChain<'_> {}

// ─── convenience methods wired into MohuError ────────────────────────────────

impl MohuError {
    /// Returns an iterator over the full source chain of this error.
    ///
    /// See [`ErrorChain`] for details and examples.
    pub fn chain(&self) -> ErrorChain<'_> {
        ErrorChain::new(self)
    }

    /// Returns the root (innermost) cause of this error, unwrapping all
    /// `Context` wrappers.
    pub fn root_cause(&self) -> &MohuError {
        ErrorChain::root(self)
    }

    /// Returns the depth of the `Context` wrapper chain.
    ///
    /// 0 means this error has no context wrappers at all.
    pub fn chain_depth(&self) -> usize {
        ErrorChain::depth(self)
    }

    /// Returns all context message strings from outermost to innermost.
    pub fn context_messages(&self) -> Vec<&str> {
        ErrorChain::context_messages(self)
    }
}
