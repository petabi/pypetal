mod decomposition;
mod neighbors;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

/// A Python module implemented in Rust.
#[pymodule]
fn pypetal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(self::decomposition::decomposition))?;
    m.add_wrapped(wrap_pymodule!(self::neighbors::neighbors))?;
    Ok(())
}
