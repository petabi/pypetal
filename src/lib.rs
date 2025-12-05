use pyo3::prelude::*;

mod decomposition;

#[pymodule]
fn _petal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&decomposition::module(m.py())?)?;

    Ok(())
}
