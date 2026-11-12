use pyo3::prelude::*;

mod decomposition;

#[pymodule(name = "_petal")]
fn petal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(&decomposition::module(m.py())?)?;

    Ok(())
}
