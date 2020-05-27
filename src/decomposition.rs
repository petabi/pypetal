use numpy::{IntoPyArray, PyArray2};
use petal_decomposition as petal;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;

#[pymodule]
fn decomposition(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PCA>()?;
    Ok(())
}

#[pyclass]
struct PCA {
    inner: petal::Pca<f64>,
}

#[pymethods]
impl PCA {
    #[new]
    fn new(n_components: usize) -> Self {
        let inner = petal::Pca::new(n_components);
        PCA { inner }
    }

    #[getter]
    fn n_components_(&self) -> PyResult<usize> {
        Ok(self.inner.n_components())
    }

    fn fit(&mut self, x: &PyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        self.inner
            .fit(&x)
            .map_err(|err| Exception::py_err(format!("{}", err)))
    }

    fn transform(&self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        self.inner
            .transform(&x)
            .map(|a| a.into_pyarray(py).to_owned())
            .map_err(|err| Exception::py_err(format!("{}", err)))
    }
}
