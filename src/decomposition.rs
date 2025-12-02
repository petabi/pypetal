#![allow(clippy::used_underscore_binding)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use petal_decomposition as petal;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pymodule]
pub fn decomposition(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastIca>()?;
    m.add_class::<Pca>()?;
    Ok(())
}

#[pyclass]
struct FastIca {
    inner: petal::FastIca<f64>,
}

#[pymethods]
impl FastIca {
    #[new]
    fn new() -> Self {
        let inner = petal::FastIca::new();
        FastIca { inner }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        self.inner
            .fit(&x)
            .map_err(|err| PyException::new_err(format!("{err}")))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn transform(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        self.inner
            .transform(&x)
            .map(|a| a.into_pyarray(py).unbind())
            .map_err(|err| PyException::new_err(format!("{err}")))
    }
}

#[pyclass]
struct Pca {
    inner: petal::Pca<f64>,
}

#[pymethods]
impl Pca {
    #[new]
    fn new(n_components: usize) -> Self {
        let inner = petal::Pca::new(n_components);
        Pca { inner }
    }

    #[getter]
    fn n_components_(&self) -> usize {
        self.inner.n_components()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        self.inner
            .fit(&x)
            .map_err(|err| PyException::new_err(format!("{err}")))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn transform(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        self.inner
            .transform(&x)
            .map(|a| a.into_pyarray(py).unbind())
            .map_err(|err| PyException::new_err(format!("{err}")))
    }
}
