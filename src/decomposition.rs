use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use petal_decomposition as petal;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pymodule]
fn decomposition(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastICA>()?;
    m.add_class::<PCA>()?;
    Ok(())
}

#[pyclass]
struct FastICA {
    inner: petal::FastIca<f64>,
}

#[pymethods]
impl FastICA {
    #[new]
    fn new() -> Self {
        let inner = petal::FastIca::new();
        FastICA { inner }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        self.inner
            .fit(&x)
            .map_err(|err| PyException::new_err(format!("{}", err)))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn transform(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        self.inner
            .transform(&x)
            .map(|a| a.into_pyarray(py).to_owned())
            .map_err(|err| PyException::new_err(format!("{}", err)))
    }
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

    #[allow(clippy::needless_pass_by_value)]
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        self.inner
            .fit(&x)
            .map_err(|err| PyException::new_err(format!("{}", err)))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn transform(&self, py: Python, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x = x.as_array();
        self.inner
            .transform(&x)
            .map(|a| a.into_pyarray(py).to_owned())
            .map_err(|err| PyException::new_err(format!("{}", err)))
    }
}
