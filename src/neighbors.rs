#![allow(clippy::used_underscore_binding)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use petal_neighbors as petal;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymodule]
pub fn neighbors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BallTree>()?;
    Ok(())
}

/// A ball tree data structure for efficient nearest neighbor searches.
///
/// The ball tree partitions data points into a nested set of hyperspheres
/// ("balls"), allowing for efficient nearest neighbor queries in
/// multi-dimensional spaces.
#[pyclass]
pub struct BallTree {
    inner: petal::BallTree<'static, f64, petal::distance::Euclidean>,
}

#[pymethods]
impl BallTree {
    /// Creates a new BallTree from a 2D array of points using Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `points` - A 2D numpy array where each row represents a point in the space.
    ///
    /// # Errors
    ///
    /// Returns an error if the input array is empty or has non-contiguous rows.
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(points: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let points = points.as_array().to_owned();
        let inner = petal::BallTree::euclidean(points)
            .map_err(|err| PyValueError::new_err(format!("{err}")))?;
        Ok(BallTree { inner })
    }

    /// Returns the number of points in the tree.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.num_points()
    }

    /// Returns the number of nodes in the tree.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Finds the single nearest neighbor to a query point.
    ///
    /// # Arguments
    ///
    /// * `point` - A 1D numpy array representing the query point.
    ///
    /// # Returns
    ///
    /// A tuple containing the index of the nearest neighbor and its distance.
    #[allow(clippy::needless_pass_by_value)]
    fn query_nearest(&self, point: PyReadonlyArray1<f64>) -> (usize, f64) {
        let point = point.as_array();
        self.inner.query_nearest(&point)
    }

    /// Finds the k nearest neighbors to a query point.
    ///
    /// # Arguments
    ///
    /// * `point` - A 1D numpy array representing the query point.
    /// * `k` - The number of nearest neighbors to find.
    ///
    /// # Returns
    ///
    /// A tuple containing two numpy arrays:
    /// - The indices of the k nearest neighbors.
    /// - The distances to the k nearest neighbors.
    ///
    /// Results are sorted by ascending distance.
    #[allow(clippy::needless_pass_by_value)]
    fn query<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        k: usize,
    ) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>) {
        let point = point.as_array();
        let (indices, distances) = self.inner.query(&point, k);
        (indices.into_pyarray(py), distances.into_pyarray(py))
    }

    /// Finds all neighbors within a given radius of a query point.
    ///
    /// # Arguments
    ///
    /// * `point` - A 1D numpy array representing the query point.
    /// * `radius` - The maximum distance for neighbors to be included.
    ///
    /// # Returns
    ///
    /// A numpy array containing the indices of all points within the radius.
    #[allow(clippy::needless_pass_by_value)]
    fn query_radius<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        radius: f64,
    ) -> Bound<'py, PyArray1<usize>> {
        let point = point.as_array();
        let indices = self.inner.query_radius(&point, radius);
        indices.into_pyarray(py)
    }
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::array;
    use petal_neighbors as petal;

    #[test]
    fn test_ball_tree_construction() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tree = petal::BallTree::euclidean(points).unwrap();
        assert_eq!(tree.num_points(), 4);
    }

    #[test]
    fn test_ball_tree_query_nearest() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tree = petal::BallTree::euclidean(points).unwrap();
        let query = array![0.1, 0.1];
        let (idx, dist) = tree.query_nearest(&query);
        assert_eq!(idx, 0);
        assert!((dist - 0.1_f64.hypot(0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_ball_tree_query_k_nearest() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tree = petal::BallTree::euclidean(points).unwrap();
        let query = array![0.0, 0.0];
        let (indices, distances) = tree.query(&query, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0], 0); // Nearest is the origin itself
        assert!((distances[0] - 0.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_ball_tree_query_radius() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tree = petal::BallTree::euclidean(points).unwrap();
        let query = array![0.0, 0.0];
        let indices = tree.query_radius(&query, 1.1);
        // Should include the origin, (1,0), and (0,1), but not (1,1) which is at distance sqrt(2)
        assert_eq!(indices.len(), 3);
    }
}
