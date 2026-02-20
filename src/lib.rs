//! A from-scratch MLP library for MNIST digit classification
//!
//! This library implements a multilayer perceptron using only Rust's standard
//! library.
//!
//! # Structure
//!
//! - `matrix`: Linear algebra operations
//! - `layer`: Neural network layers
//! - `network`: Network composition
//! - `loss`: Loss functions
//! - `data`: Data loading

// Module declarations
pub mod layer;
pub mod matrix;
pub mod utils;

pub use matrix::Matrix;

// ============================================================================
// TEST UTILITIES AND HELPERS
// ============================================================================

#[cfg(test)]
pub mod test_utils {
    /// Creates a small test matrix for quick testing
    ///
    /// # Example
    /// ```
    /// use std_ml::test_utils;
    /// let m = test_utils::create_small_matrix();
    /// assert_eq!(m.rows(), 2);
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn create_small_matrix() -> crate::Matrix {
        crate::Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }

    /// Creates a small test vector for quick testing
    ///
    /// # Example
    /// ```
    /// use std_ml::test_utils;
    /// let v = test_utils::create_small_vector();
    /// assert_eq!(v.len(), 4);
    /// ```
    pub fn create_small_vector() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0]
    }

    /// Compares two floating point values with epsilon tolerance
    ///
    /// # Example
    /// ```
    /// use std_ml::test_utils::approx_eq;
    /// assert!(approx_eq(1.0 / 3.0, 0.3333333333, 1e-9));
    /// ```
    pub fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    /// Compares two vectors of floating point values
    ///
    /// # Example
    /// ```
    /// use std_ml::test_utils::vec_approx_eq;
    /// let a = vec![1.0, 2.0, 3.0];
    /// let b = vec![1.0000001, 2.0000001, 3.0000001];
    /// assert!(vec_approx_eq(&a, &b, 1e-6));
    /// ```
    pub fn vec_approx_eq(a: &[f64], b: &[f64], epsilon: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y, epsilon))
    }
}
