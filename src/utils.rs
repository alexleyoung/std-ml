use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crate::Matrix;

/// A pseudo-random number generator
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new() -> Self {
        let state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self { state }
    }

    /// Create new RNG with seed
    pub fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next value in [0, 1)
    pub fn next(&mut self) -> f64 {
        // Simple LCG: X_{n+1} = (a * X_n + c) % m
        // Using constants from numerical recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state as f64) / (u64::MAX as f64)
    }

    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        self.next() * range + min
    }

    /// Generate integer in [min, max)
    pub fn next_int(&mut self, min: i64, max: i64) -> i64 {
        let range = max - min;
        (self.next() * range as f64) as i64 + min
    }

    /// Fill vector with random values in [min, max)
    pub fn fill(&mut self, vec: &mut [f64], min: f64, max: f64) {
        for v in vec.iter_mut() {
            *v = self.next_range(min, max)
        }
    }
}

/// Add two vecs and return a new vec
pub fn add_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Take the outer product of two vecs and return the result
pub fn outer_prod(a: &[f64], b: &[f64]) -> Matrix {
    let mut data = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            data.push(x * y);
        }
    }
    Matrix::new(a.len(), b.len(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_with_seed_deterministic() {
        let mut rng1 = Rng::with_seed(42);
        let mut rng2 = Rng::with_seed(42);

        for _ in 0..10 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_rng_next_range() {
        let mut rng = Rng::with_seed(42);
        for _ in 0..100 {
            let val = rng.next_range(-5.0, 5.0);
            assert!(val >= -5.0 && val < 5.0);
        }
    }

    #[test]
    fn test_rng_next_int() {
        let mut rng = Rng::with_seed(42);
        for _ in 0..100 {
            let val = rng.next_int(0, 10);
            assert!(val >= 0 && val < 10);
        }
    }

    #[test]
    fn test_rng_fill() {
        let mut rng = Rng::with_seed(42);
        let mut v = vec![0.0; 10];
        rng.fill(&mut v, -1.0, 1.0);

        for val in &v {
            assert!(*val >= -1.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_add_vecs() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = add_vecs(&a, &b);

        assert_eq!(c, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_vecs_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let c = add_vecs(&a, &b);

        assert!(c.is_empty());
    }

    #[test]
    fn test_outer_prod() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let m = outer_prod(&a, &b);

        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);

        assert_eq!(m.get(0, 0), 3.0);
        assert_eq!(m.get(0, 1), 4.0);
        assert_eq!(m.get(0, 2), 5.0);
        assert_eq!(m.get(1, 0), 6.0);
        assert_eq!(m.get(1, 1), 8.0);
        assert_eq!(m.get(1, 2), 10.0);
    }

    #[test]
    fn test_outer_prod_identity_case() {
        let a = vec![1.0];
        let b = vec![1.0];
        let m = outer_prod(&a, &b);

        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 1);
        assert_eq!(m.get(0, 0), 1.0);
    }
}
