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
    use std::time::Duration;

    const EPSILON: f64 = 10e-6;
    const MIN: f64 = 0.0;
    const MAX: f64 = 10.0;

    fn approx_equal(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn unseeded_rng_changes_rand() {
        let mut rng1 = Rng::new();
        std::thread::sleep(Duration::from_secs(1));
        let mut rng2 = Rng::new();

        assert!(!approx_equal(rng1.next(), rng2.next()));
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut rng1 = Rng::with_seed(0);
        let mut rng2 = Rng::with_seed(0);
        let mut rng3 = Rng::with_seed(1);
        let mut rng4 = Rng::with_seed(1);

        assert!(approx_equal(rng1.next(), rng2.next()));
        assert!(approx_equal(rng3.next(), rng4.next()));
        assert!(!approx_equal(rng1.next(), rng3.next()));
    }

    #[test]
    fn next_range_is_bounded() {
        let mut rng = Rng::with_seed(2);

        for _ in 0..1_000_000 {
            let val = rng.next_range(MIN, MAX);
            assert!(val >= MIN && val < MAX);
        }
    }

    #[test]
    fn next_int_is_bounded() {
        let mut rng = Rng::with_seed(3);
        for _ in 0..1_000_000 {
            let val = rng.next_int(3, 7);
            assert!(val >= 3 && val < 7);
        }
    }

    #[test]
    fn next_int_covers_range() {
        let mut rng = Rng::with_seed(4);
        let mut seen = [false; 4]; // values 3, 4, 5, 6
        for _ in 0..100_000 {
            let val = rng.next_int(3, 7);
            seen[(val - 3) as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "not all values in [3, 7) were generated"
        );
    }

    #[test]
    fn vec_fill_is_random() {
        let mut rng1 = Rng::with_seed(5);
        let mut rng2 = Rng::with_seed(6);
        let mut vec1 = vec![0.0; 1_000_000];
        let mut vec2 = vec![0.0; 1_000_000];

        rng1.fill(&mut vec1, MIN, MAX);
        rng2.fill(&mut vec2, MIN, MAX);
        assert!(vec1.iter().zip(&vec2).any(|(&a, &b)| !approx_equal(a, b)));
    }

    #[test]
    fn fill_respects_bounds() {
        let mut rng = Rng::with_seed(7);
        let mut vec = vec![0.0; 100_000];
        rng.fill(&mut vec, -5.0, 5.0);
        for &v in &vec {
            assert!(v >= -5.0 && v < 5.0);
        }
    }

    #[test]
    fn add_vecs_correctness() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, -1.0, 0.5];
        let result = add_vecs(&a, &b);
        assert!(approx_equal(result[0], 5.0));
        assert!(approx_equal(result[1], 1.0));
        assert!(approx_equal(result[2], 3.5));
    }

    #[test]
    fn add_vecs_commutativity() {
        let a = vec![1.0, -2.0, 3.5];
        let b = vec![0.5, 4.0, -1.0];
        let ab = add_vecs(&a, &b);
        let ba = add_vecs(&b, &a);
        ab.iter()
            .zip(&ba)
            .for_each(|(&x, &y)| assert!(approx_equal(x, y)));
    }

    #[test]
    fn add_vecs_with_zeros() {
        let a = vec![1.0, 2.0, 3.0];
        let zeros = vec![0.0; 3];
        let result = add_vecs(&a, &zeros);
        a.iter()
            .zip(&result)
            .for_each(|(&x, &y)| assert!(approx_equal(x, y)));
    }

    #[test]
    fn outer_prod_shape() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let m = outer_prod(&a, &b);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);
    }

    #[test]
    fn outer_prod_values() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let m = outer_prod(&a, &b);
        // m[i][j] == a[i] * b[j]
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                assert!(approx_equal(m.get(i, j), ai * bj));
            }
        }
    }

    #[test]
    fn outer_prod_with_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let zeros = vec![0.0, 0.0];
        let m = outer_prod(&a, &zeros);
        for i in 0..a.len() {
            for j in 0..zeros.len() {
                assert!(approx_equal(m.get(i, j), 0.0));
            }
        }
    }
}
