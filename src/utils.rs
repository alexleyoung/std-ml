use std::time::SystemTime;
use std::time::UNIX_EPOCH;

/// Creates a pseudo-random number generator
///
/// # Example
/// ```
/// use std_ml::utils::Rng;
/// let mut rng = Rng::new(); // auto-seeded
/// let val = rng.next();
/// assert!(val >= 0.0 && val < 1.0);
/// ```
///
/// # Example
/// ```
/// use std_ml::utils::Rng;
/// let mut rng = Rng::with_seed(42);
/// let val = rng.next();  // Deterministic sequence
/// assert!(val >= 0.0 && val < 1.0);
/// ```
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

pub fn add_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
