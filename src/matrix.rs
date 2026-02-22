use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub};

/// A matrix stored in row-major order
/// Data is a (flat) Vec<f64> of size rows * cols
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert!(rows * cols == data.len());
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    pub fn transpose(&self) -> Matrix {
        let mut res = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j * res.cols + i] = self.data[i * self.cols + j];
            }
        }
        res
    }

    /// Math Operators
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    pub fn add_inplace(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x - y)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    // In-place subtraction of a Matrix [other] scaled by scalar [scale]
    pub fn sub_scale_inplace(&mut self, other: &Matrix, scale: f64) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a -= scale * b;
        }
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut res = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    // directly index to avoid unnecessary assertions from
                    // get() and set(). compiler will guaranteed skip bounds check
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                res.set(i, j, sum);
            }
        }
        res
    }

    pub fn mul_vec(&self, vec: &[f64]) -> Vec<f64> {
        assert!(self.cols == vec.len());
        (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| self.data[i * self.cols + j] * vec[j])
                    .sum()
            })
            .collect()
    }
}

// &Matrix + &Matrix
impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.add(rhs)
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        self.add_inplace(&rhs);
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.sub(rhs)
    }
}

// &Matrix * f64
impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Matrix {
        let data = self.data.iter().map(|x| x * rhs).collect();
        Matrix::new(self.rows, self.cols, data)
    }
}

// &Matrix * &[f64]
impl Mul<&[f64]> for &Matrix {
    type Output = Vec<f64>;
    fn mul(self, rhs: &[f64]) -> Vec<f64> {
        self.mul_vec(rhs)
    }
}

// &Matrix * &Matrix
impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        self.mul(rhs)
    }
}

// Display for pretty printing
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix {}x{} [\n", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let m = Matrix::new(2, 3, vec![1.0; 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
    }

    #[test]
    #[should_panic]
    fn test_new_wrong_size() {
        let _ = Matrix::new(2, 3, vec![1.0; 5]); // Should be 6
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::new(2, 2, vec![0.0; 4]);
        m.set(1, 1, 42.0);
        assert_eq!(m.get(1, 1), 42.0);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let m = Matrix::new(2, 2, vec![0.0; 4]);
        let _ = m.get(2, 0);
    }

    #[test]
    fn test_addition() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = &a + &b;

        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_scalar_multiply() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a * 2.5;

        assert_eq!(c.get(0, 0), 2.5);
        assert_eq!(c.get(1, 1), 10.0);
    }

    #[test]
    fn test_multiply_2x2() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.mul(&b);

        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_multiply_non_square() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.mul(&b);

        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    #[should_panic]
    fn test_multiply_dimension_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0; 4]);
        let b = Matrix::new(3, 2, vec![1.0; 6]);
        let _ = a.mul(&b);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_transpose_twice() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose().transpose();

        for i in 0..m.rows() {
            for j in 0..m.cols() {
                assert!((m.get(i, j) - t.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m.get(i, j), 0.0);
            }
        }
    }

    #[test]
    fn test_display() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let _ = format!("{}", m);
    }

    #[test]
    fn test_add_inplace() {
        let mut a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        a.add_inplace(&b);

        assert_eq!(a.get(0, 0), 6.0);
        assert_eq!(a.get(0, 1), 8.0);
        assert_eq!(a.get(1, 0), 10.0);
        assert_eq!(a.get(1, 1), 12.0);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.sub(&b);

        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_sub_operator() {
        let a = Matrix::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a - &b;

        assert_eq!(c.get(0, 0), 9.0);
        assert_eq!(c.get(1, 1), 36.0);
    }

    #[test]
    fn test_sub_scale_inplace() {
        let mut a = Matrix::new(2, 2, vec![10.0, 10.0, 10.0, 10.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        a.sub_scale_inplace(&b, 2.0);

        assert_eq!(a.get(0, 0), 8.0);
        assert_eq!(a.get(0, 1), 6.0);
        assert_eq!(a.get(1, 0), 4.0);
        assert_eq!(a.get(1, 1), 2.0);
    }

    #[test]
    fn test_add_assign() {
        let mut a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        a += b;

        assert_eq!(a.get(0, 0), 6.0);
        assert_eq!(a.get(1, 1), 12.0);
    }

    #[test]
    fn test_mul_vec() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = vec![1.0, 2.0, 3.0];
        let result = m.mul_vec(&v);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 14.0);
        assert_eq!(result[1], 32.0);
    }

    #[test]
    fn test_mul_vec_operator() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let v = vec![1.0, 2.0];
        let result = &m * &v[..];

        assert_eq!(result, vec![5.0, 11.0]);
    }
}
