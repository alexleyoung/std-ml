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
    fn new() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 1), 5.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    #[should_panic]
    fn new_incorrect_dims() {
        // Should panic: 2x3 matrix needs 6 elements, not 5
        let _ = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn row_major_order() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Should panic: incorrect dimension indexing
        m.get(3, 2);
    }

    #[test]
    fn transpose() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);
        assert_eq!(mt.get(0, 0), 1.0);
        assert_eq!(mt.get(1, 0), 2.0);
        assert_eq!(mt.get(2, 0), 3.0);
        assert_eq!(mt.get(0, 1), 4.0);
        assert_eq!(mt.get(1, 1), 5.0);
        assert_eq!(mt.get(2, 1), 6.0);
    }

    #[test]
    fn transpose_twice() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mtt = m.transpose().transpose();
        assert_eq!(m.rows(), mtt.rows());
        assert_eq!(m.cols(), mtt.cols());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), mtt.get(i, j));
            }
        }
    }

    #[test]
    fn add() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    #[should_panic]
    fn add_incorrect_dims() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = a.add(&b);
    }

    #[test]
    fn sub() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.sub(&b);
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    #[should_panic]
    fn sub_incorrect_dims() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = a.sub(&b);
    }

    #[test]
    fn sub_scale() {
        let mut a = Matrix::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let b = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        a.sub_scale_inplace(&b, 3.0);
        // 10 - 3*2 = 4, 20 - 3*4 = 8, 30 - 3*6 = 12, 40 - 3*8 = 16
        assert_eq!(a.get(0, 0), 4.0);
        assert_eq!(a.get(0, 1), 8.0);
        assert_eq!(a.get(1, 0), 12.0);
        assert_eq!(a.get(1, 1), 16.0);
    }

    #[test]
    #[should_panic]
    fn sub_scale_incorrect_dims() {
        let mut a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.sub_scale_inplace(&b, 1.0);
    }

    #[test]
    fn mat_scale() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let scaled = &m * 2.5;
        assert_eq!(scaled.get(0, 0), 2.5);
        assert_eq!(scaled.get(0, 1), 5.0);
        assert_eq!(scaled.get(1, 0), 7.5);
        assert_eq!(scaled.get(1, 1), 10.0);
    }

    #[test]
    fn mat_mul_vec() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = vec![1.0, 2.0, 3.0];
        let result = m.mul_vec(&v);
        assert_eq!(result.len(), 2);
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(result[0], 14.0);
        assert_eq!(result[1], 32.0);
    }

    #[test]
    #[should_panic]
    fn mat_mul_vec_incorrect_dims() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = vec![1.0, 2.0]; // Wrong length: needs 3, has 2
        let _ = m.mul_vec(&v);
    }

    #[test]
    fn mat_mul_mat() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.mul(&b);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn identity_mul() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let i = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let result = m.mul(&i);
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(result.get(row, col), m.get(row, col));
            }
        }
    }

    #[test]
    #[should_panic]
    fn mat_mul_mat_incorrect_dims() {
        // Can't multiply 2x2 by 3x2
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = a.mul(&b);
    }
}
