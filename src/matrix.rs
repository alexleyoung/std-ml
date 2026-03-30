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
}
