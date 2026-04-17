use crate::Matrix;

pub trait Loss {
    fn loss(&self, p: &[f64], y: &[f64]) -> f64;
    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64>;
    fn loss_batch(&self, p: &Matrix, y: &[f64]) -> f64;
    fn gradient_batch(&self, p: &Matrix, y: &[f64]) -> Matrix;
}

pub struct MSE {}

impl Loss for MSE {
    // Calculate the mean squared error
    fn loss(&self, p: &[f64], y: &[f64]) -> f64 {
        let n = p.len() as f64;
        assert!(
            n as usize == y.len(),
            "Prediction size should match truth size"
        );

        let mut loss = 0.0;
        p.iter()
            .zip(y.iter())
            .for_each(|(&p, &y)| loss += (p - y).powf(2.0));
        loss / n
    }

    fn loss_batch(&self, p: &Matrix, y: &[f64]) -> f64 {
        assert!(
            p.cols() == y.len(),
            "Prediction size should match truth size"
        );

        let mut loss = 0.0;
        p.iter()
            .zip(y.iter())
            .for_each(|(&p, &y)| loss += (p - y).powf(2.0));
        loss / p.cols() as f64
    }

    // calculate dL/dp
    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64> {
        let n = p.len() as f64;
        assert!(
            n as usize == y.len(),
            "Prediction size should match truth size"
        );

        p.iter()
            .zip(y.iter())
            .map(|(&p, &y)| 2.0 / n * (p - y))
            .collect()
    }

    fn gradient_batch(&self, p: &Matrix, y: &[f64]) -> Matrix {
        let n = p.cols() as f64;
        assert!(
            n as usize == y.len(),
            "Prediction size should match truth size"
        );

        Matrix::new(
            p.rows(),
            p.cols(),
            p.iter()
                .zip(y.iter())
                .map(|(&p, &y)| 2.0 / n * (p - y))
                .collect(),
        )
    }
}

pub struct CrossEntropy {}

impl Loss for CrossEntropy {
    fn loss(&self, p: &[f64], y: &[f64]) -> f64 {
        assert!(
            p.len() == y.len(),
            "Prediction size should match truth size"
        );

        let max_val = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = p.iter().map(|&x| (x - max_val).exp()).sum();

        -(p.iter()
            .zip(y.iter())
            .map(|(&x, &y)| {
                let px = (x - max_val).exp() / exp_sum;
                y * px.ln()
            })
            .sum::<f64>())
    }

    fn loss_batch(&self, p: &Matrix, y: &[f64]) -> f64 {
        assert_eq!(p.rows(), y.len());
        let total: f64 = (0..p.rows())
            .map(|i| {
                let row = p.get_row(i);
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
                let label = y[i] as usize;
                let px = (row[label] - max_val).exp() / exp_sum;
                -px.ln()
            })
            .sum();
        total / p.rows() as f64
    }

    fn gradient(&self, p: &[f64], y: &[f64]) -> Vec<f64> {
        assert!(
            p.len() == y.len(),
            "Prediction size should match truth size"
        );

        let max_val = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = p.iter().map(|&x| (x - max_val).exp()).sum();

        p.iter()
            .zip(y.iter())
            .map(|(&x, &y)| {
                let px = (x - max_val).exp() / exp_sum;
                px - y
            })
            .collect()
    }

    fn gradient_batch(&self, p: &Matrix, y: &[f64]) -> Matrix {
        assert_eq!(p.rows(), y.len());
        let mut data = Vec::with_capacity(p.rows() * p.cols());
        for i in 0..p.rows() {
            let row = p.get_row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let label = y[i] as usize;
            for (j, &x) in row.iter().enumerate() {
                let px = (x - max_val).exp() / exp_sum;
                data.push(px - if j == label { 1.0 } else { 0.0 });
            }
        }
        Matrix::new(p.rows(), p.cols(), data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
