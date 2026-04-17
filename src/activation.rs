use crate::Layer;
use crate::Matrix;

pub struct ReLU {
    // d/dx = 1 iff x > 0
    output: Option<Vec<f64>>,
    batch_output: Option<Matrix>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            output: None,
            batch_output: None,
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output: Vec<f64> = input.iter().map(|&x| x.max(0.0)).collect();
        self.output = Some(output.clone());
        output
    }

    fn forward_batch(&mut self, input: Matrix) -> Matrix {
        let output = Matrix::new(
            input.rows(),
            input.cols(),
            input.iter().map(|&x| x.max(0.0)).collect(),
        );
        self.batch_output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let output = self
            .output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter()
            .zip(&output)
            .map(|(&g, &o)| g * if o > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }

    fn backward_batch(&mut self, mut grad_output: Matrix) -> Matrix {
        let output = self
            .batch_output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter_mut()
            .zip(output.iter())
            .for_each(|(g, &o)| *g *= if o > 0.0 { 1.0 } else { 0.0 });
        grad_output
    }

    fn update(&mut self, _: f64) {}

    fn zero_grad(&mut self) {}
}

pub struct LeakyReLU {
    // d/dx = 1 iff x > 0
    alpha: f64,
    output: Option<Vec<f64>>,
    batch_output: Option<Matrix>,
}

impl LeakyReLU {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            output: None,
            batch_output: None,
        }
    }
}

impl Layer for LeakyReLU {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output: Vec<f64> = input
            .iter()
            .map(|&x| if x > 0.0 { x } else { self.alpha * x })
            .collect();
        self.output = Some(output.clone());
        output
    }

    fn forward_batch(&mut self, input: Matrix) -> Matrix {
        let output = Matrix::new(
            input.rows(),
            input.cols(),
            input
                .iter()
                .map(|&x| if x > 0.0 { x } else { self.alpha * x })
                .collect(),
        );
        self.batch_output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let output = self
            .output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter()
            .zip(&output)
            .map(|(&g, &o)| g * if o > 0.0 { 1.0 } else { self.alpha })
            .collect()
    }

    fn backward_batch(&mut self, mut grad_output: Matrix) -> Matrix {
        let output = self
            .batch_output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter_mut()
            .zip(output.iter())
            .for_each(|(g, &o)| *g *= if o > 0.0 { 1.0 } else { self.alpha });
        grad_output
    }

    fn update(&mut self, _: f64) {}

    fn zero_grad(&mut self) {}
}

pub struct Sigmoid {
    // d/dx = output (1 - output)
    output: Option<Vec<f64>>,
    batch_output: Option<Matrix>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            output: None,
            batch_output: None,
        }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output: Vec<f64> = input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        self.output = Some(output.clone());
        output
    }

    fn forward_batch(&mut self, input: Matrix) -> Matrix {
        let output = Matrix::new(
            input.rows(),
            input.cols(),
            input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
        );
        // let output: Vec<f64> = input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        self.batch_output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let output = self
            .output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter()
            .zip(&output)
            .map(|(&g, &o)| g * o * (1.0 - o))
            .collect()
    }

    fn backward_batch(&mut self, mut grad_output: Matrix) -> Matrix {
        let output = self
            .batch_output
            .take()
            .expect("Forward must be called before backward");

        grad_output
            .iter_mut()
            .zip(output.iter())
            .for_each(|(g, &o)| *g = *g * o * (1.0 - o));
        grad_output
    }

    fn update(&mut self, _: f64) {}

    fn zero_grad(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
}
