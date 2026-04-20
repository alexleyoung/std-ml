use crate::{
    matrix::Matrix,
    utils::{Rng, add_vecs, add_vecs_inplace, outer_prod},
};

pub trait Layer {
    /// Transform the input, via weights/biases or activation, and pass forward
    ///
    /// Parameters:
    /// [input]: vector of output from previous layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Transform a batch of inputs
    ///
    /// Parameters:
    /// [input]: batch_size * vector matrix of output from previous layer
    fn forward_batch(&mut self, input: Matrix) -> Matrix;

    /// Calculate dL/dx (error w.r.t the input this layer got) to give to previous layer
    ///
    /// Parameters:
    /// [grad_output]: gradient vector w.r.t. this activation's output
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64>;

    /// Parameters:
    /// [input]: batch_size * grad_output matrix of output from previous layer
    fn backward_batch(&mut self, grad_output: Matrix) -> Matrix;

    /// Parameters:
    /// [learning_rate]: this layer's learning rate
    fn update(&mut self, learning_rate: f64, lambda: f64);

    fn zero_grad(&mut self);
}

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Matrix,              // shape: (out_features, in_features)
    bias: Vec<f64>,              // shape: (out_features,)
    grad_weight: Matrix,         // shape: (out_features, in_features)
    grad_bias: Vec<f64>,         // shape: (out_features,)
    input: Option<Vec<f64>>,     // shape: (in_features,); cache input for back prop
    batch_input: Option<Matrix>, // shape: (batch_size, in_features); cache input for back prop
}

impl Linear {
    /// Simple layer construction with weight and bias initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weights = vec![0.0; in_features * out_features];
        Rng::new().fill(&mut weights, -0.1, 0.1);

        Self {
            in_features,
            out_features,
            weight: Matrix::new(out_features, in_features, weights),
            bias: vec![0.0; out_features],
            grad_weight: Matrix::zeros(out_features, in_features),
            grad_bias: vec![0.0; out_features],
            input: None,
            batch_input: None,
        }
    }

    pub fn set_weight(&mut self, weight: Matrix) {
        self.weight = weight;
    }

    pub fn set_bias(&mut self, bias: Vec<f64>) {
        self.bias = bias;
    }
}

impl Layer for Linear {
    /// Calculate y = Wx + b to pass forward to following layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(self.in_features, input.len());

        let wx = &self.weight * input;

        self.input = Some(input.to_vec());
        add_vecs(&wx, &self.bias)
    }

    fn forward_batch(&mut self, input: Matrix) -> Matrix {
        assert_eq!(self.in_features, input.cols());

        // input:          (batch, in)
        // weight.T:       (in, out)
        // out:            (batch, out)
        let mut out = &input * &self.weight.transpose();
        for i in 0..out.rows() {
            add_vecs_inplace(out.get_row_mut(i), &self.bias);
        }

        self.batch_input = Some(input);
        out
    }

    /// Back propagation
    ///
    /// Parameters:
    /// [grad_output]: gradient vector w.r.t. this layer's output (size out_features)
    ///
    /// Return:
    /// Weight transpose * [grad_output] for previous layers to use in gradient calculations
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let input = self
            .input
            .take()
            .expect("Forward must be called before backward");
        self.grad_weight += outer_prod(grad_output, &input);
        for (b, g) in self.grad_bias.iter_mut().zip(grad_output.iter()) {
            *b += g;
        }
        &self.weight.transpose() * grad_output
    }

    fn backward_batch(&mut self, grad_output: Matrix) -> Matrix {
        let input = self
            .batch_input
            .take()
            .expect("Forward must be called before backward");

        // accumulate gradients from samples
        for i in 0..input.rows() {
            self.grad_weight += outer_prod(grad_output.get_row(i), input.get_row(i));
            for (b, g) in self.grad_bias.iter_mut().zip(grad_output.get_row(i)) {
                *b += g;
            }
        }
        // grad_output:    (batch, out)
        // weight:         (out, in)
        // returned:       (batch, in)
        &grad_output * &self.weight
    }

    /// Update weights and biases
    ///
    /// Parameters:
    /// [learning_rate]: the learning rate of this layer
    /// [lambda]: regularization parameter
    fn update(&mut self, learning_rate: f64, lambda: f64) {
        // subtract gradients
        self.weight
            .sub_scale_inplace(&self.grad_weight, learning_rate);
        for (b, grad) in self.bias.iter_mut().zip(&self.grad_bias) {
            *b -= learning_rate * grad;
        }
        // regularization
        self.weight.iter_mut().for_each(|w| *w -= 2.0 * lambda * *w);
    }

    /// Reset gradients for use in between forward batches
    fn zero_grad(&mut self) {
        self.grad_weight = Matrix::zeros(self.out_features, self.in_features);
        self.grad_bias = vec![0.0; self.out_features];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
