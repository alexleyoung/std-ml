use crate::{
    matrix::Matrix,
    utils::{Rng, add_vecs, outer_prod},
};

pub trait Layer {
    /// Transform the input, via weights/biases or activation, and pass forward
    ///
    /// Parameters:
    /// [input]: vector of output from previous layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Calculate dL/dx (error w.r.t the input this layer got) to give to previous layer
    ///
    /// Parameters:
    /// [grad_output]: gradient vector w.r.t. this activation's output
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64>;

    /// Parameters:
    /// [learning_rate]: this layer's learning rate
    fn update(&mut self, learning_rate: f64);

    fn zero_grad(&mut self);
}

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Matrix,          // shape: (out_features, in_features)
    bias: Vec<f64>,          // shape: (out_features,)
    grad_weight: Matrix,     // shape: (out_features, in_features)
    grad_bias: Vec<f64>,     // shape: (out_features,)
    input: Option<Vec<f64>>, // shape: (in_features,); cache input for back prop
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
        }
    }

    pub fn set_weight(&mut self, weight: Matrix) {
        self.weight = weight;
    }

    pub fn set_bias(&mut self, bias: Vec<f64>) {
        self.bias = bias;
    }

    pub fn forward_batch(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs.iter().map(|x| self.forward(x)).collect()
    }
}

impl Layer for Linear {
    /// Calculate y = Wx + b to pass forward to following layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(self.in_features, input.len());

        self.input = Some(input.to_vec());
        let wx = &self.weight * input;
        add_vecs(&wx, &self.bias)
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

    /// Update weights and biases
    ///
    /// Parameters:
    /// [learning_rate]: the learning rate of this layer
    fn update(&mut self, learning_rate: f64) {
        self.weight
            .sub_scale_inplace(&self.grad_weight, learning_rate);
        for (b, grad) in self.bias.iter_mut().zip(&self.grad_bias) {
            *b -= learning_rate * grad;
        }
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
