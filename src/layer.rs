use crate::{
    matrix::Matrix,
    utils::{Rng, add_vecs, outer_prod},
};

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
        let mut data = vec![0.0; in_features * out_features];
        Rng::new().fill(&mut data, -0.1, 0.1);

        Self {
            in_features,
            out_features,
            weight: Matrix::new(out_features, in_features, data),
            bias: vec![0.0; out_features],
            grad_weight: Matrix::zeros(out_features, in_features),
            grad_bias: vec![0.0; out_features],
            input: None,
        }
    }

    /// Calculate y = Wx + b to pass forward to following layer
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert!(self.in_features == input.len());

        self.input = Some(input.to_vec());
        let wx = &self.weight * input;
        add_vecs(&wx, &self.bias)
    }

    pub fn forward_batch(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs.iter().map(|x| self.forward(x)).collect()
    }

    /// Back propagation
    ///
    /// Parameters:
    /// [grad_output]: gradient vector w.r.t. this layer's output (size out_features)
    ///
    /// Return:
    /// Weight transpose * [grad_output] for previous layers to use in gradient calculations
    pub fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
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
    pub fn update(&mut self, learning_rate: f64) {
        self.weight
            .sub_scale_inplace(&self.grad_weight, learning_rate);
        for (b, grad) in self.bias.iter_mut().zip(&self.grad_bias) {
            *b -= learning_rate * grad;
        }
    }

    /// Reset gradients for use in between forward batches
    pub fn zero_grad(&mut self) {
        self.grad_weight = Matrix::zeros(self.out_features, self.in_features);
        self.grad_bias = vec![0.0; self.out_features];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_single() {
        // Create layer: 2 input features, 3 output features
        let mut layer = Linear::new(2, 3);

        // Set specific weights for testing
        layer.weight = Matrix::new(
            3,
            2,
            vec![
                1.0, 2.0, // output 0
                3.0, 4.0, // output 1
                5.0, 6.0, // output 2
            ],
        );
        layer.bias = vec![0.1, 0.2, 0.3];

        // Input: [1.0, 2.0]
        let input = vec![1.0, 2.0];
        let output = layer.forward(&input);

        // Manual calculation:
        // out[0] = 1*1 + 2*2 + 0.1 = 1 + 4 + 0.1 = 5.1
        // out[1] = 1*3 + 2*4 + 0.2 = 3 + 8 + 0.2 = 11.2
        // out[2] = 1*5 + 2*6 + 0.3 = 5 + 12 + 0.3 = 17.3
        assert!((output[0] - 5.1).abs() < 1e-10);
        assert!((output[1] - 11.2).abs() < 1e-10);
        assert!((output[2] - 17.3).abs() < 1e-10);
    }

    #[test]
    fn test_linear_forward_batch() {
        let mut layer = Linear::new(2, 2);

        // Set weights
        layer.weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.bias = vec![0.0, 0.0];

        // Batch of 2 examples
        let inputs = vec![
            vec![1.0, 2.0], // First example
            vec![3.0, 4.0], // Second example
        ];

        let outputs = layer.forward_batch(&inputs);

        // Check dimensions
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 2);

        // Verify first example
        // out[0] = 1*1 + 2*2 = 5
        // out[1] = 1*3 + 2*4 = 11
        assert!((outputs[0][0] - 5.0).abs() < 1e-10);
        assert!((outputs[0][1] - 11.0).abs() < 1e-10);

        // Verify second example
        // out[0] = 3*1 + 4*2 = 11
        // out[1] = 3*3 + 4*4 = 25
        assert!((outputs[1][0] - 11.0).abs() < 1e-10);
        assert!((outputs[1][1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_vs_batch_equivalence() {
        // Forward pass should give same result for single example
        // whether we use forward() or forward_batch()
        let mut layer = Linear::new(3, 2);

        // Set reproducible weights
        layer.weight = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        layer.bias = vec![0.5, 0.6];

        let input = vec![1.0, 2.0, 3.0];

        // Single forward
        let single_out = layer.forward(&input);

        // Batch forward (with batch size 1)
        let batch_out = layer.forward_batch(&vec![input.clone()])[0].clone();

        // Should be identical
        for i in 0..2 {
            assert!((single_out[i] - batch_out[i]).abs() < 1e-15);
        }
    }

    #[test]
    #[should_panic]
    fn test_dimension_mismatch() {
        let mut layer = Linear::new(2, 3);
        let input = vec![1.0];
        let _ = layer.forward(&input);
    }

    #[test]
    fn test_backward_gradient_accumulation() {
        let mut layer = Linear::new(2, 2);
        layer.weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.bias = vec![0.0, 0.0];
        layer.zero_grad();

        let input = vec![1.0, 2.0];
        let grad_output = vec![0.5, 1.0];

        layer.forward(&input);
        let grad_input = layer.backward(&grad_output);

        assert_eq!(grad_input.len(), 2);

        assert_eq!(layer.grad_weight.get(0, 0), 0.5);
        assert_eq!(layer.grad_weight.get(0, 1), 1.0);
        assert_eq!(layer.grad_weight.get(1, 0), 1.0);
        assert_eq!(layer.grad_weight.get(1, 1), 2.0);

        assert_eq!(layer.grad_bias[0], 0.5);
        assert_eq!(layer.grad_bias[1], 1.0);
    }

    #[test]
    fn test_backward_multiple_calls_accumulate() {
        let mut layer = Linear::new(2, 2);
        layer.weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.bias = vec![0.0, 0.0];
        layer.zero_grad();

        let input = vec![1.0, 1.0];
        let grad_output = vec![1.0, 1.0];

        layer.forward(&input);
        layer.backward(&grad_output);
        layer.forward(&input);
        layer.backward(&grad_output);

        assert_eq!(layer.grad_weight.get(0, 0), 2.0);
        assert_eq!(layer.grad_bias[0], 2.0);
    }

    #[test]
    fn test_update() {
        let mut layer = Linear::new(2, 2);
        layer.weight = Matrix::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        layer.bias = vec![1.0, 2.0];
        layer.grad_weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.grad_bias = vec![0.5, 0.5];

        layer.update(0.1);

        assert_eq!(layer.weight.get(0, 0), 9.9);
        assert_eq!(layer.weight.get(1, 1), 39.6);
        assert_eq!(layer.bias[0], 0.95);
        assert_eq!(layer.bias[1], 1.95);
    }

    #[test]
    fn test_zero_grad() {
        let mut layer = Linear::new(2, 2);
        layer.grad_weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.grad_bias = vec![5.0, 6.0];

        layer.zero_grad();

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(layer.grad_weight.get(i, j), 0.0);
            }
        }
        assert_eq!(layer.grad_bias, vec![0.0, 0.0]);
    }

    #[test]
    fn test_numerical_gradient_weights() {
        let mut layer = Linear::new(2, 3);
        layer.weight = Matrix::new(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        layer.bias = vec![0.0, 0.0, 0.0];
        layer.zero_grad();

        let input = vec![1.0, 2.0];
        let target = vec![0.5, 0.5, 0.5];

        let output = layer.forward(&input);
        let grad_output: Vec<f64> = output.iter().zip(&target).map(|(o, t)| o - t).collect();
        layer.backward(&grad_output);

        let epsilon = 1e-5;

        for i in 0..3 {
            for j in 0..2 {
                let original = layer.weight.get(i, j);

                layer.weight.set(i, j, original + epsilon);
                let out_plus = layer.forward(&input);
                let loss_plus: f64 = out_plus
                    .iter()
                    .zip(&target)
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f64>()
                    / 2.0;

                layer.weight.set(i, j, original - epsilon);
                let out_minus = layer.forward(&input);
                let loss_minus: f64 = out_minus
                    .iter()
                    .zip(&target)
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f64>()
                    / 2.0;

                layer.weight.set(i, j, original);

                let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic_grad = layer.grad_weight.get(i, j);

                let diff = (numerical_grad - analytic_grad).abs();
                let denom = numerical_grad.abs() + analytic_grad.abs() + 1e-8;
                assert!(
                    diff / denom < 1e-4,
                    "Weight grad mismatch at ({}, {}): numerical={}, analytic={}",
                    i,
                    j,
                    numerical_grad,
                    analytic_grad
                );
            }
        }
    }

    #[test]
    fn test_numerical_gradient_bias() {
        let mut layer = Linear::new(2, 3);
        layer.weight = Matrix::new(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        layer.bias = vec![0.1, 0.2, 0.3];
        layer.zero_grad();

        let input = vec![1.0, 2.0];
        let target = vec![0.5, 0.5, 0.5];

        let output = layer.forward(&input);
        let grad_output: Vec<f64> = output.iter().zip(&target).map(|(o, t)| o - t).collect();
        layer.backward(&grad_output);

        let epsilon = 1e-5;

        for i in 0..3 {
            let original = layer.bias[i];

            layer.bias[i] = original + epsilon;
            let out_plus = layer.forward(&input);
            let loss_plus: f64 = out_plus
                .iter()
                .zip(&target)
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f64>()
                / 2.0;

            layer.bias[i] = original - epsilon;
            let out_minus = layer.forward(&input);
            let loss_minus: f64 = out_minus
                .iter()
                .zip(&target)
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f64>()
                / 2.0;

            layer.bias[i] = original;

            let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic_grad = layer.grad_bias[i];

            let diff = (numerical_grad - analytic_grad).abs();
            let denom = numerical_grad.abs() + analytic_grad.abs() + 1e-8;
            assert!(
                diff / denom < 1e-4,
                "Bias grad mismatch at {}: numerical={}, analytic={}",
                i,
                numerical_grad,
                analytic_grad
            );
        }
    }

    #[test]
    fn test_numerical_gradient_input() {
        let mut layer = Linear::new(2, 3);
        layer.weight = Matrix::new(3, 2, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        layer.bias = vec![0.0, 0.0, 0.0];
        layer.zero_grad();

        let mut input = vec![1.0, 2.0];
        let target = vec![0.5, 0.5, 0.5];

        let output = layer.forward(&input);
        let grad_output: Vec<f64> = output.iter().zip(&target).map(|(o, t)| o - t).collect();
        let grad_input = layer.backward(&grad_output);

        let epsilon = 1e-5;

        for j in 0..2 {
            let original = input[j];

            input[j] = original + epsilon;
            let out_plus = layer.forward(&input);
            let loss_plus: f64 = out_plus
                .iter()
                .zip(&target)
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f64>()
                / 2.0;

            input[j] = original - epsilon;
            let out_minus = layer.forward(&input);
            let loss_minus: f64 = out_minus
                .iter()
                .zip(&target)
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f64>()
                / 2.0;

            input[j] = original;

            let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            let analytic_grad = grad_input[j];

            let diff = (numerical_grad - analytic_grad).abs();
            let denom = numerical_grad.abs() + analytic_grad.abs() + 1e-8;
            assert!(
                diff / denom < 1e-4,
                "Input grad mismatch at {}: numerical={}, analytic={}",
                j,
                numerical_grad,
                analytic_grad
            );
        }
    }

    #[test]
    fn test_numerical_gradient_larger_layer() {
        let mut layer = Linear::new(4, 5);
        layer.weight = Matrix::new(
            5,
            4,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0,
            ],
        );
        layer.bias = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        layer.zero_grad();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let output = layer.forward(&input);
        let grad_output: Vec<f64> = output.iter().zip(&target).map(|(o, t)| o - t).collect();
        layer.backward(&grad_output);

        let epsilon = 1e-5;
        let mut max_error: f64 = 0.0;

        for i in 0..5 {
            for j in 0..4 {
                let original = layer.weight.get(i, j);

                layer.weight.set(i, j, original + epsilon);
                let out_plus = layer.forward(&input);
                let loss_plus: f64 = out_plus
                    .iter()
                    .zip(&target)
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f64>()
                    / 2.0;

                layer.weight.set(i, j, original - epsilon);
                let out_minus = layer.forward(&input);
                let loss_minus: f64 = out_minus
                    .iter()
                    .zip(&target)
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f64>()
                    / 2.0;

                layer.weight.set(i, j, original);

                let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                let analytic_grad = layer.grad_weight.get(i, j);

                let diff = (numerical_grad - analytic_grad).abs();
                let denom = numerical_grad.abs() + analytic_grad.abs() + 1e-8;
                max_error = max_error.max(diff / denom);
            }
        }

        assert!(
            max_error < 1e-4,
            "Max relative error too high: {}",
            max_error
        );
    }
}
