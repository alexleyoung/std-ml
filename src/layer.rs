use crate::matrix::Matrix;

pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Matrix, // shape: (out_features, in_features)
    bias: Vec<f64>, // shape: (out_features,)
    // For backprop: store input
    input: Option<Vec<f64>>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        todo!()
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        todo!()
    }

    pub fn forward_batch(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        todo!()
    }

    pub fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        todo!()
    }

    pub fn update(&mut self, learning_rate: f64) {
        todo!()
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

        // Verify input was stored for backprop
        assert!(layer.input.is_some());
        assert_eq!(layer.input.as_ref().unwrap().len(), 2);
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
        let input = vec![1.0]; // Wrong size: should be 2
        let _ = layer.forward(&input);
    }
}
