pub trait Activation {
    /// Parameters:
    /// [input]: vector of output from previous layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Parameters:
    /// [grad_output]: gradient vector w.r.t. this activation's output
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64>;

    fn zero_grad(&mut self) {}
}

pub struct ReLU {
    // d/dx = 1 iff x > 0
    output: Option<Vec<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for ReLU {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output: Vec<f64> = input.iter().map(|&x| x.max(0.0)).collect();
        self.output = Some(output.clone());
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
}

pub struct SoftMax {
    // d/dx = class.exp() / classes.exp().sum()
    output: Option<Vec<f64>>,
}

impl SoftMax {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for SoftMax {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        // subtract all input from max for numerical stability
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = input.iter().map(|&x| (x - max_val).exp()).sum();
        let output: Vec<f64> = input
            .iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect();
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, _grad_output: &[f64]) -> Vec<f64> {
        unimplemented!("Use CrossEntropy loss instead");
    }
}

pub struct Sigmoid {
    // d/dx = output (1 - output)
    output: Option<Vec<f64>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Activation for Sigmoid {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output: Vec<f64> = input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        self.output = Some(output.clone());
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let mut relu = ReLU::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let output = relu.forward(&input);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_forward_all_negative() {
        let mut relu = ReLU::new();
        let input = vec![-5.0, -3.0, -1.0];
        let output = relu.forward(&input);

        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_forward_all_positive() {
        let mut relu = ReLU::new();
        let input = vec![1.0, 2.0, 3.0];
        let output = relu.forward(&input);

        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU::new();
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let grad_output = vec![1.0, 1.0, 1.0, 1.0];

        relu.forward(&input);
        let grad_input = relu.backward(&grad_output);

        // derivative is 0 for negative/zero, 1 for positive
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_relu_backward_grad_scaling() {
        let mut relu = ReLU::new();
        let input = vec![1.0, 2.0];
        let grad_output = vec![0.5, 1.5];

        relu.forward(&input);
        let grad_input = relu.backward(&grad_output);

        assert_eq!(grad_input, vec![0.5, 1.5]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![0.0];
        let output = sigmoid.forward(&input);

        // sigmoid(0) = 1 / (1 + 1) = 0.5
        assert!((output[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_forward_large_negative() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![-10.0];
        let output = sigmoid.forward(&input);

        // sigmoid(-10) ≈ 0
        assert!(output[0] < 0.0001);
    }

    #[test]
    fn test_sigmoid_forward_large_positive() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![10.0];
        let output = sigmoid.forward(&input);

        // sigmoid(10) ≈ 1
        assert!(output[0] > 0.9999);
    }

    #[test]
    fn test_sigmoid_output_range() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0];
        let output = sigmoid.forward(&input);

        for o in &output {
            assert!(*o >= 0.0 && *o <= 1.0, "Output {} not in [0,1]", o);
        }
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![0.0];
        let grad_output = vec![1.0];

        sigmoid.forward(&input);
        let grad_input = sigmoid.backward(&grad_output);

        // At x=0, sigmoid=0.5, derivative = 0.5 * 0.5 = 0.25
        assert!((grad_input[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_numerical_gradient() {
        let mut sigmoid = Sigmoid::new();
        let input = vec![0.5, -0.3, 1.2];
        let epsilon = 1e-5;

        // Compute analytical gradient
        sigmoid.forward(&input);
        let grad_output = vec![1.0, 1.0, 1.0];
        let grad_input = sigmoid.backward(&grad_output);

        // Compute numerical gradient
        for i in 0..input.len() {
            let mut input_plus = input.clone();
            let mut input_minus = input.clone();
            input_plus[i] += epsilon;
            input_minus[i] -= epsilon;

            let mut s_plus = Sigmoid::new();
            let mut s_minus = Sigmoid::new();
            let out_plus = s_plus.forward(&input_plus);
            let out_minus = s_minus.forward(&input_minus);

            // Loss is just sum of outputs for gradient check
            let numerical_grad =
                (out_plus.iter().sum::<f64>() - out_minus.iter().sum::<f64>()) / (2.0 * epsilon);

            let diff = (grad_input[i] - numerical_grad).abs();
            let denom = numerical_grad.abs() + grad_input[i].abs() + 1e-8;
            assert!(
                diff / denom < 1e-4,
                "Sigmoid gradient mismatch at {}: numerical={}, analytic={}",
                i,
                numerical_grad,
                grad_input[i]
            );
        }
    }

    #[test]
    fn test_softmax_forward() {
        let mut softmax = SoftMax::new();
        let input = vec![1.0, 2.0, 3.0];
        let output = softmax.forward(&input);

        // Sum should be 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check ratios: exp(1):exp(2):exp(3) = e: e^2: e^3
        // With stability shift: subtract 3
        // exp(-2), exp(-1), exp(0) = 0.135, 0.368, 1.0
        // normalized: 0.135/1.503, 0.368/1.503, 1.0/1.503
        assert!((output[0] - 0.09003057).abs() < 1e-5);
        assert!((output[1] - 0.24472847).abs() < 1e-5);
        assert!((output[2] - 0.66524096).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_forward_single_element() {
        let mut softmax = SoftMax::new();
        let input = vec![5.0];
        let output = softmax.forward(&input);

        assert!((output[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_forward_identical_inputs() {
        let mut softmax = SoftMax::new();
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let output = softmax.forward(&input);

        // All equal = 1/4 each
        for o in &output {
            assert!((o - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_output_sums_to_one() {
        let mut softmax = SoftMax::new();
        let input = vec![-10.0, -5.0, 0.0, 5.0, 10.0, 100.0];
        let output = softmax.forward(&input);

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum was {}", sum);

        // All should be positive
        for o in &output {
            assert!(*o > 0.0, "Output {} should be positive", o);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut softmax = SoftMax::new();
        // Very large values that would overflow without stability trick
        let input = vec![1000.0, 1001.0, 1002.0];
        let output = softmax.forward(&input);

        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Higher input should have higher output
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
    }

    #[test]
    #[should_panic]
    fn test_relu_backward_without_forward() {
        let mut relu = ReLU::new();
        let grad_output = vec![1.0];
        relu.backward(&grad_output);
    }

    #[test]
    #[should_panic]
    fn test_sigmoid_backward_without_forward() {
        let mut sigmoid = Sigmoid::new();
        let grad_output = vec![1.0];
        sigmoid.backward(&grad_output);
    }

    #[test]
    #[should_panic]
    fn test_softmax_backward_without_forward() {
        let mut softmax = SoftMax::new();
        let grad_output = vec![1.0];
        softmax.backward(&grad_output);
    }
}
