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
