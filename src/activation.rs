pub trait Activation {
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;
    fn backward(&mut self, grad_input: &[f64]) -> Vec<f64>;
    fn zero_grad(&mut self) {}
}

pub struct ReLU {
    output: Option<Vec<f64>>,
}

impl Activation for ReLU {}
