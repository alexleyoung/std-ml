use crate::Layer;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut out = input.to_vec();
        for layer in &mut self.layers {
            out = layer.forward(&out);
        }
        out
    }

    pub fn backward(&mut self, grad_output: &[f64]) {
        let mut dx = grad_output.to_vec();
        for layer in self.layers.iter_mut().rev() {
            dx = layer.backward(&dx);
        }
    }

    pub fn update(&mut self, learning_rate: f64) {
        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation::ReLU, layer::Linear, matrix::Matrix};

    #[test]
    fn test_add_layer() {
        let mut net = Network::new();
        let linear = Linear::new(5, 5);
        net.add_layer(Box::new(linear));
        assert!(net.layers.len() == 1);
    }

    #[test]
    fn test_forward_output_shape_and_activation() {
        let mut l1 = Linear::new(3, 3);
        l1.set_weight(Matrix::new(
            3,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ));
        l1.set_bias(vec![0.0, 0.0, 0.0]);

        let mut l3 = Linear::new(3, 1);
        l3.set_weight(Matrix::new(1, 3, vec![1.0, 1.0, 1.0]));
        l3.set_bias(vec![0.0]);

        let mut n = Network::new();
        n.add_layer(Box::new(l1));
        n.add_layer(Box::new(ReLU::new()));
        n.add_layer(Box::new(l3));

        let out = n.forward(&[-1.0, 0.0, 1.0]);

        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-10);
    }
}
