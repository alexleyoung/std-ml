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

    pub fn backward(&mut self, grd_output: &[f64]) {
        let mut dx = grd_output.to_vec();
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
    use crate::layer::Linear;

    fn create_n_layer_network(n: usize) -> Network {
        let mut net = Network::new();
        (0..n)
            .into_iter()
            .for_each(|_| net.add_layer(Box::new(Linear::new(3, 3))));
        net
    }

    #[test]
    fn test_add_layer() {
        let mut net = Network::new();
        let linear = Linear::new(5, 5);
        net.add_layer(Box::new(linear));
        assert!(net.layers.len() == 1);
    }

    #[test]
    fn test_forward() {
        let mut net = create_n_layer_network(5);
    }
}
