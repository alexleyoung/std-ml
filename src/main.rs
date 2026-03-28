use std_ml::{
    activation::ReLU,
    layer::Linear,
    loader::DataLoader,
    loss::{CrossEntropy, Loss},
    network::Network,
};

fn main() {
    let mut net = Network::new();
    net.add_layer(Box::new(Linear::new(784, 256)));
    net.add_layer(Box::new(ReLU::new()));
    net.add_layer(Box::new(Linear::new(256, 10)));
    let loss_fn = CrossEntropy {};

    let dataloader = DataLoader::new("", "");

    for (data, targets) in dataloader {
        let out = net.forward(data);
        loss_fn.loss(&out, targets)
    }
}
