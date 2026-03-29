use std_ml::{
    activation::ReLU,
    layer::Linear,
    loader::IDXDataLoader,
    loss::{CrossEntropy, Loss},
    network::Network,
};

fn main() {
    let mut net = Network::new();
    net.add_layer(Box::new(Linear::new(784, 256)));
    net.add_layer(Box::new(ReLU::new()));
    net.add_layer(Box::new(Linear::new(256, 10)));
    let loss_fn = CrossEntropy {};

    let dataloader = IDXDataLoader::new(
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
        64,
    );

    for (batch_idx, (data, targets)) in dataloader.iter().enumerate() {
        net.zero_grad();
        let num_samples = data.len() / 784;
        for i in 0..num_samples {
            let sample = &data[i * 784..(i + 1) * 784];
            let label = targets[i] as usize;
            let mut one_hot = vec![0.0; 10];
            one_hot[label] = 1.0;
            let out = net.forward(sample);
            loss_fn.loss(&out, &one_hot);
        }
        println!("batch {batch_idx}: {num_samples} samples");
    }
}
