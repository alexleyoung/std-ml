use std::time::SystemTime;

use std_ml::{
    Matrix,
    activation::ReLU,
    layer::Linear,
    loader::IDXDataLoader,
    loss::{CrossEntropy, Loss},
    network::Network,
};

fn main() {
    let mut model = Network::new();
    model.add_layer(Box::new(Linear::new(784, 256)));
    model.add_layer(Box::new(ReLU::new()));
    model.add_layer(Box::new(Linear::new(256, 10)));
    let loss_fn = CrossEntropy {};

    let dataloader = IDXDataLoader::new(
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
        64,
    );
    let test_dataloader = IDXDataLoader::new(
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        1,
    );

    let learning_rate = 0.01;
    let epochs = 5;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut sample_count = 0;

        for (data, targets) in dataloader.iter() {
            model.zero_grad();
            let num_samples = data.len() / 784;

            let input_data: Vec<f64> = data.iter().map(|&x| x / 255.0).collect();
            let input = Matrix::new(num_samples, 784, input_data);

            let labels: Vec<f64> = targets.iter().map(|&t| t as f64).collect();

            let start = SystemTime::now();
            let out = model.forward_batch(input);

            epoch_loss += loss_fn.loss_batch(&out, &labels) * num_samples as f64;
            sample_count += num_samples;

            let grad = loss_fn.gradient_batch(&out, &labels);
            model.backward_batch(grad);

            println!(
                "batch time: {}",
                SystemTime::elapsed(&start).unwrap().as_nanos(),
            );

            model.update(learning_rate / num_samples as f64);
        }

        let mut correct = 0;
        let mut total = 0;
        for (data, targets) in test_dataloader.iter() {
            let sample: Vec<f64> = data.iter().map(|&x| x / 255.0).collect();
            let label = targets[0] as usize;
            let out = model.forward(&sample);
            let predicted = out
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            if predicted == label {
                correct += 1;
            }
            total += 1;
        }

        let avg_loss = epoch_loss / sample_count as f64;
        let test_accuracy = correct as f64 / total as f64;
        println!("epoch {epoch}: avg_loss = {avg_loss:.4}");
        println!("epoch {epoch}: test_accuracy = {test_accuracy:.4}");
    }
}
