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
    model.add_layer(Box::new(Linear::new(784, 128)));
    model.add_layer(Box::new(ReLU::new()));
    model.add_layer(Box::new(Linear::new(128, 10)));
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

    println!(
        "{:>5} | {:>10} | {:>10} | {:>8} | {:>8}",
        "epoch", "loss", "accuracy", "time", "total"
    );
    println!("{}", "-".repeat(55));

    let total_start = SystemTime::now();
    for epoch in 0..epochs {
        let epoch_start = SystemTime::now();
        let mut epoch_loss = 0.0;
        let mut sample_count = 0;

        for (data, targets) in dataloader.iter() {
            model.zero_grad();
            let num_samples = data.len() / 784;

            let input_data: Vec<f64> = data.iter().map(|&x| x / 255.0).collect();
            let input = Matrix::new(num_samples, 784, input_data);

            let labels: Vec<f64> = targets.iter().map(|&t| t as f64).collect();

            let out = model.forward_batch(input);

            epoch_loss += loss_fn.loss_batch(&out, &labels) * num_samples as f64;
            sample_count += num_samples;

            let grad = loss_fn.gradient_batch(&out, &labels);
            model.backward_batch(grad);

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
        let elapsed = SystemTime::elapsed(&epoch_start).unwrap().as_secs_f32();
        let total_elapsed = SystemTime::elapsed(&total_start).unwrap().as_secs_f32();
        println!(
            "{:>5} | {:>10.4} | {:>9.2}% | {:>7.2}s | {:>7.2}s",
            epoch,
            avg_loss,
            test_accuracy * 100.0,
            elapsed,
            total_elapsed,
        );
    }
}
