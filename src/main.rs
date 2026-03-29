use std_ml::{
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

            for i in 0..num_samples {
                let sample: Vec<f64> = data[i * 784..(i + 1) * 784]
                    .iter()
                    .map(|&x| x / 255.0)
                    .collect();
                let label = targets[i] as usize;
                let mut one_hot = vec![0.0; 10];
                one_hot[label] = 1.0;

                let out = model.forward(&sample);
                epoch_loss += loss_fn.loss(&out, &one_hot);

                let grad = loss_fn.gradient(&out, &one_hot);
                model.backward(&grad);

                sample_count += 1;
            }

            model.update(learning_rate / num_samples as f64);
        }

        let mut correct = 0;
        let mut total = 0;
        for (data, targets) in test_dataloader.iter() {
            let num_samples = data.len() / 784;
            for i in 0..num_samples {
                let sample: Vec<f64> = data[i * 784..(i + 1) * 784]
                    .iter()
                    .map(|&x| x / 255.0)
                    .collect();
                let label = targets[i] as usize;
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
        }

        let avg_loss = epoch_loss / sample_count as f64;
        let test_accuracy = correct as f64 / total as f64;
        println!("epoch {epoch}: avg_loss = {avg_loss:.4}");
        println!("epoch {epoch}: test_accuracy = {test_accuracy:.4}");
    }
}
