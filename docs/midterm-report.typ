#set page(paper: "us-letter", margin: 1in)
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Midterm Report: std-ml] \
  #text(
    size: 14pt,
    style: "italic",
  )[A Standard-Library-Only Neural Network Framework in Rust]
]

#v(1em)

#text(weight: "bold")[Abstract] \
This project focuses on the ground-up implementation of a Multilayer Perceptron (MLP) neural network using the Rust
programming language. The primary motivation is to gain a deep practical understanding of neural networks and their
implentations like in PyTorch or TensorFlow. Things like the mechanics of fast matrix operations, backpropagation, and
gradient descent are all discussed greatly in theory but are often abstracted away from us as users. Using the MNIST
handwritten digit dataset as a benchmark, I've developed a simple, modular MLP library consisting of custom matrix
primitives, layers, activation functions, loss functions, and a data loader. Preliminary results indicate a 92%
classification accuracy. While functional, the current implementation faces performance bottlenecks in matrix
multiplication and subpar accuracy. This report details the current architecture and outlines future plans for
optimization using BLAS-inspired techniques and the potential development of a Rust-based visualizer.

#v(2em)

= Introduction
In today's tech landscape, it is easier than ever to ignore the decades of math and computer science fundamentals which
have led to technology like modern LLMs. To better understand machine learning fundamentals, I am building an MLP
from scratch (STD Library only) in Rust.

The primary machine learning task is multi-class classification using the *MNIST dataset*,
which consists of 70,000 grayscale images of handwritten digits #cite(<lecun-mnist>). The
library can currently support a fully connected feed-forward network. By avoiding external
ML libraries, I'm better able to understand both API design decisions, and performance
implications at every level, from random bias initialization to matrix multiplication to
zero-copy dataloader iteration.

= Related Work
There is no shortage of (significantly better) multi-purpose machine learning libraries that
currently exist. Being the current SOTA (as far as I'm aware), I often reference PyTorch #cite(<pytorch>),
especially when designing parts of my library's API. I have also begun looking into
BLAS/BLIS/FLAME, historic and educational programmatic linear algebra standards and papers
#cite(<blis>) #cite(<flame-gemm>), as I aim to optimize my matrix multiplication to
accelerate model training.

= Methods
My implementation, `std-ml`, is a Rust std-lib only neural netwoork library designed with modularity currently
supporting multilayer perceptron models. Each module is placed in its own aptly-named file in the `src/` directory.

== matrix.rs
At the very core of machine learning is matrices. My implementation uses flat `Vec<f64>` to prevent unnecessary memory
fetching and ensure cache locality, with strides used to handle indexing. Basic arithmetic was implemented as necessary,
often with variations depending on ownership: transpose, matrix addition, scaling, and of course
multiplication. Rust also offers convenient operator traits which I've implemented to enable traditional operator use
with the custom matrices. Matrix multiplication is currently implemented naively.

== layer.rs
To populate networks, we need layers. I define a general `Layer` trait as well as a linear implementation of said trait
in this module to enable basic linear models.

==== Layer
The `Layer` trait represents any single layer in a `network`. It defines four methods:

`forward(&[f64])->Vec<f64>`

`backward(&[f64])->Vec<f64>`

`update(f64)`

`zero_grad()`

Forward and backward are the calculations done on their respective passes; they both take a vector of shape
`in_features` and `out_features` respectively, and output `out_features` and `in_features` respectively. `update(f64)`
takes a learning rate, and applies its gradients to its weights and biases. `zero_grad()` is a helper to reset gradients
during batched training.

==== Linear
Currently implemented is the `Linear` layer. On the forward pass, output is calculated via $y=bold(W x+b)$. The backward
pass calculates the gradients of loss (passed as arg) w.r.t. its weights and biases and returns the gradient of loss
w.r.t. its input for the previous layer.

== activation.rs
(Non-linear) Activations are necessary to increase model complexity and enable learning. The `activation` module
contains implementations for common activation functions like `ReLU` and `Sigmoid`. All activations implement `Layer`
for composability in `network`s. Note, only `forward()` (activation) and `backward()` (gradient) are implemented,
though, since `update()` and `zero_grad()` are not relevant for activations.

== network.rs
With layers and activations defined, we need something to compose them together. The `network` module holds the
`Network` struct, which stores a `Vec<Box<dyn Layer>>` to support heterogeneous layer types. It exposes a `forward`,
`backward`, `update`, and `zero_grad` that simply iterate through its layers in order (forward pass) or reverse
(backward pass). Note, this does incur potentially unnecessary overhead from heap allocation and dynamic disaptching of
the `Layer`s. I may rework this to be a generic multi-type struct where the user can pick a layer type and activation.

== loss.rs
Training requires a way to measure error. The module provides both the loss value computation and its gradient w.r.t.
predictions, which gets fed into the network's backward pass. I've implemented both Cross-Entropy loss,
which is standard for multi-class classification tasks like MNIST, and MSE.

== dataloader.rs
To feed data into the network, I built a zero-copy MNIST data loader that parses the original IDX binary format. Note,
the loader is intentionally naive, to work for arbitrary IDX data. The user (me) is expected to do necessary
transformations (like normalizing pixel brightness and creating one-hot vectors). The loader natively supports batching
and iteration.

= Preliminary Results
I evaluated the current implementation on the standard MNIST test set (10,000 images) after training for 10 epochs with
a learning rate of 0.1. The network achieved *92%* accuracy with a single hidden layer of 128 neurons and ReLU
activation. Below is a comparison against well-known baselines on the same dataset:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: horizon,
    [*Model*], [*Accuracy*], [*Notes*],
    [Linear Classifier (1-layer)], [88.0%], [No preprocessing],
    [My MLP (128)], [92%], [Current implementation],
    [K-NN (Euclidean)], [95.0%], [No preprocessing],
    [2-layer NN (300 HU)], [95.3%], [MSE loss, no preprocessing],
    [3-layer NN (500+300 HU)], [98.47%], [Softmax, cross-entropy, weight decay],
    [SVM (Gaussian Kernel)], [98.6%], [No preprocessing],
    [LeNet-5], [99.05%], [No distortions],
    [Committee of 35 CNNs], [99.77%], [Elastic distortions],
  ),
  caption: [MNIST classification accuracy across model architectures. Baseline numbers from @lecun-mnist-benchmarks.],
)

The current 92% pails compared to most other model benchmarks. This is reasonable since I'm using random, weight
initialization, no regularization, and a single hidden layer, as this was a quick end-to-end test. After spending some
time on optimization, I will revisit these benchmarks and do further analysis.

Training performance wise, std-ml is noticeably slower than optimized frameworks. A single epoch currently takes
several seconds due to naive triple-loop matrix multiplication without SIMD or cache-blocking optimizations. This
is a primary target for future work.

= Future Plan
Over the next few weeks, the focus will shift from "correctness" to "performance" and completeness:

1. *Matrix Optimization*: Study BLIS/FLAME to implement tiled matrix multiplication with optimized memory
access patterns for better cache utilization.
2. *Model Performance*: Better weight initialization, regularization, dropout, batch normalization, and other potential
  optimizations to improve model performance.
3. *Visualization*: Build a simple front-end to visualize weights/biases, error, accuracy, and more.

#pagebreak()
#bibliography("./refs.bib", title: "References")
