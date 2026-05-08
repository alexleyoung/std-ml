#set page(paper: "us-letter", margin: 0.8in)
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Final Report: std-ml] \
  #text(
    size: 14pt,
    style: "italic",
  )[A Standard-Library-Only Neural Network Framework in Rust] \
  #v(0.5em)
  #text(size: 11pt)[Alex Young] \
  #text(size: 10pt)[#link(
    "https://github.com/alexleyoung/std-ml",
  )[`github.com/alexleyoung/std-ml`]]
]

#v(1em)

#text(weight: "bold")[Abstract] \
This project is a ground-up implementation of a Multilayer Perceptron (MLP) neural network library written in Rust, using
only the standard library. The motivation is to gain a deep practical understanding of the mechanics that frameworks like
PyTorch and TensorFlow normally abstract away: matrix operations, backpropagation, gradient descent, weight
initialization, regularization, and batched training. I have developed a modular library composed of custom matrix primitives,
layers, activation functions, loss functions, and a zero-copy IDX data loader, training and testing on the MNIST
handwritten digits dataset. Since the midterm report, I extended the framework with mini-batch training, multiple
weight initialization strategies (Random, LeCun, Glorot, He), L2 regularization, additional activations (LeakyReLU,
Tanh), shuffled iteration in the data loader, and a transposed inner-loop matrix multiplication for better cache
behavior. These additions raised MNIST test accuracy from *92%* at the midterm to *97.27%* in the final implementation,
and I also added Fashion MNIST achieving \~85% prediction accuracy. This report details the final architecture, the
optimizations applied since the midterm, and a discussion of remaining performance bottlenecks.

#v(1.5em)

= Introduction
The current tech landscape makes it easier than ever to ignore the decades of math and computer science fundamentals that
have led to technology like modern LLMs. To better understand machine learning fundamentals, I built an MLP from
scratch (standard library only) in Rust.

The primary machine learning task is multi-class classification using the *MNIST dataset*, which consists of 70,000
grayscale images of handwritten digits #cite(<lecun-mnist>). The library supports fully connected feed-forward
networks of arbitrary depth, with composable activations, multiple weight initialization schemes, batched training,
and L2 regularization. By avoiding external ML libraries, I was able to constantly challenge myself with API design
decisions and performance implications like random bias initialization, matrix multiplication, and zero-copy
dataloader iteration.

Since the midterm, the focus shifted from getting a basic proof of concept to increasing model performance. I added
the optimizations that the midterm report flagged as future work: weight initialization strategies for different
activation functions, L2 regularization, and mini-batch training. The library also gained a second dataset
(Fashion MNIST) for benchmark variety, a few more activation functions, and a small matmul tweak for cache locality.

= Related Work
There is no shortage of (significantly better) general-purpose machine learning libraries. PyTorch #cite(<pytorch>)
remains my main reference for API design. Whenever I was faced with a design decision that was not immediately obvious
to me, I looked to PyTorch for reference. For weight initialization, I followed the LeCun, Glorot
(Xavier), and He papers, which derive variance bounds tailored to the activation function being used. For
matrix multiplication, I continued reading BLAS/BLIS/FLAME #cite(<blis>) #cite(<flame-gemm>), which explain numerous
modern GEMM optimizations like tiling and parallelization. I have not yet implemented the
full BLIS-style packing scheme due to time constraints, but I was able to implement a small optimization regarding
access of operands in row-major order during the inner matrix multiplication loop, which is discussed later.

= Methods
My implementation, `std-ml`#footnote[Source available at #link("https://github.com/alexleyoung/std-ml")[`github.com/alexleyoung/std-ml`].],
is a Rust std-lib only neural network library designed with modularity, supporting multilayer perceptron models with
mini-batch training, configurable weight initialization, and L2 regularization. Each module is placed in its own
aptly-named file in the `src/` directory.

== matrix.rs
At the very core of machine learning is matrices. My implementation uses a flat `Vec<f64>` to prevent unnecessary
memory fetching and ensure cache locality, with strides for indexing. Basic arithmetic is implemented
with variations depending on ownership: `transpose`, addition, scaling, and multiplication, plus in-place variants
(`add_inplace`, `sub_scale_inplace`) used in the hot paths of training to avoid allocating a fresh matrix per step.
Rust's operator traits (`Add`, `Sub`, `Mul`, `AddAssign`) are implemented to enable natural mathematical syntax
with the custom matrices.

Matrix multiplication is still implemented as a triple loop, but with one notable change since the midterm: the
right-hand operand is transposed up front so that the innermost loop walks both operands in row-major order.
Because the matrix is stored row-major, this turns what would have been a column-stride access pattern (one cache
miss roughly every element, for large matrices) into a sequential read, which is much more friendly to the CPU's
prefetcher. The cost is a single $O(n^2)$ transpose pass that allocates one extra matrix; the benefit is that the
inner $O(n^3)$ loop runs noticeably faster for the matrix sizes used in MNIST training. This is a long way from a
real BLIS-style kernel, but it is an optimization nonetheless.

== layer.rs
To populate networks, we need layers. The module defines the `Layer` trait and a `Linear` implementation.

==== Layer
The `Layer` trait represents any single layer in a network. It defines six methods, divided between single-sample
and batched variants:

`forward(&[f64]) -> Vec<f64>` and `forward_batch(Matrix) -> Matrix`

`backward(&[f64]) -> Vec<f64>` and `backward_batch(Matrix) -> Matrix`

`update(f64, f64)`

`zero_grad()`

The single-sample variants take and return vectors of shape `in_features` / `out_features`. The batched variants
take and return matrices of shape `(batch_size, in_features)` / `(batch_size, out_features)` so that an entire
mini-batch can flow through the network in one call. `update(learning_rate, lambda)` takes the learning rate and
the L2 regularization coefficient, and applies accumulated gradients to its weights and biases. `zero_grad()`
resets accumulated gradients between batches.

==== Linear
The `Linear` layer implements a linear layer, scaling and adding weights and biases. The forward pass caches its input
(either the per-sample vector or the per-batch matrix) for use in the backward pass. The backward pass computes
$(partial L) / (partial W)$ and $(partial L) / (partial b)$ via the outer product of the incoming gradient with the cached
input, accumulates them into the layer's gradient buffers, and returns $(partial L) / (partial x)$ for the previous
layer.

A significant addition since the midterm is the `Initialization` enum, which exposes four weight initialization
strategies #cite(<wiki-weight-init>). Initialization matters more than I initially thought, so I sought for better
implementations.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 7pt,
    align: horizon,
    [*Strategy*], [*Distribution*], [*Best for*],
    [`Random`], [$U(-0.1, 0.1)$], [Naive, initial implementation],
    [`LeCun`], [$U(plus.minus sqrt(3 / "fan_in"))$], [Sigmoid, tanh],
    [`Glorot`],
    [$U(plus.minus sqrt(6 / ("fan_in" + "fan_out")))$],
    [Sigmoid, tanh],

    [`He`], [$N(0, sqrt(2 / "fan_in"))$], [ReLU networks],
  ),
  caption: [Weight initialization strategies in `std-ml`.],
)

L2 regularization is also applied per-weight during `update`, with the regularization strength controlled via the
`lambda` argument: each weight $w$ is updated as $w arrow.l w - 2 lambda w$ in addition to the gradient step. This
discourages weights from growing too large and increases model performance.

== activation.rs
(Non-linear) activations are necessary to increase model complexity and enable learning. The `activation` module
contains implementations for `ReLU`, `LeakyReLU`, `Sigmoid`, and `Tanh`. All activations implement `Layer` for
composability. Note that only `forward` / `forward_batch` (activation) and `backward` / `backward_batch`
(elementwise gradient) are implemented; `update` and `zero_grad` are no-ops since activations have no learnable
parameters.

`LeakyReLU` and `Tanh` were added since the midterm. Both were added for model testing and tuning, as well as
completeness among the new weight initialization algorithms.

== network.rs
With layers and activations defined, we need something to compose them together. The `Network` struct stores a
`Vec<Box<dyn Layer>>` to support heterogeneous layer types. It exposes `forward`, `forward_batch`, `backward`,
`backward_batch`, `update`, and `zero_grad`, which simply iterate through its layers in order (forward) or reverse
(backward). This does incur potentially unnecessary overhead from heap allocation and dynamic dispatch on `Layer`,
but in practice the cost is negligible compared to the matmul inside each layer. An alternative design would be to create
generic networks leveraging Rust generics.

== loss.rs
The loss module provides both the loss value and its gradient w.r.t. predictions in scalar and batched forms.
I implemented both Cross-Entropy loss (with softmax fused into the gradient for numerical stability
#cite(<gfg-softmax-ce>)) and MSE. In batched cross-entropy, the softmax is computed row-wise per sample with a
max-subtraction trick to prevent overflow, and the gradient is the standard $hat(y) - y$ form.

== loader.rs
To feed data into the network, I built a zero-copy MNIST data loader that parses the original IDX binary format.
The loader is intentionally generic over IDX data types (u8, i8, i16, i32, f32, f64) and supports any IDX file out of
the box. The user is expected to do necessary transformations (like normalizing pixel
brightness by dividing by 255) in their training loop. The loader natively supports batching and, since the
midterm, *shuffled iteration*: each call to `iter()` produces a freshly shuffled index permutation, which prevents
the model from overfitting to the natural order of the dataset and is a standard practice in mini-batch SGD.

== utils.rs
Assorted utilities, including an LCG-based pseudo-random number generator with both uniform (`fill`) and normal
(`fill_norm`, via the Box-Muller transform) sampling. The normal sampler is what powers He initialization. The
module also contains small vector helpers (`add_vecs`, `add_vecs_inplace`, `outer_prod`) used throughout the library.

== main.rs
The training entry point exposes a small CLI with `--dataset`, `--lr`, `--epochs`, `--lambda`, and `--batch-size`,
making it easy to test hyperparameters from the command line. The default architecture is a 2-layer MLP
($784 arrow.r 128 arrow.r 10$) with ReLU activation, He initialization, cross-entropy loss, L2 regularization, and
mini-batch SGD with batch size 32. After each epoch, accuracy is computed on the held-out test set and printed
alongside training loss and per-epoch wall-clock time.

= Experimental Results
I evaluated the final implementation on the standard MNIST test set (10,000 images) after training for 5 epochs
with a learning rate of 0.2, batch size 32, $lambda = 10^(-4)$, He initialization, and a single hidden layer of
128 ReLU neurons. The network achieved *97.27%* test accuracy, a substantial improvement over the *92%* reported
at the midterm. The same code, run on Fashion MNIST with the same hyperparameters, peaks around *85%*.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: horizon,
    [*Model*], [*Accuracy*], [*Notes*],
    [Linear Classifier (1-layer)], [88.0%], [No preprocessing],
    [My MLP, midterm (128 HU)],
    [92.0%],
    [Random init, no regularization, no batching],

    [K-NN (Euclidean)], [95.0%], [No preprocessing],
    [2-layer NN (300 HU)], [95.3%], [MSE loss, no preprocessing],
    [*My MLP, final (128 HU)*],
    [*97.27%*],
    [*He init, L2 reg, mini-batch SGD, ReLU*],

    [3-layer NN (500+300 HU)], [98.47%], [Softmax, cross-entropy, weight decay],
    [SVM (Gaussian Kernel)], [98.6%], [No preprocessing],
    [LeNet-5], [99.05%], [No distortions],
    [Committee of 35 CNNs], [99.77%], [Elastic distortions],
  ),
  caption: [MNIST classification accuracy across model architectures. Baseline numbers from
    @lecun-mnist-benchmarks.],
)

== Optimization Recap
Three changes accounted for nearly all of the gain from 92% to 97.27%:

1. *Weight initialization (He).* Switching from $U(-0.1, 0.1)$ to He $N(0, sqrt(2 / "fan_in"))$ alone closed
  most of the gap. With ReLU, the previous uniform scheme produced activations whose variance shrank rapidly with
  depth, leaving the network under-exercised at initialization. He initialization preserves activation variance
  across layers under ReLU, which made the loss drop noticeably faster in the first epoch.
2. *Mini-batch SGD.* The midterm version updated weights once per sample. Switching to batches of 32 with
  gradient accumulation gave a smoother loss curve, allowed a higher effective learning rate, and made each epoch
  somewhat faster (per-step overhead is amortized over 32 samples).
3. *L2 regularization.* Adding $lambda = 10^(-4)$ weight decay reduced the gap between training and test accuracy
  by a small but consistent amount. The effect is modest at 5 epochs but more visible when training longer.

The other changes (LeakyReLU, Tanh, Glorot/LeCun init) did not directly contribute to the model performance but made
the framework more general and were added for completeness.

== Performance
Training performance is still the weakest part of the framework. A single epoch on MNIST with batch size 32
takes \~10 seconds on my laptop, which is dominated by matrix multiplication in the linear
layers. The transposed inner-loop change in `matrix.rs::mul` measurably helped, but the multiplication is still a naive
triple loop with no SIMD intrinsics, no cache-blocking, and no parallelism. Profiling consistently shows >80% of
training time inside `Matrix::mul`. This is still the primary target for further work.

= Conclusion
`std-ml` is now a small but reasonably capable MLP framework depending only on the Rust standard library.
It trains a 2-layer ReLU MLP to *97.27%* on MNIST and \~*85%* on Fashion MNIST, a significant improvement from the *92%* baseline
reported at the midterm, and the architecture is general enough to support different layer compositions,
activation functions, weight initialization strategies, and L2 penalties.

The largest remaining limitation is matmul throughput. Here are some next steps I encountered in research:

- *Cache-blocked matmul.* Tile the multiplication into $B times B$ sub-blocks sized to fit in L1, following the
  BLIS / FLAME GEMM approach
- *SIMD via `std::simd`.* Parallelization of matrix multiplications to vastly increase throughput
- *Convolutional layers.* A `Conv2d` layer
- *Visualization.* A small front-end that plots loss/accuracy curves and visualizes first-layer weights

I learned a lot making this project, producing \~1,000 lines of (questionable) Rust. I am most happy with the exploration
I was able to do into model performance optimization, as well as training optimization as I am interested in systems and
high-performance programming.

#v(0.5em)

Code available at #link("https://github.com/alexleyoung/std-ml")[`github.com/alexleyoung/std-ml`].

#pagebreak()
#bibliography("./refs.bib", title: "References")
