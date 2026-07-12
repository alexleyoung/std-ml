# std-ml

A simple neural network library implemented from-scratch using only Rust's standard library.

---

## Modules

### `matrix.rs`
Matrices are stored as flat `Vec<f64>` for cache locality, with stride-based indexing.
Implements transpose, addition, scaling, and multiplication, with some Rust operator traits (`+`, `-`, `*`) overloaded.
Matrix multiplication is implemented naively (triple-loop).

### `layer.rs`
Defines the `Layer` trait and a `Linear` implementation. The trait exposes `forward`, `backward`, `update`, and
`zero_grad` — both single-sample and batched variants. 

`Linear` supports four weight initialization strategies:
| Initialization | Distribution | Best For |
|---|---|---|
| `Random` | `U(-0.1, 0.1)` | Quick testing |
| `LeCun` | `U(±√(3/fan_in))` | Sigmoid, tanh |
| `Glorot` | `U(±√(6/(fan_in+fan_out)))` | Sigmoid, tanh |
| `He` | `N(0, √(2/fan_in))` | ReLU networks |

L2 regularization is applied per-weight during `update`.

### `activation.rs`
Non-linear activations, all implementing `Layer` for composability. Available activations: `ReLU`, `LeakyReLU`,
`Sigmoid`, `Tanh`.

### `network.rs`
Composes layers into a `Network` via `Vec<Box<dyn Layer>>`, supporting heterogeneous layer types. Exposes `forward`,
`forward_batch`, `backward_batch`, `update`, and `zero_grad`.

### `loss.rs`
Provides the `Loss` trait with both scalar and batch variants of `loss` and `gradient`. Implemented losses:
`CrossEntropy` (numerically stable, with softmax fused into the gradient) and `MSE`.

### `loader.rs`
An IDX binary format parser supporting both MNIST datasets. It reads IDX values into a `Vec<f64>`, handles
all IDX data types (u8, i8, i16, i32, f32, f64), and supports batching and shuffled iteration.

### `utils.rs`
Assorted utilities including LCG pseudo-random number generator with uniform (`fill`) and normal (`fill_norm`, via 
Box-Muller transform) sampling. Also provides `add_vecs`, `add_vecs_inplace`, and `outer_prod`.

---

## Usage

Requires [just](https://github.com/casey/just).

```sh
just run                             # defaults
just run dataset=fashion             # Fashion MNIST
just run epochs=10 lambda=0          # no regularization
```

Or directly:

```sh
cargo run --release -- \
  --dataset <mnist|fashion> \
  --lr <float> \
  --epochs <int> \
  --lambda <float> \
  --batch-size <int>
```

**Defaults:** `dataset=mnist`, `lr=0.2`, `epochs=5`, `lambda=0.0001`, `batch-size=32`

---

## Results

Evaluated on the standard MNIST test set (10,000 images):

| Model | Accuracy | Notes |
|---|---|---|
| Linear Classifier | 88.0% | No preprocessing |
| K-NN (Euclidean) | 95.0% | No preprocessing |
| 2-layer NN (300 HU) | 95.3% | MSE loss |
| **std-ml 2-layer NN (128 HU)** | **97.27%** | Cross-entropy, He init, L2 reg, ReLU |
| 3-layer NN (500+300 HU) | 98.47% | Softmax, cross-entropy, weight decay |
| LeNet-5 | 99.05% | No distortions |

*Baseline results from [LeCun, MNIST Benchmark Results](https://web.archive.org/web/20220505031207/https://yann.lecun.com/exdb/mnist/).*

Fashion MNIST peaks around ~85% (1 hidden-layer, 128 HU).

---

## Future work

- Matrix multiplication optimization: SIMD, cache-blocking
