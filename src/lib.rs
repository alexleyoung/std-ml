//! A from-scratch MLP library for MNIST digit classification
//!
//! This library implements a multilayer perceptron using only Rust's standard
//! library.
//!
//! # Structure
//!
//! - `matrix`: Linear algebra operations
//! - `layer`: Neural network layers
//! - `network`: Network composition
//! - `loss`: Loss functions
//! - `data`: Data loading

// Module declarations
pub mod activation;
pub mod layer;
pub mod loader;
pub mod loss;
pub mod matrix;
pub mod network;
pub mod utils;

pub use layer::Layer;
pub use matrix::Matrix;
