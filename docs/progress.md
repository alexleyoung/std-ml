# Implementation Progress Plan

This document outlines the phased implementation of the MLP library. Each phase builds on the previous one, with clear learning objectives and checkpoints.

## Phase 0: Setup and Foundation (1-2 days)

### Objectives
- Set up Rust project structure
- Understand Cargo and dependencies
- Create basic directory structure

### Tasks
1. Verify Cargo project exists (already initialized)
2. Confirm directory structure:
   ```
   std-ml/
   ├── Cargo.toml          # No external dependencies
   ├── src/
   │   ├── lib.rs          # Main library file
   │   └── main.rs         # Main binary (if needed)
   ├── docs/
   │   ├── architecture.md # System design
   │   └── progress.md     # Implementation phases
   └── data/               # MNIST data
   ```
3. **Std lib only**: Cargo.toml should have NO dependencies in `[dependencies]` section
4. Run `cargo build` to verify setup

### Rust Learning Points
- Cargo project structure
- `src/lib.rs` vs `src/main.rs`
- Basic compilation and running

### Success Criteria
- `cargo build` succeeds
- Can run `cargo run` or `cargo test`
- Basic file structure is correct

### Testing Setup
Add a `tests/` directory for integration tests:
```
std-ml/
├── src/
│   ├── lib.rs          # Library root + test utilities
│   ├── matrix.rs       # Matrix module with tests
│   └── main.rs         # Binary (if needed)
├── tests/
│   ├── integration_tests.rs  # Cross-module tests
│   └── helpers.rs      # Test helpers
├── docs/
│   ├── architecture.md
│   └── progress.md
└── Cargo.toml
```

**Rust Testing Quick Reference**:

**Running Tests**:
```bash
cargo test                    # Run all tests
cargo test --lib              # Run unit tests only
cargo test --test integration_tests  # Run integration tests only
cargo test matrix_multiply    # Run specific test by name
cargo test -- --nocapture     # Show print statements
cargo test -- --test-threads=1  # Run tests sequentially
```


## Phase 1: Matrix/Vector Operations (1-2 weeks)

### Objectives
- Implement fundamental linear algebra operations using ONLY std library
- Understand Rust ownership in mathematical operations
- Build the foundation for all ML operations

### Important Constraint
**Std lib only**: This is a pure Rust implementation with no external crates. All operations must be written from scratch using:
- `Vec<T>` for storage and operations
- Basic indexing and bounds checking
- Standard Rust traits (Add, Mul, Debug, etc.)
- No matrix libraries, no BLAS, no LAPACK, no external dependencies

### Tasks

#### 1.1 Matrix Struct
```rust
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,  // row-major storage (std lib only)
}
```

**Note**: This is a from-scratch implementation using only Rust's standard library. No external crates allowed!

**Why from scratch?** Building matrix operations by hand teaches you:
- Memory layout and indexing
- Bounds checking and safety
- Algorithm optimization with loops
- The mathematical foundations of linear algebra

**What you'll implement manually**:
- Matrix multiplication (triple-nested loops)
- Transposition (index swapping)
- Element-wise operations (iterator patterns)
- Scalar multiplication

#### 1.2 Basic Operations
- **Construction**: `Matrix::new(rows, cols, data)` with validation
- **Indexing**: `matrix.get(row, col)` and `matrix.set(row, col, value)`
- **Shape**: `matrix.rows()` and `matrix.cols()`
- **Debug**: Implement `Debug` trait for easy printing

**Unit Tests to Write**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let m = Matrix::new(2, 3, vec![1.0; 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_new_wrong_size() {
        let _ = Matrix::new(2, 3, vec![1.0; 5]); // Should panic
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::new(2, 2, vec![0.0; 4]);
        m.set(1, 1, 42.0);
        assert_eq!(m.get(1, 1), 42.0);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let m = Matrix::new(2, 2, vec![0.0; 4]);
        let _ = m.get(2, 0); // Row 2 doesn't exist
    }
}
```

#### 1.3 Matrix Operations
- **Addition**: `a + b` (element-wise, or `&a + &b` to borrow)
- **Multiplication**: `a.dot(&b)` (matrix multiplication)
- **Scaling**: `matrix * scalar`
- **Transpose**: `matrix.transpose()`

**Unit Tests to Write**:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_matrix_addition() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = &a + &b;

        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_matrix_multiply() {
        // 2x3 * 3x2 = 2x2
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.dot(&b);

        // Manual verification:
        // Row 0: 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        //         1*8 + 2*10+ 3*12 = 8+20+36 = 64
        // Row 1: 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        //         4*8 + 5*10+ 6*12 = 32+50+72 = 154
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_multiply_dimension_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0; 4]);
        let b = Matrix::new(3, 2, vec![1.0; 6]);
        let _ = a.dot(&b); // cols (2) != rows (3)
    }

    #[test]
    fn test_scalar_multiply() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a * 2.0;

        assert_eq!(c.get(0, 0), 2.0);
        assert_eq!(c.get(1, 1), 8.0);
    }
}
```

#### 1.4 Vector Operations
- **Dot product**: `dot(&a, &b)`
- **Element-wise**: `add(&a, &b)`, `multiply(&a, &b)`

**Unit Tests to Write**:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_vector_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot(&a, &b);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vector_elementwise_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add(&a, &b);

        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    #[should_panic]
    fn test_vector_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let _ = dot(&a, &b);
    }
}
```

### Rust Learning Points
- **Ownership**: When to borrow vs. transfer
- **Traits**: Implementing `Add`, `Mul`, `Debug`
- **Error handling**: Result vs. panic for dimension mismatches
- **Testing**: Unit tests for each operation

### Math Learning Points
- Matrix multiplication dimensions: `(m,n) * (n,p) = (m,p)`
- Transpose: `Aᵀ[i,j] = A[j,i]`
- Dot product: `Σ a_i * b_i`

### Implementation Checklist
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_and_access() {
        // Create a matrix
        let m = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test dimensions
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);

        // Test element access
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(2, 1), 6.0);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_invalid_matrix_size() {
        // Wrong number of elements
        let _ = Matrix::new(2, 3, vec![1.0; 5]); // Should be 6 elements
    }

    #[test]
    fn test_matrix_multiply_2x2() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.dot(&b);

        // Manual calculation:
        // [1 2] * [5 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3 4]   [7 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_matrix_multiply_non_square() {
        // 2x3 * 3x2 = 2x2
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.dot(&b);

        // Row 0: 1*7 + 2*9 + 3*11 = 58, 1*8 + 2*10 + 3*12 = 64
        // Row 1: 4*7 + 5*9 + 6*11 = 139, 4*8 + 5*10 + 6*12 = 154
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_matrix_addition() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = &a + &b;

        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_matrix_scalar_multiply() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a * 2.5;

        assert_eq!(c.get(0, 0), 2.5);
        assert_eq!(c.get(0, 1), 5.0);
        assert_eq!(c.get(1, 0), 7.5);
        assert_eq!(c.get(1, 1), 10.0);
    }

    #[test]
    #[should_panic]
    fn test_multiply_dimension_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0; 4]);
        let b = Matrix::new(3, 2, vec![1.0; 6]);
        let _ = a.dot(&b); // cols (2) != rows (3)
    }

    #[test]
    #[should_panic]
    fn test_add_dimension_mismatch() {
        let a = Matrix::new(2, 3, vec![1.0; 6]);
        let b = Matrix::new(2, 2, vec![1.0; 4]);
        let _ = &a + &b;
    }

    // Test using test utilities from lib.rs
    #[test]
    fn test_with_utilities() {
        use std_ml::test_utils::approx_eq;

        // Test floating point comparison
        assert!(approx_eq(1.0 / 3.0, 0.333333333333, 1e-10));
        assert!(!approx_eq(1.0, 1.1, 1e-10));
    }

    // Property-based test
    #[test]
    fn test_transpose_property() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose().transpose();

        // Transposing twice should give original
        assert_eq!(m.rows(), t.rows());
        assert_eq!(m.cols(), t.cols());

        for i in 0..m.rows() {
            for j in 0..m.cols() {
                assert!((m.get(i, j) - t.get(i, j)).abs() < 1e-10);
            }
        }
    }

    // Test identity matrix property
    #[test]
    fn test_identity_property() {
        // Create identity matrix I
        let identity = Matrix::new(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);

        // Create test matrix A
        let a = Matrix::new(3, 2, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // I * A should equal A
        let result = identity.dot(&a);

        for i in 0..3 {
            for j in 0..2 {
                assert!((a.get(i, j) - result.get(i, j)).abs() < 1e-10);
            }
        }
    }

    // Test matrix multiplication is associative
    #[test]
    fn test_matrix_multiply_associative() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 4, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
        let c = Matrix::new(4, 5, vec![19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]);

        let left = a.dot(&b).dot(&c);  // (A * B) * C
        let right = a.dot(&(&b.dot(&c)));  // A * (B * C)

        // Compare element-wise
        for i in 0..2 {
            for j in 0..5 {
                assert!((left.get(i, j) - right.get(i, j)).abs() < 1e-9);
            }
        }
    }
}
```

### Verification Steps
1. **Numerical correctness**: Verify matrix multiplication by hand
2. **Dimension safety**: Ensure operations panic on mismatched dimensions
3. **Performance**: No unnecessary allocations

### Common Pitfalls
- Row-major vs column-major indexing confusion
- Off-by-one errors in indexing
- Forgetting to transpose when needed

---

## Phase 2: Linear Layer (Forward Pass) (1 week)

### Objectives
- Implement a single linear transformation layer
- Understand weight matrices and biases
- Handle batched inputs

### Tasks

### Rust Learning Points
- **Option<T>**: For optional state storage
- **Batch processing**: Working with different input shapes
- **Trait implementation**: Maybe implement a `Layer` trait

### Math Learning Points
- Linear transformation: `y = W·x + b`
- Matrix-vector multiplication
- Broadcasting biases

### Implementation Checklist
- [ ] Linear struct with weight and bias
- [ ] Forward pass for single example
- [ ] Forward pass for batch
- [ ] Input storage for backprop
- [ ] Initialization methods
- [ ] **Unit tests for forward pass**
- [ ] **Test single example vs batch equivalence**
- [ ] **Test with known input/output pairs**

**Unit Test Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_single() {
        // Create layer: 2 input features, 3 output features
        let mut layer = Linear::new(2, 3);

        // Set specific weights for testing
        layer.weight = Matrix::new(3, 2, vec![
            1.0, 2.0,   // output 0
            3.0, 4.0,   // output 1
            5.0, 6.0,   // output 2
        ]);
        layer.bias = vec![0.1, 0.2, 0.3];

        // Input: [1.0, 2.0]
        let input = vec![1.0, 2.0];
        let output = layer.forward(&input);

        // Manual calculation:
        // out[0] = 1*1 + 2*2 + 0.1 = 1 + 4 + 0.1 = 5.1
        // out[1] = 1*3 + 2*4 + 0.2 = 3 + 8 + 0.2 = 11.2
        // out[2] = 1*5 + 2*6 + 0.3 = 5 + 12 + 0.3 = 17.3
        assert!((output[0] - 5.1).abs() < 1e-10);
        assert!((output[1] - 11.2).abs() < 1e-10);
        assert!((output[2] - 17.3).abs() < 1e-10);

        // Verify input was stored for backprop
        assert!(layer.input.is_some());
        assert_eq!(layer.input.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_linear_forward_batch() {
        let mut layer = Linear::new(2, 2);

        // Set weights
        layer.weight = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        layer.bias = vec![0.0, 0.0];

        // Batch of 2 examples
        let inputs = vec![
            vec![1.0, 2.0],  // First example
            vec![3.0, 4.0],  // Second example
        ];

        let outputs = layer.forward_batch(&inputs);

        // Check dimensions
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 2);

        // Verify first example
        // out[0] = 1*1 + 2*2 = 5
        // out[1] = 1*3 + 2*4 = 11
        assert!((outputs[0][0] - 5.0).abs() < 1e-10);
        assert!((outputs[0][1] - 11.0).abs() < 1e-10);

        // Verify second example
        // out[0] = 3*1 + 4*2 = 11
        // out[1] = 3*3 + 4*4 = 25
        assert!((outputs[1][0] - 11.0).abs() < 1e-10);
        assert!((outputs[1][1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_vs_batch_equivalence() {
        // Forward pass should give same result for single example
        // whether we use forward() or forward_batch()
        let mut layer = Linear::new(3, 2);

        // Set reproducible weights
        layer.weight = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        layer.bias = vec![0.5, 0.6];

        let input = vec![1.0, 2.0, 3.0];

        // Single forward
        let single_out = layer.forward(&input);

        // Batch forward (with batch size 1)
        let batch_out = layer.forward_batch(&vec![input.clone()])[0].clone();

        // Should be identical
        for i in 0..2 {
            assert!((single_out[i] - batch_out[i]).abs() < 1e-15);
        }
    }

    #[test]
    #[should_panic]
    fn test_dimension_mismatch() {
        let layer = Linear::new(2, 3);
        let input = vec![1.0];  // Wrong size: should be 2
        let _ = layer.forward(&input);
    }
}
```

### Verification
- [ ] Compare with manual calculation for small case
- [ ] Verify dimensions match expected
- [ ] Test with known input/output pairs

### Common Pitfalls
- Forgetting to store input for backward pass
- Not handling batch dimension correctly
- Transpose confusion in matrix-vector multiplication
- **Std lib only**: Don't add external crates for initialization - use fixed values or manual implementation

---

## Phase 3: Backpropagation (1-2 weeks)

### Objectives
- Implement backward pass for linear layer
- Understand the chain rule
- Compute gradients for weights and biases

### Tasks

#### 3.1 Gradient Storage
```rust
pub struct Linear {
    // ... existing fields
    weight_grad: Matrix,
    bias_grad: Vec<f64>,
    input: Option<Vec<f64>>,  // needed for weight gradient
}
```

**Numerical Gradient Check** (for Phase 3):
```rust
#[test]
fn test_numerical_gradient() {
    // Create layer with specific small weights for reproducibility
    let mut layer = Linear::new(2, 2);
    // Manually set weights - no randomness needed
    layer.weight = Matrix::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]);

    // Compute analytic gradient
    let grad_output = vec![0.5, 0.5];
    layer.input = Some(vec![0.1, 0.2]);
    let _analytic_grad = layer.backward(&grad_output);

    // Compare with numerical approximation
    // ... (see architecture.md for full example)
}
```

#### 3.2 Backward Pass
**Chain rule**: `∂L/∂W = (∂L/∂output) · (∂output/∂W) = grad_output · inputᵀ`

**For weights**:
```
grad_output shape: (batch_size, out_features)
input shape: (batch_size, in_features)
grad_weight shape: (out_features, in_features)
grad_weight = grad_outputᵀ · input
```

**For biases**:
```
grad_bias = sum over batch: grad_output
```

**For input gradient** (to pass to previous layer):
```
grad_input = grad_output · weightᵀ
```

#### 3.3 Zero Gradients
#### 3.4 Update Step

### Rust Learning Points
- **Mutable references**: Updating weights in place
- **Loops and indexing**: For element-wise updates
- **Error handling**: Ensuring gradients are zeroed

### Math Learning Points
- **Chain rule**: The foundation of backpropagation
- **Matrix calculus**: Gradients are matrices/vectors
- **Dimension matching**: Crucial for correctness

### Implementation Checklist
- [ ] Gradient fields in Linear struct
- [ ] `backward()` method returns input gradient
- [ ] `zero_grad()` method
- [ ] `update()` method with learning rate
- [ ] **Numerical gradient checking (see 3.1)** - CRITICAL
- [ ] **Unit tests for backprop**
- [ ] **Test gradient accumulation (should reset to zero)**
- [ ] **Test weight updates actually change values**

### Verification Approach
**Numerical Gradient Checking** (critical for verifying backprop):
1. Set weights to small, specific values (no randomness)
2. Compute analytic gradient via backprop
3. Approximate gradient numerically: `∂f/∂x ≈ (f(x+ε) - f(x-ε)) / 2ε`
4. Compare within tolerance (e.g., 1e-5)

**Why this matters**: Backprop is tricky to get right. This verification catches bugs early.

### Common Pitfalls
- Forgetting to transpose when computing gradients
- Not handling batch dimension correctly
- Accumulating gradients without zeroing

### Verification
- **Critical**: Run numerical gradient check for every layer
- Compare analytic vs numerical gradients (should match within tolerance)
- Test on random inputs with small weights

---

## Phase 4: Activation Functions (1 week)

### Objectives
- Implement non-linear activation functions
- Understand when to use each activation
- Implement gradient computation

### Tasks

#### 4.1 Activation Trait
#### 4.2 Implementations

### Rust Learning Points
- **Trait objects**: Store activations as `Box<dyn Activation>`
- **Numerical stability**: Handling overflow/underflow
- **Method chaining**: Forward passes through multiple layers

### Math Learning Points
- **Activation functions**: Why we need non-linearity
- **Derivatives**: Computing gradients through non-linearities
- **Softmax**: Output probabilities for classification

### Implementation Checklist
- [ ] Activation trait
- [ ] ReLU forward and backward
- [ ] Sigmoid forward and backward
- [ ] Softmax forward (and backward for cross-entropy)
- [ ] **Numerical stability tests** (overflow, underflow)
- [ ] **Unit tests for each activation**
- [ ] **Test with extreme values (large positive, large negative)**

### Common Pitfalls
- Softmax overflow (not subtracting max)
- Sigmoid saturation (values close to 0 or 1)
- ReLU "dying neurons" (gradient always zero)

---

## Phase 5: Network Composition (1 week)

### Objectives
- Compose multiple layers into a network
- Orchestrate forward and backward passes
- Manage layers dynamically

### Tasks

#### 5.1 Network Struct
#### 5.2 Forward Pass
#### 5.3 Backward Pass
#### 5.4 Update
#### 5.5 Construction

### Rust Learning Points
- **Trait objects**: `Box<dyn Trait>` for heterogeneous collections
- **Ownership**: Who owns the layers? The network does.
- **Error propagation**: Handling layer failures
- **Option/Maybe types**: For optional layers

### Math Learning Points
- **Forward propagation**: Data flow through network
- **Backpropagation**: Gradient flow in reverse
- **Chain rule application**: Combining gradients across layers

### Implementation Checklist
- [ ] Network struct with layers
- [ ] `add_layer()` and `add_activation()` methods
- [ ] Forward pass through all layers
- [ ] Backward pass through all layers
- [ ] Update all layers
- [ ] **Unit tests for network operations**
- [ ] **Test network of single layer matches single layer**
- [ ] **Test dimension preservation through network**

### Common Pitfalls
- **Size mismatch**: Ensure layer outputs match next layer inputs
- **Gradient accumulation**: Forgot to zero gradients
- **Layer ordering**: Activation after linear or before?

### Verification
- [ ] Forward pass produces expected shape
- [ ] Backward pass doesn't panic
- [ ] Weights update correctly

---

## Phase 6: Loss Functions (1 week)

### Objectives
- Implement loss functions for different tasks
- Understand mathematical foundations
- Connect loss to backpropagation

### Tasks

#### 6.1 Loss Trait
#### 6.2 MSE Loss
#### 6.3 Cross-Entropy Loss

### Rust Learning Points
- **Generic functions**: Work with any loss type
- **Error handling**: Numerical stability in log
- **Trait bounds**: What can a loss function do?

### Math Learning Points
- **MSE**: For regression tasks
- **Cross-Entropy**: For classification with softmax
- **Simplification**: Gradient of softmax + cross-entropy is simple

### Implementation Checklist
- [ ] Loss trait
- [ ] MSE loss compute and gradient
- [ ] Cross-entropy loss compute and gradient
- [ ] **Numerical stability tests (log(0) prevention)**
- [ ] **Unit tests for each loss function**
- [ ] **Test loss gradient vs numerical gradient**
- [ ] **Test that loss decreases with better predictions**

### Common Pitfalls
- **Log(0)**: Add epsilon to predictions
- **Normalization**: Divide by batch size or not?
- **One-hot encoding**: Target should be one-hot for cross-entropy

---

## Phase 7: Data Loading and Preprocessing (1-2 weeks)

### Objectives
- Load MNIST data from CSV files
- Preprocess data (normalization, one-hot encoding)
- Create data loaders for batching

### Tasks

#### 7.1 Data Structures
#### 7.2 Loading MNIST
#### 7.3 Batch Iteration

### Rust Learning Points
- **File I/O**: Reading CSV files
- **Parsing**: String to number conversions
- **Error handling**: `Result` propagation with `?`
- **Iterators**: Creating batch iterators
- **No randomness**: With std lib only, consider deterministic shuffling or user-provided seed

### Math Learning Points
- **Normalization**: Why we divide by 255 or standardize
- **One-hot encoding**: Representing categorical data
- **Batch size**: Tradeoffs between noise and stability

### Implementation Checklist
- [ ] MnistExample struct
- [ ] CSV parsing function
- [ ] One-hot encoding
- [ ] Data normalization
- [ ] Batch iteration
- [ ] Shuffling
- [ ] Train/validation split
- [ ] **Unit tests for data loading**
- [ ] **Test normalization preserves range**
- [ ] **Test batch size handling**
- [ ] **Test train/val split doesn't leak data**

### Common Pitfalls
- **CSV parsing**: Handle empty lines, malformed data
- **Normalization**: Do it consistently (train/test)
- **Memory**: Loading all data at once vs streaming

---

## Phase 8: Training Loop (1-2 weeks)

### Objectives
- Implement full training loop
- Track training metrics
- Handle validation

### Tasks

#### 8.1 Training Function
#### 8.2 Validation
#### 8.3 Training History

### Rust Learning Points
- **Mutable vs immutable**: Network is mutable during training, immutable during validation
- **Error propagation**: Handling training failures
- **Data structures**: Designing history/tracking structs
- **Borrowing**: Multiple borrows of data loader

### Math Learning Points
- **Overfitting**: Training loss decreasing but validation loss increasing
- **Learning rate**: Too high = divergence, too low = slow convergence
- **Early stopping**: When to stop training

### Implementation Checklist
- [ ] Training function
- [ ] Validation function
- [ ] Metrics tracking (loss, accuracy)
- [ ] Training history struct
- [ ] Console logging of progress
- [ ] Early stopping (optional)
- [ ] **Unit tests for training loop**
- [ ] **Test that loss decreases on small dataset**
- [ ] **Test metrics computation**
- [ ] **Integration test: train on 10 examples, check progress**

### Common Pitfalls
- **Forgetting to zero gradients**: Accumulation bug
- **Not shuffling**: Model sees data in same order
- **Learning rate too high**: Loss explodes
- **Memory leaks**: Not resetting data loader properly

### Verification
- [ ] Loss decreases over epochs (on small subset)
- [ ] Can overfit small batch (loss → 0)
- [ ] Validation metrics make sense

---

## Phase 9: Optimization and Polish (1-2 weeks)

### Objectives
- Implement optimization algorithms
- Add regularization
- Improve training stability
- Create visualization/plotting

### Tasks

#### 9.1 Optimizer Abstraction
#### 9.2 Regularization
#### 9.3 Training Visualization

### Rust Learning Points
- **Polymorphism**: Different optimizers share interface
- **Memory management**: Storing optimizer state (momentum)
- **File I/O**: Exporting data for visualization

### Math Learning Points
- **Momentum**: Helps escape local minima
- **Weight decay**: Prevents overfitting
- **Adam**: Advanced optimizer (stretch goal)

### Implementation Checklist
- [ ] Optimizer trait
- [ ] SGD implementation
- [ ] Momentum implementation
- [ ] L2 regularization
- [ ] Training visualization
- [ ] Hyperparameter tuning guide

### Common Pitfalls
- **Forgetting velocity initialization**: Check if empty before update
- **L2 regularization order**: Add to gradient before or after clipping?
- **Numerical issues**: Large momentum causing instability

---

## Phase 10: Evaluation and Testing (1 week)

### Objectives
- Evaluate model on test set
- Create confusion matrix
- Calculate precision, recall, F1
- Generate reports

### Tasks

#### 10.1 Evaluation Metrics
#### 10.2 Confusion Matrix

### Rust Learning Points
- **Complex data structures**: Confusion matrix as nested Vec
- **Floating point comparisons**: Use partial_cmp carefully
- **Error handling**: Handle division by zero in metrics

### Math Learning Points
- **Confusion matrix**: True positives, false positives, etc.
- **Precision/Recall**: Tradeoffs in classification
- **F1 score**: Harmonic mean of precision and recall

### Implementation Checklist
- [ ] Prediction method (forward without storing state)
- [ ] Confusion matrix computation
- [ ] Precision, recall, F1 calculations
- [ ] Accuracy calculation
- [ ] Report generation
- [ ] Visualization (text-based)

### Common Pitfalls
- **Division by zero**: Handle empty classes
- **One-hot vs class index**: Ensure consistency
- **Test set leakage**: Only evaluate on held-out data

---

## Phase 11: Documentation and Examples (1 week)

### Objectives
- Document all code
- Create usage examples
- Write README
- Add doc tests

### Tasks

#### 11.1 Documentation
#### 11.2 Examples
- `examples/train_mnist.rs`: Full training script
- `examples/predict.rs`: Load and predict
- `examples/visualize.rs`: Show weights/activations

#### 11.3 README
- Project description
- Installation
- Usage examples
- Architecture overview

### Rust Learning Points
- **Doc comments**: `///` and `/// #`
- **Doc tests**: Code examples that run as tests
- **Module documentation**: `//!` comments at top of files

### Implementation Checklist
- [ ] Module-level documentation
- [ ] Function-level documentation
- [ ] Doc tests for key functions
- [ ] Usage examples
- [ ] README
- [ ] Architecture diagram

---

## Stretch Goals (After Phase 11)

### 12.1 Advanced Optimizers
- Adam
- RMSprop
- Adagrad

### 12.2 Batch Normalization
- Normalize layer inputs
- Reduce internal covariate shift

### 12.3 Dropout
- Randomly zero neurons during training
- Regularization technique

### 12.4 Convolutional Layers
- For image data (though MNIST works with MLP)
- Weight sharing

### 12.5 Early Stopping
- Monitor validation loss
- Stop when not improving

### 12.6 Hyperparameter Search
- Grid search or random search
- Learning rate, hidden sizes, etc.

### 12.7 Visualization
- Export metrics to CSV for plotting externally
- Text-based visualizations in terminal
- Print weight statistics

---

## Weekly Schedule Suggestion

| Week | Phase | Focus |
|------|-------|-------|
| 1-2 | 0-1 | Setup, Matrix operations |
| 3 | 2 | Linear layer forward pass |
| 4-5 | 3 | Backpropagation |
| 6 | 4 | Activation functions |
| 7 | 5 | Network composition |
| 8 | 6 | Loss functions |
| 9-10 | 7 | Data loading |
| 11-12 | 8 | Training loop |
| 13 | 9 | Optimization, polish |
| 14 | 10 | Evaluation |
| 15 | 11 | Documentation |
| 16+ | 12+ | Stretch goals |

## Key Learning Milestones

### Milestone 1: Backpropagation Works
- Numerical gradient check passes
- You can compute gradients correctly

### Milestone 2: Overfit Small Dataset
- Train on 10-100 examples
- Achieve near-zero training loss
- Proves network can learn

### Milestone 3: MNIST > 90% Accuracy
- Train on full MNIST
- Achieve reasonable accuracy
- Shows practical utility

### Milestone 4: Clean Architecture
- Code is readable and well-documented
- Tests pass
- Can explain any part of the code

### Milestone 5: Std Lib Mastery
- All operations written from scratch
- No external dependencies in Cargo.toml
- Deep understanding of memory layout and algorithms
- Comfortable implementing linear algebra manually

## Tracking Progress

Keep a progress log in `docs/progress.md` with:
- Date and phase completed
- Key learnings
- Challenges encountered
- Next steps

## Questions to Keep in Mind

1. **Rust**: "Is this code idiomatic Rust?"
2. **ML**: "Do I understand the math behind this?"
3. **Both**: "Can I explain this to someone else?"

Remember: **Learning is the goal, not perfection.** It's okay to make mistakes and refactor as you learn.
