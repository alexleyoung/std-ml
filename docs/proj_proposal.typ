#set page(paper: "us-letter", margin: 1in)
#set par(justify: true)

// Title Section
#align(center)[
  #text(size: 18pt, weight: "bold")[Project Proposal: std-ml] \
  #text(
    size: 14pt,
    style: "italic",
  )[A Standard-Library-Only Neural Network Framework in Rust]
]

#v(1em)

// Team Members
== 2 Team Members
- *Alex Young*

#v(1em)

== 3 Project Details

=== 3.1 Project Objective
- *Objective:* Design and implement a functional machine learning library from scratch using only the Rust standard library (`std`), avoiding all external linear algebra or ML crates
- *Problem to Solve:* Cultivate a deep understanding of machine learning fundamentals by forcing the manual implementation of matrix calculus, and memory-efficient data structures, and ML algorithms
- *Importance:* Understanding the underlying mechanics of machine learning algorithms and their implementations is crucial for leveraging the full potential of machine learning
- *Role of ML:* Being an ever-important subject given the rapid development of large language models, its more important than ever to have a deep understanding of how the technology works to be able to leverage it effectively

=== 3.2 Datasets
- *Data Source:* The *MNIST Database of Handwritten Digits*, obtained from the Yann LeCun repository
- *Collection:* Data consists of grayscale images ($28 times 28$ pixels) that have been size-normalized and centered
- *Features & Labels:* Features are the 784 raw pixel intensities (0–255); labels are integers 0–9
- *Data Split:* 50,000 examples for training, 10,000 for validation, and 10,000 for testing

=== 3.3 Machine Learning Algorithm
I will implement a *Multi-Layer Perceptron (MLP)*
- *Justification:* The MLP is the foundational architecture of deep learning, extending linear models into a "deep"
structure. Implementing this in Rust provides a rigorous challenge in memory management (ownership/borrowing) and manual
backpropagation math, making it an ideal scope for a semester project.

=== 3.4 Expected Outcomes
- *Library Engine:* A custom Rust library performing matrix operations and backpropagation using only `std`
- *Performance:* A trained model achieving $>95%$ accuracy on MNIST
- *Analysis:* Documenting how hyperparameters (learning rate, layer size) affect bias and variance, benchmarking training time and memory usage
