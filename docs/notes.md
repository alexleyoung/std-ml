Back propagation:

dL (gradient of loss) w.r.t y (dL/dy) is passed backwards from final (outout) layer

use this to calculate dL/dW, dL/dB, dL/dx

its all chain rule... dL/dx = dL/dy * dy/dx
                               "g"
                d Loss w.r.t inout
                                deriv. of activation

activation functions:
- need these to increase model complexity (otherwise, all layers collapse to a linear combination, like a linear
regression)
- these sit in between layers, non-linearly transforming raw layer input to feed forward
- back propagation works the same; recieve the gradient w.r.t its output (dL/dy), use this to find dL/dx, to continue
passing backward. see above for finding dL/dx

matrix optimization:
- BLAS: Basic Linear Algebra Subprograms
    - lin alg so prolific we created an API for it 50 years ago
- always operate (in this case) row-major to minimize cache missess
- use SIMD for fast mul/add
- Tiling/cache-blocking; partition matrix into B x B sub-matricies which fit into L1/L2 cache

