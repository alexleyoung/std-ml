---
name: review
description: Teaching-focused code review for MLP library checkpoints. Reviews architecture, Rust quality, and ML correctness with Socratic guidance.
license: MIT
compatibility: opencode
---

## Purpose
Provide educational code reviews at checkpoint completions. The goal is learning, not perfection.

## Review Process

### 1. Run Tests First
```
cargo test
```
Report any failures. Failing tests take priority over style feedback.

### 2. Architecture Review
Examine module organization and ML library design:
- Does the code structure match the intended phase in `docs/progress.md`?
- Are responsibilities cleanly separated (matrix ops vs layers vs training)?
- Is the public API intuitive for an ML library?
- Ask: "How would a user compose layers into a network?"

### 3. Rust Quality Review
Check idioms and best practices:
- Ownership/borrowing: Are unnecessary clones present? Could references work?
- Error handling: Using `Result`/`Option` appropriately vs panics?
- Trait implementations: `Debug`, `Add`, `Mul` where appropriate?
- Iterator patterns: Could explicit loops use iterators instead?
- **Std lib only**: Verify no external crates in `Cargo.toml`

Ask: "What does this ownership pattern tell you about who owns this data?"

### 4. ML Correctness Review
Verify mathematical correctness:
- **Dimensions**: Do matrix operations produce expected shapes?
- **Formulas**: Is backprop using the correct gradient formulas?
- **Numerical stability**: Is softmax subtracting max? Is log protected from zero?
- **Gradient checking**: Has numerical gradient verification been done?

Ask: "Can you walk through the dimensions at each step of this forward pass?"

## Teaching Approach

Instead of stating issues directly, ask questions:
- "What happens if you call this with mismatched dimensions?"
- "Why might we want to store the input during forward pass?"
- "How could you verify this gradient is correct?"

Point to relevant sections in `AGENTS.md` or `docs/progress.md` for deeper learning.

## Output Format

1. **Test Results**: Summary of `cargo test`
2. **Strengths**: What's working well
3. **Questions to Consider**: Socratic prompts for improvement areas
4. **Resources**: References to docs/ for concepts to review

## Persistence

After completing the review, **append** to `docs/reviews.md`:

```markdown
## Review: YYYY-MM-DD - Phase X: [Phase Name]

### Test Results
[PASS/FAIL summary]

### Strengths
- ...

### Questions to Consider
- ...

### Resources
- ...

### Next Steps
- ...
```

Create `docs/reviews.md` if it doesn't exist. Each review gets a timestamped section.
