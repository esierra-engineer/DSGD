# Technical Documentation: DSClassifierMultiQ Class

## Overview

The `DSClassifierMultiQ` class is a tabular interpretable classifier based on **Dempster-Shafer Theory** (DST) and **Gradient Descent** optimization. It implements a multi-class classification approach using belief functions and combines evidence from multiple rules using the Dempster-Shafer combination rule.

### Key Characteristics
- **Framework**: PyTorch-based neural network implementation
- **Theory**: Dempster-Shafer Theory for uncertain reasoning
- **Optimization**: Gradient descent with Adam or SGD optimizers
- **Multi-class Support**: Native support for any number of classes
- **Interpretability**: Rule-based system with explainable predictions
- **Hardware Acceleration**: Supports CPU, CUDA (GPU), and MPS (Apple Silicon)

---

## Class Architecture

### Inheritance
```python
DSClassifierMultiQ(ClassifierMixin)
```
- Inherits from `sklearn.base.ClassifierMixin` for scikit-learn compatibility
- Wraps a `DSModelMultiQ` instance (accessible via `.model` attribute)

### Key Components

#### 1. DSClassifierMultiQ (Classifier Layer)
- Manages training loop, optimization, and prediction
- Handles data preprocessing and batching
- Implements debug modes for performance monitoring

#### 2. DSModelMultiQ (Model Layer)
- PyTorch `nn.Module` implementation
- Manages rules and their associated mass functions
- Performs forward pass using Dempster-Shafer combination
- Implements commonality transformation for efficiency

#### 3. DSRule (Rule Definition)
- Wrapper for lambda functions with descriptions
- Represents logical conditions on feature vectors

---

## Constructor Parameters

```python
DSClassifierMultiQ(num_classes, lr=0.005, max_iter=200, min_iter=2,
                   min_dloss=0.0001, optim="adam", lossfn="MSE",
                   debug_mode=False, step_debug_mode=False,
                   batch_size=4000, num_workers=1,
                   precompute_rules=False, device="cpu",
                   force_precompute=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | **Required** | Number of classes in the classification problem |
| `lr` | float | 0.005 | Initial learning rate for gradient descent |
| `max_iter` | int | 200 | Maximum number of training epochs |
| `min_iter` | int | 2 | Minimum number of epochs before convergence check |
| `min_dloss` | float | 0.0001 | Minimum loss variation to consider convergence |
| `optim` | str | "adam" | Optimizer: "adam" or "sgd" |
| `lossfn` | str | "MSE" | Loss function: "MSE" or "CE" (CrossEntropy) |
| `debug_mode` | bool | False | Enable detailed training metrics |
| `step_debug_mode` | bool | False | Enable step-by-step debugging |
| `batch_size` | int | 4000 | Batch size for training |
| `num_workers` | int | 1 | Number of workers for data loading |
| `precompute_rules` | bool | False | Cache rule evaluations per instance |
| `device` | str | "cpu" | Device: "cpu", "cuda", or "mps" |
| `force_precompute` | bool | False | Force precomputation (high memory usage) |

---

## Core Methods

### 1. fit()
```python
fit(X, y, add_single_rules=False, single_rules_breaks=2,
    add_mult_rules=False, column_names=None, **kwargs)
```

**Purpose**: Train the classifier using gradient descent optimization.

**Parameters**:
- `X`: Training feature matrix (numpy array, shape: [n_samples, n_features])
- `y`: Training labels (numpy array, shape: [n_samples], values: 0 to num_classes-1)
- `add_single_rules`: If True, automatically generate single-attribute rules
- `single_rules_breaks`: Number of breaks for statistical rule generation
- `add_mult_rules`: If True, generate pairwise multiplicative rules
- `column_names`: List of feature names for rule descriptions

**Returns**:
- `losses`: List of loss values per epoch
- `epoch`: Number of epochs completed
- `dt`: Training time (in debug mode)

**Training Process**:
1. Optionally generate rules automatically
2. Create optimizer (Adam or SGD) and loss criterion
3. Add index column to X for rule caching
4. Convert data to PyTorch tensors
5. Create DataLoader for batching
6. Training loop:
   - Forward pass through model
   - Compute loss
   - Backpropagation
   - Optimizer step
   - Normalize masses (ensure valid probability distributions)
7. Check convergence (if epoch > min_iter and loss change < min_dloss)

### 2. predict()
```python
predict(X, one_hot=False)
```

**Purpose**: Predict class labels for input samples.

**Parameters**:
- `X`: Feature matrix (numpy array or pandas DataFrame)
- `one_hot`: If True, return probability scores instead of class labels

**Returns**:
- Class predictions (if one_hot=False) or probability distributions (if one_hot=True)

**Process**:
1. Set model to evaluation mode
2. Clear rule cache
3. Add index column to X
4. Convert to PyTorch tensor
5. Perform forward pass (no gradient computation)
6. Return argmax (class) or probabilities

### 3. predict_proba()
```python
predict_proba(X)
```

**Purpose**: Get probability estimates for all classes.

**Returns**: Probability matrix (shape: [n_samples, n_classes])

### 4. predict_explain()
```python
predict_explain(x)
```

**Purpose**: Provide interpretable explanation for a single prediction.

**Parameters**:
- `x`: Single feature vector

**Returns**:
- `pred`: Probability distribution over classes
- `cls`: Predicted class
- `df_rls`: DataFrame with rules and their mass values
- `builder`: String explanation

---

## Internal Mechanisms

### Forward Pass (DSModelMultiQ.forward())

The forward pass implements the Dempster-Shafer combination rule using **commonality transformation** for efficiency.

#### Standard Mode (force_precompute=False)
```
For each sample x:
  1. Select applicable rules: sel = {i | rule_i(x) = True}
  2. Get masses: mt = masses[sel]
  3. Transform to commonalities: qt = mt[:, :-1] + mt[:, -1] * 1
  4. Combine: res = product(qt along rules)
  5. Normalize: out = res / sum(res)
```

#### Precompute Mode (force_precompute=True)
```
1. Transform all masses to commonalities: qs
2. Repeat for all samples: qt
3. Select applicable rules per sample: sel
4. Set non-applicable rules to 1: qt[sel] = 1
5. Product along rules: res = product(qt, dim=1)
6. Normalize: out = res / sum(res)
```

**Commonality Transformation**:
- Converts mass functions to commonality functions
- For singletons: q(A) = m(A) + m(Ω)
- Where Ω is uncertainty (all possible classes)
- **Key Advantage**: Dempster's rule becomes simple product in commonality space

### Mass Normalization

After each optimization step, masses are normalized:
```python
1. Clamp to [0, 1]: t.clamp_(0., 1.)
2. If sum < 1: Add remainder to uncertainty (last element)
3. If sum >= 1: Divide by sum to normalize
```

This ensures:
- All masses are valid probabilities
- Total mass equals 1
- Uncertainty absorbs numerical errors

### Rule Selection and Caching

**Without precompute_rules**:
- Rules evaluated every forward pass
- Simple but slower

**With precompute_rules**:
- First evaluation cached in `rmap` dictionary
- Keyed by sample index
- Significant speedup for iterative training

**With force_precompute**:
- All rule evaluations computed once
- Stored as boolean tensor
- Highest memory usage
- Fastest forward pass

---

## Memory Usage Analysis

### Memory Components

#### 1. Model Parameters (Trainable)
```
Size = num_rules × (num_classes + 1) × sizeof(float32)
Example: 100 rules, 3 classes
  = 100 × 4 × 4 bytes = 1.6 KB
```
**Impact**: Negligible, even for thousands of rules

#### 2. Input Data Tensors
```
X: n_samples × n_features × sizeof(float32)
y: n_samples × sizeof(long) (or n_samples × num_classes for MSE)

Example: 10,000 samples, 20 features
  X = 10,000 × 20 × 4 = 800 KB
  y = 10,000 × 4 = 40 KB (CE) or 10,000 × 3 × 4 = 120 KB (MSE)
```
**Impact**: Moderate, scales with dataset size

#### 3. Rule Cache (precompute_rules=True)
```
rmap: dict mapping sample_index -> list of rule indices
Average: n_samples × avg_rules_per_sample × sizeof(int)

Example: 10,000 samples, avg 50 applicable rules
  = 10,000 × 50 × 4 bytes = 2 MB
```
**Impact**: Moderate, depends on rule density

#### 4. Precomputed Rules (force_precompute=True)
```
_all_rules: n_samples × num_rules × sizeof(bool)

Example: 10,000 samples, 100 rules
  = 10,000 × 100 × 1 byte = 1 MB

Large case: 100,000 samples, 1,000 rules
  = 100,000 × 1,000 × 1 byte = 100 MB
```
**Impact**: HIGH - Can be prohibitive for large datasets/many rules

#### 5. Batch Processing Tensors
```
Per batch:
  - qt (commonalities): batch_size × num_rules × num_classes × sizeof(float32)

Example: batch_size=4000, 100 rules, 3 classes
  = 4,000 × 100 × 3 × 4 = 4.8 MB per batch
```
**Impact**: Moderate, cleared between batches

#### 6. Gradient Tensors (Training)
```
Approximately doubles parameter memory:
  = 2 × num_rules × (num_classes + 1) × sizeof(float32)
```
**Impact**: Negligible for masses, but consider intermediate gradients

### Total Memory Estimates

| Configuration | 10K samples, 100 rules, 3 classes | 100K samples, 1K rules, 10 classes |
|---------------|-----------------------------------|-------------------------------------|
| **Minimal** (no precompute) | ~5 MB | ~60 MB |
| **With precompute_rules** | ~7 MB | ~70 MB |
| **With force_precompute** | ~6 MB | ~160 MB |

**CUDA Memory**: Add ~20-50% overhead for CUDA context and PyTorch operations

---

## Performance Bottlenecks

### Identified Bottlenecks

#### 1. Rule Evaluation (Without Caching)
**Location**: `DSModelMultiQ._select_rules()`
```python
for i in range(self.n):
    if self.preds[i](x):  # Python lambda call per rule per sample
        sel.append(i)
```

**Impact**:
- O(n_samples × num_rules) lambda calls
- Python function overhead
- No vectorization

**Severity**: HIGH for many rules/large datasets

**Mitigation**: Use `precompute_rules=True` or `force_precompute=True`

#### 2. Rule Evaluation in Precompute Mode
**Location**: `DSModelMultiQ._select_all_rules()`
```python
for i, sample, index in zip(count(), X, indices):
    for j in range(self.n):
        sel[i, j] = not bool(self.preds[j](sample))  # Still Python lambda
```

**Impact**:
- Still O(n_samples × num_rules) but amortized over epochs
- One-time cost at start of training

**Severity**: MEDIUM (one-time cost)

#### 3. Batch Product in Forward Pass
**Location**: `DSModelMultiQ.forward()`
```python
res = qt.prod(1)  # Product along rules dimension
```

**Impact**:
- O(batch_size × num_rules × num_classes)
- Vectorized but still grows with rule count
- Can cause numerical underflow with many rules

**Severity**: MEDIUM to HIGH with many rules

**Mitigation**: Consider log-space computation for many rules

#### 4. Normalization After Each Batch
**Location**: `DSModelMultiQ.normalize()`
```python
for t in self._params:  # Iterates over all rules
    t.clamp_(0., 1.)
    # ... normalization logic
```

**Impact**:
- O(num_rules) per batch
- Not vectorized over rules

**Severity**: LOW to MEDIUM

**Possible Optimization**: Vectorize normalization

#### 5. Data Transfer (CPU ↔ GPU)
**Locations**: Throughout forward pass
```python
x = x.cpu().data.numpy()  # GPU to CPU for lambda evaluation
```

**Impact**:
- Memory bandwidth bottleneck
- Required for Python lambda rules

**Severity**: HIGH when using CUDA

**Root Cause**: Rules are Python lambdas, not CUDA kernels

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Rule Generation | O(n_features × breaks × n_samples) | O(num_rules) |
| Rule Evaluation (no cache) | O(n_samples × num_rules × epochs) | O(1) |
| Rule Evaluation (precompute) | O(n_samples × num_rules) | O(n_samples × num_rules) |
| Forward Pass | O(batch_size × num_rules × num_classes) | O(batch_size × num_rules × num_classes) |
| Backward Pass | O(batch_size × num_rules × num_classes) | O(num_rules × num_classes) |

---

## CUDA Optimization Opportunities

### Current CUDA Support
- ✅ Tensor operations (forward/backward) run on GPU
- ✅ Data tensors transferred to GPU
- ❌ Rule evaluation runs on CPU (Python lambdas)
- ❌ Frequent CPU-GPU transfers for rule selection

### Optimization Strategies

#### 1. **Vectorized Rule Evaluation** ⭐⭐⭐
**Current**: Python lambdas evaluated per sample
```python
if self.preds[i](x):  # Python function call
```

**Proposed**: Convert rules to vectorized tensor operations
```python
# Example: Rule "x[0] > 5.2"
rule_tensor = X[:, 0] > 5.2  # Fully vectorized on GPU
```

**Benefits**:
- Eliminates CPU-GPU transfers
- Full GPU acceleration
- 10-100× speedup possible

**Challenges**:
- Requires rule compilation/translation
- Complex for arbitrary lambdas
- May need rule DSL (domain-specific language)

**Implementation Complexity**: HIGH

#### 2. **Force Precompute Mode** ⭐⭐
**Current Implementation**: Already available via `force_precompute=True`

**Optimization**:
```python
# Move boolean tensor to GPU
self._all_rules = torch.zeros(n, self.n, dtype=torch.bool).to(self.device)
```

**Benefits**:
- One-time CPU cost
- All subsequent operations on GPU
- 2-5× speedup in forward pass

**Trade-offs**:
- High memory usage: O(n_samples × num_rules)
- Not suitable for very large datasets

**Implementation Complexity**: LOW (already partially implemented)

#### 3. **Fused Kernels for Combination** ⭐⭐⭐
**Current**: Separate operations for transformation and product
```python
qt = mt[:, :-1] + mt[:, -1].view(-1, 1) * torch.ones_like(mt[:, :-1])
res = qt.prod(1)
```

**Proposed**: Custom CUDA kernel for combined operation
```cuda
__global__ void dempster_combination_kernel(
    float* masses, bool* rule_mask, float* output,
    int batch_size, int num_rules, int num_classes)
```

**Benefits**:
- Reduced memory bandwidth
- Eliminate intermediate tensors
- 2-3× speedup possible

**Implementation Complexity**: HIGH (requires CUDA C++)

#### 4. **Log-Space Computation** ⭐⭐
**Current**: Direct product (numerical underflow risk)
```python
res = qt.prod(1)
```

**Proposed**: Log-space for numerical stability
```python
log_qt = torch.log(qt)
log_res = log_qt.sum(1)
res = torch.exp(log_res)
```

**Benefits**:
- Numerical stability with many rules
- Same computational complexity
- Enables more rules without underflow

**Implementation Complexity**: LOW

#### 5. **Mixed Precision Training** ⭐
**Current**: FP32 throughout

**Proposed**: Use AMP (Automatic Mixed Precision)
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    y_pred = self.model.forward(Xi)
    loss = criterion(y_pred, yi)
```

**Benefits**:
- 1.5-2× speedup on modern GPUs
- Reduced memory usage
- Maintained accuracy

**Implementation Complexity**: LOW

#### 6. **Batch Parallelism** ⭐
**Current**: Sequential batch processing

**Proposed**: Overlap data transfer and computation
```python
# Use pinned memory and streams
train_loader = DataLoader(..., pin_memory=True)
# Prefetch next batch while processing current
```

**Benefits**:
- Hide data transfer latency
- 10-20% speedup

**Implementation Complexity**: MEDIUM

### Recommended Implementation Priority

1. **Force Precompute Mode** (if memory allows) - Immediate 2-5× speedup
2. **Log-Space Computation** - Better numerical stability
3. **Mixed Precision Training** - Easy 1.5-2× speedup
4. **Vectorized Rule Evaluation** - Biggest potential, highest effort

---

## Speedup Recommendations Summary

### Quick Wins (Minimal Code Changes)

1. **Enable force_precompute**:
   ```python
   DSC = DSClassifierMultiQ(3, force_precompute=True, device="cuda")
   ```
   - Expected: 2-5× training speedup
   - Cost: O(n_samples × num_rules) memory

2. **Increase batch_size** (with CUDA):
   ```python
   DSC = DSClassifierMultiQ(3, batch_size=16000, device="cuda")
   ```
   - Expected: 1.5-2× speedup
   - Requires: More GPU memory

3. **Use CUDA if available**:
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   DSC = DSClassifierMultiQ(3, device=device)
   ```
   - Expected: 5-10× speedup over CPU
   - Requires: NVIDIA GPU with CUDA

### Medium-Term Optimizations (Code Modifications)

4. **Implement Mixed Precision Training**
   - Modify `_optimize()` method to use AMP
   - Expected: 1.5-2× additional speedup on modern GPUs

5. **Optimize Normalization**
   - Vectorize mass normalization across all rules
   - Expected: 10-20% speedup

### Long-Term Optimizations (Significant Refactoring)

6. **Rule Vectorization Framework**
   - Create tensor-based rule representation
   - Expected: 10-100× speedup
   - Effort: HIGH

7. **Custom CUDA Kernels**
   - Fused operations for forward pass
   - Expected: 2-3× additional speedup
   - Effort: HIGH

---

## Best Practices

### Memory Efficiency
1. Start with `precompute_rules=False` for very large datasets
2. Use `precompute_rules=True` for moderate datasets with many epochs
3. Only use `force_precompute=True` when memory allows and speed is critical
4. Monitor memory with: `torch.cuda.memory_allocated()` (CUDA)

### Performance Tuning
1. **Rule Count**: Keep under 500 rules for best performance
2. **Batch Size**:
   - CPU: 1000-4000 samples
   - GPU: 8000-16000 samples (or larger)
3. **Learning Rate**: Start with 0.005, adjust based on convergence
4. **Convergence**: Use `min_dloss=1e-7` for better accuracy

### CUDA Usage
```python
if torch.cuda.is_available():
    device = "cuda"
    batch_size = 16000  # Larger batches on GPU
    num_workers = 0  # GPU doesn't need multiple workers
else:
    device = "cpu"
    batch_size = 4000
    num_workers = 4  # CPU benefits from parallelism

DSC = DSClassifierMultiQ(
    num_classes=3,
    device=device,
    batch_size=batch_size,
    num_workers=num_workers,
    force_precompute=True  # If memory allows
)
```

---

## Compatibility Notes

### Environment Tested
- **Python**: 3.12.3
- **PyTorch**: 2.10.0 (with CUDA 12.9)
- **NumPy**: 2.4.3
- **Pandas**: 3.0.1
- **Scikit-learn**: 1.8.0
- **SciPy**: 1.17.1

### Breaking Changes from Old Versions
- NumPy 2.0+: Removed some deprecated APIs (code is compatible)
- PyTorch 2.0+: Changed gradient behavior (code handles correctly)
- Pandas 3.0+: Changed some indexing behavior (code is compatible)

### Known Issues
1. **UserWarning about non-writable arrays**: Harmless warning from PyTorch
2. **MPS Backend**: Limited support, may be slower than CPU for small batches
3. **num_workers > 0 with CUDA**: May cause errors, use `num_workers=0`

---

## Conclusion

The `DSClassifierMultiQ` class provides a robust, interpretable classification framework with good performance characteristics. The main bottleneck is rule evaluation in Python, which can be mitigated through caching strategies. For production use with large datasets, consider implementing vectorized rule evaluation or using force_precompute mode with sufficient memory.

**Key Takeaways**:
- Use CUDA for 5-10× speedup
- Enable `force_precompute` if memory allows (2-5× speedup)
- Keep rule count reasonable (< 500 rules)
- Rule evaluation is the primary bottleneck
- Vectorization of rules would provide the biggest performance gain
