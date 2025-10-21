# Understanding PyTorch Training Loop: Weights, Gradients, and Zeroing

This file details the fundamental concepts of updating weights and biases in a simple linear regression model using PyTorch. We'll walk through the process manually, then map each step to PyTorch code, focusing on how `w.grad.zero_()`, `optimizer.zero_grad()`, and `torch.no_grad()` interact with parameters and their gradients.

## The Model and Data

We use a simple linear regression model:
`y_hat = w * x + b`
Where:
- `y_hat` is the predicted output.
- `w` is the weight (a single scalar parameter).
- `b` is the bias (a single scalar parameter).
- `x` is the input.

Our goal is to learn `w` and `b` to minimize the difference between `y_hat` and the true output `y`.

**Dataset:**
We'll use a small synthetic dataset:
`X = [1, 2, 3, 4, 5]`
`y = 1 * X + 3` (The true relationship is `w=1, b=3`)

This means our true `y` values are:
| x | true y |
|---|--------|
| 1 | 4      |
| 2 | 5      |
| 3 | 6      |
| 4 | 7      |
| 5 | 8      |

**Loss Function:** Mean Squared Error (MSE)
`L = (1/N) * sum( (y_hat_i - y_i)^2 )`

**Initial Parameters:**
We'll start with `w=0.0` and `b=0.0`.
**Learning Rate (eta):** `0.01`

---

## Manual Walkthrough: First Iteration

Let's trace one full update cycle (forward pass, loss, gradients, update) manually.

### Step 0: Setup - Parameters & Data
Initial: `w = 0.0`, `b = 0.0`
`X = [1, 2, 3, 4, 5]`
`y = [4, 5, 6, 7, 8]`

### Step 1: Forward Pass (Compute Predictions `y_hat`)
`y_hat_i = w * x_i + b`
Since `w=0` and `b=0`:
| x | y_true | `y_hat = 0 * x + 0` | error = `y_hat` - y_true | error² |
|---|--------|---------------------|--------------------------|--------|
| 1 | 4      | 0                   | -4                       | 16     |
| 2 | 5      | 0                   | -5                       | 25     |
| 3 | 6      | 0                   | -6                       | 36     |
| 4 | 7      | 0                   | -7                       | 49     |
| 5 | 8      | 0                   | -8                       | 64     |

### Step 2: Compute Loss (MSE)
`L = (1/N) * sum( (y_hat_i - y_i)^2 )`
`L = (1/5) * (16 + 25 + 36 + 49 + 64) = 190/5 = 38`

### Step 3: Compute Gradients (`dL/dw`, `dL/db`)
The gradients for MSE (mean reduction) are:
`dL/dw = (2/N) * sum_i( (y_hat_i - y_i) * x_i )`
`dL/db = (2/N) * sum_i( (y_hat_i - y_i) )`

Let's use the `error = (y_hat - y_true)` from Step 1: `[-4, -5, -6, -7, -8]`

| x | error | error * x |
|---|-------|-----------|
| 1 | -4    | -4        |
| 2 | -5    | -10       |
| 3 | -6    | -18       |
| 4 | -7    | -28       |
| 5 | -8    | -40       |

`Sum(error * x) = -4 - 10 - 18 - 28 - 40 = -100`
`Sum(error) = -4 - 5 - 6 - 7 - 8 = -30`

Now apply the formulas:
`dL/dw = (2/5) * (-100) = -40`
`dL/db = (2/5) * (-30) = -12`
So, `grad_w = -40`, `grad_b = -12`.

### Step 4: Gradient Descent Update
`w := w - eta * dL/dw`
`b := b - eta * dL/db`
Using `eta = 0.01`:
`w = 0 - 0.01 * (-40) = 0 + 0.4 = 0.4`
`b = 0 - 0.01 * (-12) = 0 + 0.12 = 0.12`

**After 1st update:** `w = 0.4`, `b = 0.12`

---

## Manual Walkthrough: Second Iteration (The `zero_grad()` Effect)

Now, let's see two scenarios for the second iteration: with and without `zero_grad()`.

### Scenario A: With `zero_grad()` (The Correct Way)

Assume `w = 0.4`, `b = 0.12`. We first **zero the gradients**.
`w.grad = 0`, `b.grad = 0` (this is crucial)

#### Step 1: Forward Pass (New Predictions)
`y_hat_i = 0.4 * x_i + 0.12`
| x | y_true | `y_hat = 0.4x + 0.12` | error = `y_hat` - y_true |
|---|--------|-----------------------|--------------------------|
| 1 | 4      | 0.52                  | -3.48                    |
| 2 | 5      | 0.92                  | -4.08                    |
| 3 | 6      | 1.32                  | -4.68                    |
| 4 | 7      | 1.72                  | -5.28                    |
| 5 | 8      | 2.12                  | -5.88                    |

#### Step 2: Compute Loss
(We won't detail the sum, but it would be `~ 24.096`)

#### Step 3: Compute New Gradients
Using new errors:
`Sum(error * x) = (-3.48 * 1) + (-4.08 * 2) + ... + (-5.88 * 5) = -76.2`
`Sum(error) = (-3.48) + (-4.08) + ... + (-5.88) = -23.4`

New gradients:
`dL/dw = (2/5) * (-76.2) = -30.48`
`dL/db = (2/5) * (-23.4) = -9.36`
**Since we zeroed, PyTorch now sets `w.grad = -30.48`, `b.grad = -9.36`**.

#### Step 4: Gradient Descent Update
`w = 0.4 - 0.01 * (-30.48) = 0.4 + 0.3048 = 0.7048`
`b = 0.12 - 0.01 * (-9.36) = 0.12 + 0.0936 = 0.2136`
**After 2nd update (with zero_grad):** `w = 0.7048`, `b = 0.2136` (A controlled, intended step).

---

### Scenario B: Without `zero_grad()` (The Accumulation Problem)

Assume `w = 0.4`, `b = 0.12`.
**Crucially, `w.grad` is still `-40` and `b.grad` is still `-12` from the first iteration.**

#### Step 1 & 2: Forward Pass & Loss
(Same as Scenario A, predictions and errors are identical).

#### Step 3: Compute New Gradients
The raw new gradients are `-30.48` for `w` and `-9.36` for `b`.
**BUT, without `zero_grad()`, PyTorch *adds* these to the existing gradients:**
`w.grad` becomes `(-40) + (-30.48) = -70.48`
`b.grad` becomes `(-12) + (-9.36) = -21.36`

#### Step 4: Gradient Descent Update
`w = 0.4 - 0.01 * (-70.48) = 0.4 + 0.7048 = 1.1048`
`b = 0.12 - 0.01 * (-21.36) = 0.12 + 0.2136 = 0.3336`
**After 2nd update (without zero_grad):** `w = 1.1048`, `b = 0.3336` (An overshot, unintended step).

This clearly shows why `zero_grad()` is essential: to prevent gradients from accumulating across iterations and causing incorrect, unstable updates.

---

## PyTorch Code Walkthrough

Let's map these manual steps to actual PyTorch code. The comments explain what each line does.

```python
import torch
import torch.optim as optim

# --- 0. DATA AND MODEL INITIALIZATION ---
X = torch.tensor([1., 2., 3., 4., 5.])         
y = 1.0 * X + 3.0                               

# requires_grad=True to TRACK THE GRADIENTS!
w = torch.tensor(0.0, requires_grad=True)        
b = torch.tensor(0.0, requires_grad=True)        

lr = 0.01                                        # Learning rate
N = X.shape[0]                                   # Number of samples (for MSE mean)

print("--- Initial State ---")
print(f"w = {w.item():.4f}, b = {b.item():.4f}")
print(f"w.grad = {w.grad}, b.grad = {b.grad}") # Initially None
print("-" * 30)

# --- ITERATION 1: Demonstrating one full step ---

print("\n=== ITERATION 1 (with zero_grad) ===")

# Manual equivalent: w.grad = 0, b.grad = 0
# At this point, w.grad and b.grad are still None
# or, if a backward pass happened, they'd have values.
# For the first pass, they don't exist yet, so zero_() would be skipped.
# We'll explicitly set them to None to simulate a clean start for demonstration clarity.
if w.grad is not None:
    w.grad = None
if b.grad is not None:
    b.grad = None
print(f"1.1 After manual 'zero_grad' (for clarity, setting to None): w.grad={w.grad}, b.grad={b.grad}")

# --- Step 1: FORWARD PASS ---
# Math: y_hat_i = w * x_i + b
y_pred = w * X + b
print(f"1.2 Forward Pass: y_pred = {y_pred.tolist()}")
# Expected: [0.0, 0.0, 0.0, 0.0, 0.0] (since w=0, b=0)

# --- Step 2: COMPUTE LOSS (MSE) ---
# Math: L = (1/N) * sum_i (y_hat_i - y_i)^2
loss = ((y_pred - y) ** 2).mean()
print(f"1.3 Loss: {loss.item():.2f}")
# Expected: 38.0

# --- Step 3: BACKWARD PASS (Autograd computes gradients) ---
# Math: dL/dw = (2/N) * sum(error * x); dL/db = (2/N) * sum(error)
loss.backward() # This computes dL/dw and dL/db and stores them in w.grad, b.grad
print(f"1.4 After Backward Pass: w.grad = {w.grad.item():.2f}, b.grad = {b.grad.item():.2f}")
# Expected: w.grad = -40.00, b.grad = -12.00 (matches manual calculation)

# --- Step 4: OPTIMIZER STEP (Update parameters) ---
# Math: parameter = parameter - lr * gradient
with torch.no_grad(): # Disable gradient tracking for this update operation
    w -= lr * w.grad  # Update w
    b -= lr * b.grad  # Update b
print(f"1.5 After Update: w = {w.item():.4f}, b = {b.item():.4f}")
# Expected: w = 0.4000, b = 0.1200 (matches manual calculation)

print("-" * 30)

# --- ITERATION 2, SCENARIO A: WITH zero_grad() (Correct way) ---

print("\n=== ITERATION 2A (WITH zero_grad) ===")

# --- Step A: ZERO GRADS ---
# Manual equivalent: w.grad = 0, b.grad = 0
w.grad.zero_()
b.grad.zero_()
print(f"2A.1 After zero_grad: w.grad={w.grad.item():.2f}, b.grad={b.grad.item():.2f}")
# Expected: w.grad = 0.00, b.grad = 0.00 (cleared from previous -40, -12)

# --- Step B: FORWARD PASS ---
y_pred = w * X + b
print(f"2A.2 Forward Pass: y_pred = {y_pred.tolist()}")
# Expected: [0.52, 0.92, 1.32, 1.72, 2.12] (0.4*x + 0.12)

# --- Step C: COMPUTE LOSS ---
loss = ((y_pred - y) ** 2).mean()
print(f"2A.3 Loss: {loss.item():.3f}")
# Expected: 24.096

# --- Step D: BACKWARD PASS ---
loss.backward()
print(f"2A.4 After Backward Pass: w.grad = {w.grad.item():.2f}, b.grad = {b.grad.item():.2f}")
# Expected: w.grad = -30.48, b.grad = -9.36 (new gradients for current w,b)

# --- Step E: OPTIMIZER STEP ---
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
print(f"2A.5 After Update: w = {w.item():.4f}, b = {b.item():.4f}")
# Expected: w = 0.7048, b = 0.2136 (controlled update)

print("-" * 30)

# --- ITERATION 2, SCENARIO B: WITHOUT zero_grad() (Accumulation problem) ---

# To demonstrate, we need to reset w,b and their grads to the state
# *after* the first update, as if zero_grad() was never called for iteration 2.
w_no_zero = torch.tensor(0.4, requires_grad=True)
b_no_zero = torch.tensor(0.12, requires_grad=True)

# Manually assign the gradients that would have been left from iteration 1
# This simulates forgetting to call zero_grad() before the second backward pass
w_no_zero.grad = torch.tensor(-40.0)
b_no_zero.grad = torch.tensor(-12.0)

print("\n=== ITERATION 2B (WITHOUT zero_grad) ===")
print(f"Initial state for 2B: w = {w_no_zero.item():.4f}, b = {b_no_zero.item():.4f}")
print(f"  Existing w.grad = {w_no_zero.grad.item():.2f}, b.grad = {b_no_zero.grad.item():.2f}")
# Expected: w.grad = -40.00, b.grad = -12.00

# --- Step B: FORWARD PASS (Same as 2A) ---
y_pred_no_zero = w_no_zero * X + b_no_zero
print(f"2B.1 Forward Pass: y_pred = {y_pred_no_zero.tolist()}")
# Expected: [0.52, 0.92, 1.32, 1.72, 2.12]

# --- Step C: COMPUTE LOSS (Same as 2A) ---
loss_no_zero = ((y_pred_no_zero - y) ** 2).mean()
print(f"2B.2 Loss: {loss_no_zero.item():.3f}")
# Expected: 24.096

# --- Step D: BACKWARD PASS (!!! GRADIENTS ACCUMULATE !!!) ---
loss_no_zero.backward() # This will ADD new gradients to existing -40 and -12
print(f"2B.3 After Backward Pass: w.grad = {w_no_zero.grad.item():.2f}, b.grad = {b_no_zero.grad.item():.2f}")
# Expected: w.grad = (-40) + (-30.48) = -70.48
# Expected: b.grad = (-12) + (-9.36) = -21.36

# --- Step E: OPTIMIZER STEP ---
with torch.no_grad():
    w_no_zero -= lr * w_no_zero.grad
    b_no_zero -= lr * b_no_zero.grad
print(f"2B.4 After Update: w = {w_no_zero.item():.4f}, b = {b_no_zero.item():.4f}")
# Expected: w = 0.4 - 0.01 * (-70.48) = 1.1048
# Expected: b = 0.12 - 0.01 * (-21.36) = 0.3336
print("-" * 30)

# --- Using torch.optim.SGD for comparison (briefly) ---
print("\n=== Using torch.optim.SGD (standard practice) ===")
w_optim = torch.tensor(0.0, requires_grad=True)
b_optim = torch.tensor(0.0, requires_grad=True)
optimizer = optim.SGD([w_optim, b_optim], lr=lr)

# Simulating a training step using optimizer API
print("Simulating first optimizer step:")
y_pred_optim = w_optim * X + b_optim
loss_optim = ((y_pred_optim - y)**2).mean()

# The backward pass computes gradients
loss_optim.backward()
print(f"  w_optim.grad before optimizer.step(): {w_optim.grad.item():.2f}") # -40

# The optimizer.step() updates the parameters (w_optim, b_optim)
optimizer.step() 
print(f"  w_optim after optimizer.step(): {w_optim.item():.4f}") # 0.4

# IMPORTANT: Clear gradients for the next iteration
optimizer.zero_grad() 
print(f"  w_optim.grad after optimizer.zero_grad(): {w_optim.grad.item():.2f}") # 0.0
print("-" * 30)