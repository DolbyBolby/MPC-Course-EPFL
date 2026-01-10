# MPC Cost Matrices Documentation

## Overview
This document details the cost function matrices **Q** and **R** for each MPC velocity controller in Deliverable 3.1.

The general cost function is defined as:
$$J = \sum_{i=0}^{N-1} \left( (x_i - x_s)^T Q (x_i - x_s) + (u_i - u_s)^T R (u_i - u_s) \right) + (x_N - x_s)^T Q_f (x_N - x_s)$$

where:
- $x_s$ = equilibrium state
- $u_s$ = equilibrium input
- $N$ = prediction horizon (number of steps)
- $Q_f$ = terminal cost matrix (computed via DLQR)

---

## State and Control Indices

Each controller operates on a reduced-dimensional subsystem:

| Controller | State Indices | Control Index | State Dimension | Control Dimension | States |
|:----------:|:-------------:|:-------------:|:----------------:|:------------------:|:--------|
| **xvel** | [1, 4, 6] | [1] | 3 | 1 | [y-pos, y-accel, x-vel] |
| **yvel** | [0, 3, 7] | [0] | 3 | 1 | [x-pos, x-accel, y-vel] |
| **zvel** | [9, 11] | [2] | 2 | 1 | [z-vel, z-accel] |
| **roll** | [2, 5] | [3] | 2 | 1 | [roll, roll-rate] |

---

## Cost Matrices Q and R

### xvel (X-Velocity Controller)

**State Dimension:** 3×3  
**Control Dimension:** 1×1  
**State Variables:** [y-position, y-acceleration, x-velocity]

| Metric | Matrix | Values |
|:------:|:------:|:------:|
| **Q** | Diagonal | diag([5.0, 200.0, 50.0]) |
| **R** | Diagonal | 1.0 × I₁ |
| **Interpretation** | - | High penalty on y-acceleration (200) to maintain pitch angle constraints |

```python
Q = np.diag([5.0, 200.0, 50.0])
R = 1 * np.eye(1)
```

**Matrix Form:**
$$Q = \begin{bmatrix} 5.0 & 0 & 0 \\ 0 & 200.0 & 0 \\ 0 & 0 & 50.0 \end{bmatrix}, \quad R = \begin{bmatrix} 1.0 \end{bmatrix}$$

---

### yvel (Y-Velocity Controller)

**State Dimension:** 3×3  
**Control Dimension:** 1×1  
**State Variables:** [x-position, x-acceleration, y-velocity]

| Metric | Matrix | Values |
|:------:|:------:|:------:|
| **Q** | Diagonal | diag([5.0, 200.0, 50.0]) |
| **R** | Diagonal | 1.0 × I₁ |
| **Interpretation** | - | Identical to xvel for symmetry |

```python
Q = np.diag([5.0, 200.0, 50.0])
R = 1 * np.eye(1)
```

**Matrix Form:**
$$Q = \begin{bmatrix} 5.0 & 0 & 0 \\ 0 & 200.0 & 0 \\ 0 & 0 & 50.0 \end{bmatrix}, \quad R = \begin{bmatrix} 1.0 \end{bmatrix}$$

---

### zvel (Z-Velocity Controller)

**State Dimension:** 2×2  
**Control Dimension:** 1×1  
**State Variables:** [z-velocity, z-acceleration]

| Metric | Matrix | Values |
|:------:|:------:|:------:|
| **Q** | Diagonal | 50.0 × I₂ |
| **R** | Diagonal | 0.1 × I₁ |
| **Interpretation** | - | Lower input penalty (0.1) to allow faster thrust changes |

```python
Q = 50 * np.eye(2)
R = 0.1 * np.eye(1)
```

**Matrix Form:**
$$Q = \begin{bmatrix} 50.0 & 0 \\ 0 & 50.0 \end{bmatrix}, \quad R = \begin{bmatrix} 0.1 \end{bmatrix}$$

---

### roll (Roll Controller)

**State Dimension:** 2×2  
**Control Dimension:** 1×1  
**State Variables:** [roll angle, roll rate]

| Metric | Matrix | Values |
|:------:|:------:|:------:|
| **Q** | Diagonal | 1.0 × I₂ |
| **R** | Diagonal | 1.0 × I₁ |
| **Interpretation** | - | Balanced penalties for attitude control |

```python
Q = 1 * np.eye(2)
R = 1 * np.eye(1)
```

**Matrix Form:**
$$Q = \begin{bmatrix} 1.0 & 0 \\ 0 & 1.0 \end{bmatrix}, \quad R = \begin{bmatrix} 1.0 \end{bmatrix}$$

---

## Summary Comparison - All Matrices

| Controller | Dimension | Q Matrix | R Matrix | Notes |
|:----------:|:---------:|:--------:|:--------:|:------|
| **xvel** | 3×3, 1×1 | $$\begin{bmatrix} 5 & 0 & 0 \\ 0 & 200 & 0 \\ 0 & 0 & 50 \end{bmatrix}$$ | $$\begin{bmatrix} 1 \end{bmatrix}$$ | High penalty on acceleration (200) |
| **yvel** | 3×3, 1×1 | $$\begin{bmatrix} 5 & 0 & 0 \\ 0 & 200 & 0 \\ 0 & 0 & 50 \end{bmatrix}$$ | $$\begin{bmatrix} 1 \end{bmatrix}$$ | Symmetric with xvel |
| **zvel** | 2×2, 1×1 | $$\begin{bmatrix} 50 & 0 \\ 0 & 50 \end{bmatrix}$$ | $$\begin{bmatrix} 0.1 \end{bmatrix}$$ | Low input penalty (0.1) for vertical |
| **roll** | 2×2, 1×1 | $$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$ | $$\begin{bmatrix} 1 \end{bmatrix}$$ | Minimal penalties for attitude |

---

## Physical Interpretation

### Q Matrix (State Penalty)
- **High Q values** penalize large deviations from equilibrium
- xvel/yvel: High acceleration penalty (200) enforces pitch angle constraints
- zvel: Moderate penalty (50) for smoother vertical motion
- roll: Low penalty (1) allows more aggressive attitude changes

### R Matrix (Control Penalty)
- **Higher R values** penalize aggressive control inputs
- xvel/yvel: R=1.0 limits thrust magnitude
- zvel: R=0.1 allows larger thrust variations (lower inertia)
- roll: R=1.0 balanced control

---

## Terminal Cost Matrix Qf

The terminal cost matrix **Qf** is computed automatically using discrete-time Linear Quadratic Regulator (DLQR):

```python
K, Qf, _ = dlqr(self.A, self.B, Q, R)
K = -K  # State feedback gain
```

This ensures **recursive feasibility** and guarantees stability of the closed-loop system.

---

## Configuration Parameters

- **Sampling Time:** Ts = 0.05 s
- **Prediction Horizon:** H = 5-7 seconds (N = 100-140 steps)
- **Solver:** CVXPY with PIQP/OSQP backend
- **Discretization:** Zero-order hold via `scipy.signal.cont2discrete()`

