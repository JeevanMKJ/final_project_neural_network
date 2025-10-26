# Definitions

This document contains definitions of the terms that I am learning during the learning journy of building the neural network. 
Definitions and concept descriptions will be updated and improved over the course of the development of the neural network.

### Weighted Sum

---
The formula for the **Weighted Sum** (or pre-activation value) $z$ inside a single neuron is:

$$z = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b$$ 

Where:
* $z$: the **Weighted Sum**.
* $x_i$: the individual input features (values from the previous layer or the initial data).
* $w_i$: the **weights** assigned to each input feature.
* $b$: the **bias term**.

Same formula but in E-notation:
$$z = \sum_{i=1}^{n} w_i x_i + b$$


### ReLu (Rectified Linear Unit) Function

---
The ReLu function is a non-linear piece-wise activation function used to introduce non-linearity into the neural network.

$$f(x) = \max(0, x)$$

The piece-wise function can be thought of as a logic gate.
It checks if an input is greater than 0.
If so it will pass through.
If it is negative it will be set to 0.


$$f(x) = \begin{cases} x & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

**Derivative of the $\text{ReLU}$ Function**

$$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

Where:
* $f'(z)$: the **derivative** of the $\text{ReLU}$ output with respect to its input $z$.
* $z$: the **Weighted Sum** (pre-activation value).
* $1$: the gradient for positive inputs, allowing the gradient to flow back.
* $0$: the gradient for non-positive inputs (leads to "dead" $\text{ReLU}$s).

### Sigmoid Function

---
The Sigmoid function is used at the output layer to display a binary result. **Output Range:** The function takes any real number input (from $-\infty$ to $+\infty$) and **compresses** it into a narrow range: $\mathbf{[0, 1]}$.

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Where:
* $x$ is the **Weighted Sum** (the raw score) calculated by the neuron.
* $e$ is Euler's number (the base of the natural logarithm).

**Derivative of the $\text{Sigmoid}$ Function**

$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$


Where:
* $\sigma'(z)$: the **derivative** of the $\text{Sigmoid}$ output with respect to its input $z$.
* $\sigma(z)$: the **Sigmoid function's output** (a value between 0 and 1).
* $\sigma(z)(1 - \sigma(z))$: the expression showing the derivative is always between **0 and 0.25**, which contributes to the **vanishing gradient problem**.

### Mean Squared Error (MSE) or Categorical Cross-Entropy (CCE)

--- 

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Where:
* $y_i$: the true target value.
* $\hat{y}_i$: the network's prediction (final output $a$ of the last layer).
* $N$: the number of samples.

**Derivative of $Mean Squared Error$ Function**

$$\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y)$$



