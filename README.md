<div align="center">

# 🧠 Micrograd (From Scratch)

*A tiny, scalar-valued autograd engine and neural network library implemented purely in Python.*

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python_3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/Environment-Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook Badge" />
  <img src="https://img.shields.io/badge/Framework-None_(From_Scratch)-success?style=for-the-badge" alt="From Scratch" />
</p>

> *"You just do it right, and do it everyday. -Shinsuke Kita"*

<br />

</div>

---

## 📖 Overview

Welcome to my from-scratch implementation of an **automatic differentiation** (autograd) engine. Inspired by Andrej Karpathy's famous Micrograd, this repository breaks down the magic of deep learning into its most fundamental building blocks.

It operates on dynamically built scalar-valued computational graphs and implements backpropagation to compute gradients, allowing you to build and train simple neural networks without relying on massive frameworks like PyTorch or TensorFlow. 

If you want to understand how deep learning *actually* works under the hood, this is where you start.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🔢 **Scalar Autograd** | The `Value` class wraps floats and supports fundamental operations (`+`, `-`, `*`, `/`, `**`, `exp`, `tanh`). |
| 🕸️ **Dynamic Graphs** | Dynamically builds a Directed Acyclic Graph (DAG) of computations as mathematical operations are performed. |
| ⏪ **Backpropagation** | A simple `.backward()` call triggers a topological sort to compute the gradient of every node via the chain rule. |
| 📊 **Visualization** | Built-in support for rendering the computational graph using `graphviz` to inspect forward and backward passes. |

---

## 🛠️ How It Works (Under the Hood)

The beating heart of this library is the `Value` class. It stores both the **data** (the forward pass) and the **gradient** (the backward pass).

<details>
<summary><b>1. The Forward Pass</b></summary>
<br>
Every time you add, multiply, or apply a function (like `tanh`) to a `Value` object, a new `Value` is created. This new object silently records:
<ul>
  <li>The mathematical operation that created it.</li>
  <li>Pointers to its "children" (the specific operands used).</li>
</ul>
This essentially records history, building out the computation graph.
</details>

<details>
<summary><b>2. The Backward Pass</b></summary>
<br>
When you call <code>.backward()</code>, the engine performs a <b>topological sort</b> to order all the nodes. It then walks backward through this ordered list, applying the chain rule of calculus at every step to route gradients from the final output all the way back to the initial inputs.
</details>

---

## 🚀 Quick Start & Example

Here is a simple example showing how to build a small expression, execute the forward pass, and backpropagate the gradients:

```python
from micrograd import Value

# 1. Initialize Inputs
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# 2. Build the computation graph (Forward Pass)
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')

# 3. Final Output
L = d * f; L.label = 'L'

# 4. Run Backpropagation
L.backward()

# 5. Inspect Gradients
print(f"Gradient of a: {a.grad}") # Output: -6.0
print(f"Gradient of b: {b.grad}") # Output:  4.0
print(f"Gradient of c: {c.grad}") # Output: -2.0
```

---

## 🎨 Visualizing the Graph

To make debugging beautiful and intuitive, you can trace and render the entire computational graph, including data values and gradients at every node:

```python
from graphviz import Digraph
# ... build your graph (like 'L' above) ...

# Renders an SVG of the computation graph
draw_dot(L) 
```

<div align="center">
  <p><i>The resulting graph displays rectangular nodes for values, and oval nodes for the mathematical operations linking them!</i></p>
</div>

---

## 🙏 Acknowledgements

Huge thanks to **Andrej Karpathy** for his incredible [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series, which heavily inspired and guided the creation of this project.

<div align="center">
  <br>
  <p><i>Built with curiosity and Python.</i></p>
</div>
