# 🕸️ Graph Neural Network (GNN) Implementations

[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-Graph%20ML-blue.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides from-scratch implementations of foundational Graph Neural Network (GNN) architectures using PyTorch Geometric (PyG). It focuses on graph construction, abstract message passing, and vertex classification on benchmark datasets.

## 📌 Project Overview

The project demonstrates the mechanics of node representation learning by decoupling GNN operations into `UPDATE`, `AGGREGATE`, and `MESSAGE` functions. By extending the `torch_geometric.nn.MessagePassing` base class, custom convolutional layers are built to process sparse graph representations comprising node features and edge indices in coordinate (COO) format.

## ✨ Key Features & Implementations

*   **Graph Convolutional Network (GCN):** Implements a custom `GCNConv` layer (Kipf & Welling, 2017). The forward pass explicitly handles adding self-loops to the adjacency matrix, linearly transforming node features, normalizing by node degree, and aggregating neighboring features:
    $$\mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{x}_j^{(k-1)}\mathbf{\Theta} \right)$$
*   **GraphSAGE:** Implements a custom `SAGEConv` layer (Hamilton et al., 2017) for inductive representation learning. The model is configured to evaluate multiple aggregation schemes, specifically `MEAN`, `SUM` (add), and `MAX` aggregations.
*   **Robust Training Pipeline:** Features a comprehensive training loop utilizing the Adam optimizer with weight decay, Negative Log-Likelihood loss (`F.nll_loss`) for multi-class classification, and an early stopping mechanism to prevent overfitting.
*   **Benchmark Testing:** Evaluates node classification performance on the Cora dataset (Planetoid), automatically handling training, validation, and testing masks.

## 🛠️ Technical Stack

*   **Core Frameworks:** PyTorch, PyTorch Geometric (PyG)
*   **Key Modules:** `torch_geometric.data.Data`, `torch_geometric.nn.MessagePassing`
*   **Data Processing:** `torch_geometric.datasets.Planetoid`

## 📊 Performance Benchmarks (Cora Dataset)

The models are trained and evaluated over 10 independent runs (200 epochs maximum, early stopping patience of 10) using a 16-unit hidden layer and a 0.5 dropout rate. 

*   **GCN:** 80.3% ± 0.007
*   **GraphSAGE (Mean Aggregation):** 79.4% ± 0.009
*   **GraphSAGE (Max Aggregation):** 76.7% ± 0.018
*   **GraphSAGE (Add Aggregation):** 75.3% ± 0.028

## 🚀 Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch
*   PyTorch Geometric (`torch-geometric`)

### Installation & Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/gnn-implementations.git](https://github.com/yourusername/gnn-implementations.git)
    cd gnn-implementations
    ```
2.  Install the required packages:
    ```bash
    pip install torch torch-geometric
    ```
3. Execute the notebook or training script. The script will automatically download the Cora dataset to a local `/data/Cora` directory and begin the training loop for both the GCN and GraphSAGE models.
