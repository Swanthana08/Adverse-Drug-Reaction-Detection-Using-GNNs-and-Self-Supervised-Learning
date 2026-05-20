# 💊 Drug-Drug Interaction (DDI) Prediction System

A machine learning web application that predicts **Drug-Drug Interaction (DDI) side effects** using Graph Neural Networks (GNN) and a CNN2D extension model, served via a Flask REST API with user authentication.

---

## 📌 Project Overview

Adverse drug interactions are a leading cause of preventable hospitalizations. Traditional lookup-based approaches fail to generalize to novel or unseen drug combinations. This project addresses that gap by learning from the **structural and relational properties of drugs** using graph-based deep learning.

Given two drugs (identified by their IDs and SMILES strings), the system predicts the **type of side effect interaction** that may occur between them.

---

## 🧠 Models & Approach

### Baseline Models
| Model | Description |
|---|---|
| KNN | K-Nearest Neighbors with TF-IDF drug vectors |
| Decision Tree | Classical tree-based classifier |

### Proposed Deep Learning Models
| Model | Description |
|---|---|
| **GraphCNN (GNN)** | Custom Keras graph convolutional layer that performs message passing over drug interaction graphs |
| **GraphAttentionCNN (GAT)** | Graph attention variant — assigns learned attention weights to neighbors, allowing the model to prioritize the most relevant drug relationships |
| **MultiGraphCNN** | Batched GNN supporting multiple graph inputs simultaneously |
| **MultiGraphAttentionCNN** | Multi-head attention version for batch graph inputs |
| **Extension CNN2D** | Final model — takes GNN-learned features and passes them through a 2D CNN with dropout layers to further refine classification |

### Why Graph-Based?
Drugs are not isolated entities — they share pharmacological, structural, and target-based relationships. Representing them as **graph nodes with interaction edges** lets the model reason about drugs in context, not just in isolation.

---

## 🗂️ Project Structure

```
├── app.py                          # Flask web application (routes, auth, prediction API)
├── graph_ops.py                    # Core graph convolution operation (graph_conv_op)
├── graph_cnn_layer.py              # GraphCNN Keras layer
├── graph_attention_cnn_layer.py    # GraphAttentionCNN (GAT) Keras layer
├── multi_graph_cnn_layer.py        # MultiGraphCNN for batched graphs
├── multi_graph_attention_cnn_layer.py  # Multi-head GAT for batched graphs
├── DrugSideEffectPrediction.ipynb  # Full training pipeline & evaluation notebook
├── Random_Forest.ipynb             # Baseline comparison notebook
├── Dataset/
│   ├── twosides_drugbank.csv       # Main drug interaction dataset (DrugBank + TWOSIDES)
│   ├── Interaction_information.csv # DDI type label descriptions
│   └── testData.csv                # Held-out test set
├── templates/                      # HTML templates for Flask UI
└── static/                         # Static files, result plots
```

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Dataset: `twosides_drugbank.csv` — each record contains `drug1_id`, `drug2_id`, `smiles1`, `smiles2`, and an interaction type label.
- Drug SMILES strings and IDs are concatenated and vectorized using **TF-IDF** to create numeric feature vectors.
- These vectors serve as node features for the GNN.

### 2. Graph Construction
- A graph adjacency matrix is constructed from known drug-drug interaction pairs.
- Graph convolution filters are derived from the adjacency matrix and passed into the custom Keras GNN layers.

### 3. Model Training
- **GNN** learns drug representations through neighborhood message passing.
- **GAT layers** apply multi-head attention so each drug node learns to focus on its most relevant neighbors.
- **CNN2D extension** reshapes GNN output vectors to `(N, features, 1, 1)` and applies 2D convolutions + dropout for final classification.
- Loss: Categorical cross-entropy | Optimizer: Adam

### 4. Evaluation
Compared against KNN and Decision Tree baselines on:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

### 5. Deployment
- Trained model saved as `.hdf5` weights and loaded by Flask.
- REST endpoint `/predict` accepts `drug1_id`, `drug2_id`, `smiles1`, `smiles2` via a web form.
- Predictions are stored per-user in a SQLite database.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install flask flask-sqlalchemy tensorflow keras scikit-learn pandas numpy werkzeug nbformat nbconvert
```

### Run the App
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

### Predict via Web UI
1. Register / Log in
2. Navigate to **Predict**
3. Enter `Drug 1 ID`, `Drug 2 ID`, `SMILES 1`, `SMILES 2`
4. Submit — the predicted DDI type is returned instantly

---

## 📊 Dataset

- **TWOSIDES**: A large-scale database of drug-drug interactions and their polypharmacy side effects.
- **DrugBank**: Supplementary drug identifiers and SMILES molecular structure strings.

Combined dataset: `twosides_drugbank.csv`

---

## 🏗️ Custom Layer Architecture

### `GraphCNN`
Standard GNN layer. Performs:
```
output = activation( graph_conv_op(X, num_filters, A_hat, W) + b )
```
where `A_hat` is the normalized adjacency filter matrix.

### `GraphAttentionCNN`
Multi-head GAT layer. For each attention head:
1. Project node features: `H = X @ W`
2. Compute pairwise attention: `e_ij = ELU(h_i · a + h_j · a)`
3. Mask non-edges and apply softmax: `α = softmax(e ⊙ mask)`
4. Apply dropout on attention coefficients
5. Aggregate: `output = α @ H`

Multi-head outputs are either **concatenated** or **averaged**.

### `MultiGraphAttentionCNN`
Batch-mode version of `GraphAttentionCNN` — operates on `(batch, nodes, features)` tensors for training efficiency.

---

## 👥 Team Contributions

| Contributor | Role |
|---|---|
| Swan | Data preprocessing, GNN/GAT model development, Flask backend |
| *(add teammates)* | *(add roles)* |

---

## 📈 Results

> Results and confusion matrix plots are available in the `/static/` directory after running the notebook.

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | - | - | - | - |
| Decision Tree | - | - | - | - |
| Propose GNN | - | - | - | - |
| Extension CNN2D | - | - | - | - |

*(Fill in your actual metrics from the notebook output)*

---

## 📄 License

This project is for academic/research purposes.
