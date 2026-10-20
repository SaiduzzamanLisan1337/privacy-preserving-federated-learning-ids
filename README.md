# Privacy-Preserving Federated Intrusion Detection System (FL-IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Opacus DP-SGD](https://img.shields.io/badge/Opacus-DP--SGD-red.svg)](https://opacus.ai/)
[![Dataset: NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-green.svg)](https://www.unb.ca/cic/datasets/nsl.html)

A PyTorch implementation of a Privacy-Preserving Network Intrusion Detection System (IDS) using **Federated Averaging (FedAvg)**, **Differential Privacy (DP-SGD via Meta Opacus)**, and **Dirichlet Non-IID Data Partitioning**. 

This framework simulates realistic decentralized edge clients with heterogeneous, non-IID network traffic while enforcing rigorous $(\epsilon, \delta)$-Differential Privacy guarantees to prevent model inversion and data reconstruction attacks.

---

## Key Features

* **Federated Learning Workflow:** Implements multi-client local training with centralized FedAvg aggregation without transmitting raw packet features or network payloads.
* **$(\epsilon, \delta)$-Differential Privacy (DP-SGD):** Integrated with Meta's `opacus` library to perform per-sample gradient clipping (`max_grad_norm = 1.2`) and Gaussian noise injection during client training.
* **Dirichlet Non-IID Data Heterogeneity:** Simulates real-world network distribution skew across clients using a Dirichlet distribution $\text{Dir}(\alpha)$, allowing tunable data imbalance across edge subnets.
* **Centralized vs. FL Benchmark:** Built-in baseline module to evaluate accuracy, privacy trade-offs, and convergence speed against a standard centralized IDS model.

---

## Project Structure

```text
federated-learning-ids/
├── data/
│   └── nsl-kdd/
│       └── KDDTrain+.txt     # NSL-KDD Training Dataset
├── src/
│   ├── dataset.py            # Preprocessing, standard scaling & Dirichlet non-IID partitioning
│   ├── model.py              # PyTorch Deep Neural Network (DNN) IDS Classifier
│   ├── client.py             # Differential Private Client (DP-SGD with Opacus)
│   ├── server.py             # Server-side FedAvg Aggregation & Evaluation
│   └── main_fl.py            # Primary execution pipeline & benchmark runner
├── requirements.txt
└── README.md
```

---

## Experimental Benchmarks ($lpha = 0.5$)

### 1. Dirichlet Non-IID Data Distribution Across Clients

Under a concentration parameter of $\alpha = 0.5$, network traffic is heavily skewed across edge subnets to simulate specialized domain networks (e.g., IoT subnets vs. web gateways):

| Client ID | Total Samples | Normal Packets | Attack Packets | Class Distribution |
| :--- | :--- | :--- | :--- | :--- |
| **Client 0** | 9,013 | 41 | 8,972 | **99.5% Attack** / 0.5% Normal |
| **Client 1** | 39,982 | 39,785 | 197 | **99.5% Normal** / 0.5% Attack |
| **Client 2** | 51,783 | 14,095 | 37,688 | **72.8% Attack** / 27.2% Normal |

### 2. Convergence & Accuracy Comparison

* **Centralized Baseline Accuracy:** `99.02%` (Trained on combined data)
* **Privacy Parameters:** Target $\epsilon = 10.00$, $\delta = 1\times 10^{-5}$, Max Gradient Norm $= 1.2$

| Communication Round | Global FL Model Accuracy | Privacy Budget Expended |
| :--- | :--- | :--- |
| **Round 1 / 5** | 92.28% | $\epsilon = 10.00, \delta = 1\times 10^{-5}$ |
| **Round 2 / 5** | 95.57% | $\epsilon = 10.00, \delta = 1\times 10^{-5}$ |
| **Round 3 / 5** | 96.56% | $\epsilon = 10.00, \delta = 1\times 10^{-5}$ |
| **Round 4 / 5** | 97.06% | $\epsilon = 10.00, \delta = 1\times 10^{-5}$ |
| **Round 5 / 5** | **97.88%** | $\epsilon = 10.00, \delta = 1\times 10^{-5}$ |

> **Result Summary:** Despite severe non-IID label skew ($\alpha = 0.5$) and strict local differential privacy constraints, the federated global model recovers to within **1.14%** of the centralized non-private baseline by Round 5.

---

## Quick Start

### 1. Prerequisites & Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/SaiduzzamanLisan1337/federated-learning-ids.git
cd federated-learning-ids

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place the NSL-KDD dataset (`KDDTrain+.txt`) inside `data/nsl-kdd/`:

```bash
mkdir -p data/nsl-kdd
# Move or copy KDDTrain+.txt into data/nsl-kdd/
```

*(Note: If no dataset is present, the pipeline automatically synthesizes dummy structural data to verify code integrity).*

### 3. Execution

Run the complete pipeline from the project root directory:

```bash
python -m src.main_fl
```

---

## Hyperparameter Configuration

Key parameters can be adjusted directly in `src/main_fl.py`:

```python
NUM_CLIENTS = 3          # Number of simulated federated clients
COMMUNICATION_ROUNDS = 5 # Number of global aggregation rounds
LOCAL_EPOCHS = 2         # Local client epochs per round
ALPHA = 0.5              # Dirichlet concentration parameter (lower = higher skew)
TARGET_EPSILON = 10.0    # Local DP Epsilon budget
TARGET_DELTA = 1e-5      # Local DP Delta bound
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
