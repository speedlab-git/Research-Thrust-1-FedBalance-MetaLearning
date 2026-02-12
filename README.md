# Multi-Modal Federated Crime Intelligence System  
## Research Thrust 1 & Research Thrust 2

---

## Overview

This repository contains the implementation of our **Federated Multi-Modal Crime Intelligence Framework**, developed under two complementary research thrusts:

- **Research Thrust I (RT1): Federated Data Rebalancing And Meta-Learning**
- **Research Thrust II (RT2): Interdependent Crime Analytics Via Multi-Modal Fusion**

The overall goal is to build a **privacy-preserving, multi-source crime analysis system** that integrates:

- 🎥 Surveillance video frames  
- 🐦 Social media (Twitter) text  
- 📊 Crime-report spatiotemporal data  

All models are trained under a **Federated Learning (FL)** setting using ensuring decentralized training across clients.


## Contents

- [Installation](#installation)
- [Dataset](#dataset)
-Research Thrust 1  
-Research Thrust 2


## Installation

We recommend using **Anaconda** to manage dependencies.  
We use **Python 3.11** for training and evaluation.

```bash
conda create -n FedCrime python=3.10 -y
conda activate FedCrime
pip install -r requirements.txt
```



## Dataset 


# 🔹 Research Thrust 1  
## Federated Data Rebalancing and Meta-learning

### Objective

Train federated models that classify crime categories from:

1. **Video frames (UCF-Crime dataset[https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset])**

The models output:







We preprocess the crime‐reporting data by converting event‐level records into fixed temporal windows to estimate contextual crime risk rather than detect individual incidents. Specifically, all timestamps are floored into hourly windows and grouped by semantic location type (e.g., street, bar or tavern, apartment). For each (location, hour) window, we compute the total number of reported incidents, and derive temporal attributes such as hour of day, day of week, and month. To obtain supervision without requiring explicit “no‐crime” labels, we define relative risk labels based on historical density: for each location type, we compute a per‐location incident count threshold using a training‐only quantile (e.g., the top 30% of windows). Windows whose incident counts exceed this threshold are labeled high crime likelihood, while the remaining windows are labeled low crime likelihood. This formulation yields a binary classification task that captures spatiotemporal risk patterns from historical data and produces a probabilistic risk prior that can be fused with video‐ and text‐based crime predictions.