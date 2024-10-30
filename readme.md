# LRA-SwinCB: Fire Anomaly Detection and Localization using Low-Rank Adaptation

![Graphical Abstract](./assets/graphical_abstract.jpg)

## Overview

This repository contains the implementation of **LRA-SwinCB**, a fire anomaly detection framework that integrates:

- **Swin Transformer**: A powerful backbone model for extracting deep features.
- **Classification Boost head (CB-head) Module**: For channel and spatial feature enhancement.
- **Low-Rank Adaptation (LRA)**: A parameter-efficient fine-tuning approach to improve generalizability.
- **Gradient Filtering Algorithm**: To precisely localize fire anomalies using backward propagation gradients.


