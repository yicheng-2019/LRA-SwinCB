# LRA-SwinCB: Fire Anomaly Detection and Localization using Low-Rank Adaptation

![Graphical Abstract](./assets/graphical_abstract.png)

## Overview

This repository contains the implementation of **LRA-SwinCB**, a fire anomaly detection framework that integrates:

- **Swin Transformer**: A powerful backbone model for extracting deep features.
- **Classification Boost head (CB-head) Module**: For channel and spatial feature enhancement.
- **Low-Rank Adaptation (LRA)**: A parameter-efficient fine-tuning approach to improve generalizability.
- **Gradient Filtering Algorithm**: To precisely localize fire anomalies using backward propagation gradients.

The proposed method achieves **99.79% accuracy** on public fire datasets and **99.58% accuracy** in real-world applications, demonstrating superior performance with reduced model complexity.

## Key Features

- **Parameter-Efficient Transfer Learning**: Uses LRA to enhance generalization while minimizing trainable parameters.
- **CB-head Module**: Improves feature reconstruction and emphasizes meaningful spatial and channel features.
- **Fire Anomaly Localization**: Gradient Filtering algorithm for accurate localization, providing **95%+ localization accuracy**.

## Installation

To get started, clone this repository and install dependencies:

```bash
git clone https://github.com/yicheng-2019/LRA-SwinCB.git
cd LRA-SwinCB
pip install -r requirements.txt
```

## Dataset

- We used a **Composite Open Dataset** comprising multiple public fire anomaly datasets.
- A **Real Fire Dataset** was also constructed from real-world industrial fire scenes.

Sample images are available in `./data/samples`.

## Usage

### Training

Train the model using:

```bash
python train.py --dataset_path ./data --epochs 30 --batch_size 32
```

### Inference

Run the model to detect and localize fire anomalies:

```bash
python inference.py --image_path ./test_images/sample.jpg
```

## Results

| Dataset            | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Composite Open     | 99.87%    | 99.74% | 99.80%   | 99.79%   |
| Real Fire          | 99.32%    | 99.77% | 99.55%   | 99.58%   |

## Visualization

The Gradient Filtering module produces heatmaps that help visualize fire anomaly regions. Below are some samples:

![Localization Example](./assets/localization_example.png)

## Citation

If you use this work, please cite:

```bibtex
@article{Qiu2024LRASwinCB,
  title={Fire Anomaly Detection based on Low-Rank Adaption Fine-Tuning and Localization using Gradient Filtering},
  author={Yicheng Qiu, Feng Sha, Li Niu, Guangyu Zhang},
  journal={Applied Soft Computing},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
