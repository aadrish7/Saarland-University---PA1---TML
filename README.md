# Saarland-University---PA1---TML

This repository contains the implementation for **Programming Assignment 1 (PA1)** of the **Trustworthy Machine Learning (TML)** course at Saarland University.

## Project Overview

The core objective of this project is to implement a **Membership Inference Attack (MIA)**. The attack aims to determine whether a specific data point was part of the training set of a target machine learning model.

### Key Features

- **Feature Extraction**: Extracts multiple membership-sensitive features including:
  - Prediction loss (Negative NLL Loss)
  - Correct class probability and logit
  - Prediction margin (Top 1 - Top 2 probability)
  - Prediction entropy
  - Prediction correctness
  - Test-Time Augmentation (TTA) stability features (Mean/Std of probabilities across augmentations)
  - Modified entropy (Likelihood-ratio based)
- **RMIA-like Calibration**: Implements a calibration technique similar to **Referenced Membership Inference Attack (RMIA)**. It calibrates the target probabilities by the average probability observed on a reference (public) non-member dataset for each class.
- **Attack Model**: Uses a **Logistic Regression** classifier (implemented in PyTorch) as the attack model to distinguish between members and non-members based on the extracted features.
- **Model Blending**: Includes a logic to blend the base attack model output with specific raw features (like negative loss or entropy) to maximize the **True Positive Rate (TPR) at a low False Positive Rate (FPR)** (target 5% FPR).
- **Cross-Validation**: Uses Stratified K-Fold cross-validation on the public dataset to train and evaluate the attack model.

## Implementation Details

The main implementation is contained in `final_file.py`.

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Requests (for server submission)

### Usage

The script expects the following directory structure and files (configurable in the `Configuration` section of `final_file.py`):

```text
tml26_task1/
├── model.pt (Target model state dict)
├── pub.pt   (Public dataset with membership labels)
└── priv.pt  (Private dataset for which to predict membership)
```

To run the attack and generate a submission:

```bash
python final_file.py
```

The script will:
1. Load the target ResNet-18 model.
2. Extract features from both public and private datasets using TTA.
3. Apply RMIA-like calibration.
4. Train an attack model using cross-validation on the public dataset.
5. Select the best feature blend based on TPR@5%FPR.
6. Generate a `submission.csv` file.
7. Automatically attempt to submit the results to the evaluation server.

## License

This project is for educational purposes as part of the TML course at Saarland University.
