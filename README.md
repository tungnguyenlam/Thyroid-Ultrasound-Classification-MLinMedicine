# Thyroid-Ultrasound-Classification-MLinMedicine
This is the repo for my final project on Machine Learning in Medicine course in USTH. The dataset is Thyroid Ultrasound Dataset from Kaggle

# Structure of the repo
```
Thyroid-Ultrasound-Classification-MLinMedicine/
├── cam/                        # Class Activation Maps outputs
│   ├── grad_cam/
│   └── hires_cam/
├── checkpoints/                # Saved model checkpoints per fold
├── config/
│   └── config.yaml             # Training hyperparameters & settings
├── data/                       # Dataset directory
│   ├── benign/
│   ├── malignant/
│   ├── download.py             # Script to download the dataset
│   └── split_info.txt          # Train/val/test split metadata
├── models/                     # Model architecture definitions
│   ├── densenet/
│   ├── efficientnet/
│   ├── res18/
│   ├── res50/
│   └── loss.py                 # Custom loss functions
├── notebooks/
│   └── 00_EDA.ipynb            # Exploratory Data Analysis
├── results/                    # Training logs, metrics, and plots
├── src/                        # Core source code
│   ├── dataloader.py           # Dataset & DataLoader definitions
│   ├── train.py                # Training & cross-validation pipeline
│   ├── test.py                 # Evaluation & TTA inference
│   └── utils.py                # Utility helpers
├── .gitignore
└── README.md
```