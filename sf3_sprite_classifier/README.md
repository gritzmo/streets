# SF3 Sprite Classifier

This project trains a simple CNN to classify Street Fighter III character sprite actions.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Add training images in `sprites/`, organized into subfolders named after the action labels (e.g. `idle`, `jump`).

## Training

Run the training script:

```bash
python train.py --data-dir sprites --epochs 10 --batch-size 32
```

After training, a `model.pth` file will be created with the trained weights and label mapping.

## Prediction

Use `predict.py` with a path to an image:

```bash
python predict.py path/to/image.png --model-path model.pth
```

It will output the predicted action label.
