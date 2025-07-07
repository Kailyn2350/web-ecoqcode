# ECOQCODE OCR Project

This project focuses on developing an Optical Character Recognition (OCR) system specifically designed to detect the presence of 'ECOQCODE' text within images. It utilizes a Convolutional Neural Network (CNN) for binary classification, determining whether an image contains the specified text or not.

## Features

-   **Synthetic Data Generation**: Generates a synthetic dataset of images with and without 'ECOQCODE' text, using various backgrounds and font sizes.
-   **CNN Model**: Implements a custom CNN architecture for image classification.
-   **Training and Validation**: Provides a training pipeline with data loading, transformation, and validation.
-   **Model Evaluation**: Evaluates the trained model's performance using classification metrics.
-   **ONNX Export**: Exports the trained PyTorch model to ONNX format for deployment and inference with ONNX Runtime.

## Project Structure

```
ECOQCODE_OCR/
├── CNN.ipynb               # Jupyter Notebook for data generation, model training, and ONNX export
├── eco_dataset/            # Directory for generated images and labels
│   ├── images/             # Generated image files
│   ├── labels.txt          # All generated image paths and their corresponding labels
│   ├── train_labels.txt    # Training set labels
│   └── val_labels.txt      # Validation set labels
├── backgrounds/            # Directory for background images used in data generation
└── real_backgrounds/       # Placeholder for real-world background images (optional)
```

## Setup

### Prerequisites

-   Python 3.x
-   Jupyter Notebook
-   `arial.ttf` font file (or specify another font in `CNN.ipynb`)

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd ECOQCODE_OCR
    ```

2.  **Install required Python packages:**
    ```bash
    pip install torch torchvision scikit-learn pillow tqdm jupyter
    ```

3.  **Prepare background images:**
    Place various background images (e.g., `.jpg`, `.png`) into the `backgrounds/` directory. These will be used to generate synthetic data.

## Usage

All steps are contained within the `CNN.ipynb` Jupyter Notebook.

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook CNN.ipynb
    ```

2.  **Run Cells Sequentially:**
    Execute the cells in the notebook from top to bottom.

    -   **Data Generation**: The first code cell will generate synthetic images and their labels, saving them to `eco_dataset/images/` and `eco_dataset/labels.txt`.
    -   **Dataset and Model Definition**: Subsequent cells define the custom dataset class, image transformations, and the CNN model architecture.
    -   **Data Splitting**: The notebook splits the generated labels into training and validation sets (`train_labels.txt` and `val_labels.txt`).
    -   **Training**: The training loop will train the CNN model on the generated dataset.
    -   **Evaluation**: After training, the model's performance will be evaluated on the validation set, and a classification report will be printed.
    -   **ONNX Export**: The final cell exports the trained PyTorch model to ONNX format, saving it as `ecoq_classifier.onnx` in the project root directory.

## ONNX Model Export

The trained `ECOQClassifier` model is exported to the ONNX format (`ecoq_classifier.onnx`). This allows for easy deployment and inference across various platforms and runtimes that support ONNX, such as ONNX Runtime.

The export process includes:
-   **`dummy_input`**: A dummy tensor is created to trace the model's computation graph. It assumes an input image size of `(1, 3, 224, 224)` (batch size 1, 3 color channels, 224x224 pixels).
-   **`opset_version=11`**: Specifies the ONNX operator set version for compatibility.
-   **`dynamic_axes`**: Configures dynamic batch sizing, allowing the ONNX model to accept inputs with varying batch sizes during inference.

## Future Improvements

-   Integrate real-world background images from `real_backgrounds/` for more robust data generation.
-   Explore more complex CNN architectures or pre-trained models for improved accuracy.
-   Implement a dedicated inference script using the exported ONNX model.
-   Add more sophisticated data augmentation techniques during training.
