# Panoramic Image Stitching

This project is a collection of tools and experiments for panoramic image stitching using computer vision and deep learning techniques.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

## Features

- **Image Stitching:** Utilizes the `ETH-CVG/lightglue_superpoint` model from Hugging Face for robust feature matching and `kornia` for homography estimation and image warping.
- **Batch Testing Framework:** Includes scripts to automate the testing of the stitching algorithm on a large set of images and evaluate the performance using the Structural Similarity Index (SSIM).
- **Interactive Web Application:** A `gradio` app to easily upload and stitch your own images.
- **Image Augmentation:** Tools for applying geometric and brightness jitter to images for testing the robustness of the stitching algorithm.

## Installation

To set up the project, install the required dependencies.

```bash
cd cv_project
pip install -r app/requirements.txt
```

The core dependencies are:

- `torch` & `torchvision`
- `kornia`
- `transformers`
- `opencv-python`
- `scikit-image`
- `pillow`
- `gradio`
- `numpy`
- `matplotlib`

## Usage

### Web Application

To run the Gradio web application for interactive image stitching:

```bash
cd app
python app.py
```

You can also try a live version of the application hosted on [Hugging Face Spaces](https://huggingface.co/spaces/PHarder/Automated_Panorama_Stitching).

### Batch Testing

The `batch_test` directory contains scripts to evaluate the stitching algorithm on a folder of example images. You can either change the directory of the script or load your images into the batch_test folder. The results, including SSIM scores and success rates, are saved to a JSON file.

## Directory Structure

- `app/`: Contains the Gradio web application for an interactive demonstration.
- `batch_test/`: Scripts and results for batch testing the stitching performance.
- `cv_utils/`: Core utility functions for stitching (`stitching.py`) and performance metrics (`metrics.py`).
- `image_augmentation/`: Scripts for augmenting images, such as `jitter.py`.
- `test_images/`: A collection of sample images for testing.

## License

This project is licensed under the [MIT License](LICENSE).
