# Text to Image Generation using Stable Diffusion and Dreambooth

This repository contains a Jupyter Notebook for generating images from text using Stable Diffusion and Dreambooth. The goal of this project is to leverage these advanced techniques to create high-quality images based on textual descriptions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebook Contents](#notebook-contents)
- [Results](#results)
## Overview

Stable Diffusion and Dreambooth are powerful tools for generating images from textual descriptions. Stable Diffusion uses a deep learning model trained on vast amounts of image-text pairs to generate images, while Dreambooth fine-tunes these models to generate highly personalized images. This project demonstrates how to use these tools to generate images based on given text inputs.

## Dataset

The dataset used in this project consists of image-text pairs required for fine-tuning the models. It is assumed that the user has access to appropriate datasets for training and fine-tuning.

## Requirements

To run this notebook, you need to have the following libraries and tools installed:

- pandas
- numpy
- matplotlib
- torch
- torchvision
- transformers
- diffusers
- Jupyter

You can install these libraries using pip:

```sh
pip install pandas numpy matplotlib torch torchvision transformers diffusers jupyter
```

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/ayushg212/Text_to_Image.git
    ```

2. **Navigate to the project directory:**

    ```sh
    cd Text_to_Image
    ```

3. **Install the required libraries:**

    Ensure you have the necessary libraries installed. You can install them using `pip`:

    ```sh
    pip install pandas numpy matplotlib torch torchvision transformers diffusers jupyter
    ```

4. **Launch Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

5. **Open the notebook:**

    In the Jupyter Notebook interface, open `Experiment_DreamBooth_Stable_Diffusion.ipynb`.

## Notebook Contents

1. **Introduction**
   - Overview of the project and objectives.

2. **Loading the Data**
   - Importing necessary libraries and loading the dataset.

3. **Data Preprocessing**
   - Preparing the data for model training and fine-tuning.

4. **Stable Diffusion Model Training**
   - Training the Stable Diffusion model using the preprocessed data.

5. **Dreambooth Fine-Tuning**
   - Fine-tuning the Stable Diffusion model with Dreambooth for personalized image generation.

6. **Text to Image Generation**
   - Generating images from textual descriptions using the trained and fine-tuned models.

7. **Evaluation**
   - Evaluating the generated images and discussing the quality and relevance to the text prompts.

8. **Conclusions**
   - Summary of findings, potential next steps, and improvements for future work.

## Results
The results include high-quality images generated from textual descriptions. The notebook demonstrates how to fine-tune and use Stable Diffusion and Dreambooth for personalized image generation.

   


