
# AI-Driven Flood Prediction Using Satellite Images 🌊🛰️

This project leverages Sentinel-1 SAR satellite images to develop a deep learning-based flood prediction model. The model uses CNNs (Convolutional Neural Networks) and is optimized with Genetic Algorithm (GA) to enhance performance.

## Project Structure
📂 sen12flood/ - Contains Sentinel-1 SAR data (VH, VV bands) and labels.

📂 models/ - Stores trained CNN models.

📂 processed_data/ - Contains preprocessed NumPy arrays of images.

📂 results/ - Stores model predictions and evaluations.

## Dataset

We use Sen12Flood dataset, which contains flood and non-flood images from SAR sensors. The dataset is preprocessed before training the model.

## Model Optimization
We apply GA for optimizing CNN hyperparameters to improve flood prediction accuracy.

## Results
📊 The trained model predicts flooded regions with high accuracy.

🔍 The results are stored in the results/ directory.
