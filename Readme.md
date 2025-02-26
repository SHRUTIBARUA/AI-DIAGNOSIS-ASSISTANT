# Diagnostic Assistant

## Overview
This project is an AI-powered **Diagnostic Assistant** designed to analyze medical images (Chest X-rays) and assist healthcare professionals in diagnosing diseases accurately and efficiently.

## Features
- **Deep Learning-based Medical Image Analysis** (Trained on Chest X-ray dataset)
- **Flask Web Application** for user-friendly interaction
- **Automated Disease Prediction** (e.g., Pneumonia detection)
- **Model Training Script** to generate the `diagnostic_model.h5` file
- **Supports Image Upload and Prediction**

---

## Project Structure
```
project_folder/
│── app.py                  # Flask application (Main backend)
│── train_model.py          # Model training script
│── requirements.txt        # Dependencies list
│── README.md               # Project documentation
│── models/                 # Stores the trained model
│    ├── diagnostic_model.h5  (Generated model file)
│── dataset/                # Dataset folder (Must be provided by the user)
│    ├── train/
│    │   ├── NORMAL/        # Healthy X-rays
│    │   ├── PNEUMONIA/     # Pneumonia X-rays
│    ├── test/
│    │   ├── NORMAL/
│    │   ├── PNEUMONIA/
│    ├── val/               # Validation set (Optional)
│── templates/              # HTML templates for Flask
│── static/                 # Static files (CSS, JS)
```

---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed. Then, run:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
The dataset folder (`dataset/`) **must be provided by the user**. 
- You can download a public dataset of **Chest X-rays** (e.g., NIH, Kaggle, etc.)
- Organize it as shown in the **Project Structure** section.

### 3. Train the Model (Optional, if model is missing)
If `models/diagnostic_model.h5` is missing, generate it using:
```bash
python train_model.py
```

### 4. Run the Application
To start the Flask web app, run:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000/`

---

## Submission Requirements
✅ **Public GitHub Repository**: Ensure the repository contains all required files.
✅ **Final PDF Submission**: Include the **GitHub link** and necessary details.
✅ **Mention Dataset Requirement**: Since the dataset is not included, users need to provide it.
✅ **Ensure Model File Exists**: If `diagnostic_model.h5` is missing, users must train the model using `train_model.py`.

---

## License
This project is open-source and available under the MIT License.

---

## Contact
For questions, reach out via GitHub Issues.


