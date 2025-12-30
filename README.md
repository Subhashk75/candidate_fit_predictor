# Candidate Fit Score Predictor

A machine learning system that predicts how well a candidate matches a job description, providing a compatibility score (0-100%) and explanation of key matching factors.

## Features

- **Text Analysis**: Parses and analyzes resumes/CVs and job descriptions
- **Feature Extraction**: Extracts skills, experience, education, and other relevant features
- **Multiple ML Models**: Supports Random Forest, Gradient Boosting, Linear Regression, SVM, and Neural Networks
- **Explainable AI**: Provides feature importance and matching factor explanations
- **Web API**: RESTful API endpoint for predictions (Flask-based)
- **Visualizations**: Generates charts and graphs of matching factors
- **Batch Processing**: Supports multiple candidate-job predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd candidate-fit-predictor

2 .Install dependencies:
pip install -r requirements.txt

3. Download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')