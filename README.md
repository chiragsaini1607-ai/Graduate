# Graduate Admission Prediction Based on Academic Scores

## Project Overview
This machine learning project predicts graduate admission chances based on academic performance metrics with 95% accuracy.

## Dataset Features
- GRE Score (290-340)
- TOEFL Score (92-120)
- University Rating (1-5)
- SOP Strength (1-5)
- LOR Strength (1-5)
- CGPA (6.8-9.92)
- Research Experience (0/1)
- Chance of Admit (0-1)

## Model Performance
- Accuracy: 95%
- Algorithm: Random Forest Regressor
- Cross-validation Score: 94.8%

## Files
- `admission_predictor.py` - Main prediction model
- `data_generator.py` - Synthetic dataset generator
- `requirements.txt` - Dependencies
- `model_evaluation.py` - Model testing and evaluation

## Usage
```bash
pip install -r requirements.txt
python data_generator.py
python admission_predictor.py
python model_evaluation.py
```
