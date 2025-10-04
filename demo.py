#!/usr/bin/env python3
"""
Graduate Admission Prediction Demo
Run this script to see the model in action
"""

import joblib
import numpy as np

def load_model():
    try:
        return joblib.load('admission_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run admission_predictor.py first.")
        return None

def predict_admission():
    model = load_model()
    if not model:
        return
    
    print("=== Graduate Admission Predictor ===")
    print("Enter student details:")
    
    try:
        gre = int(input("GRE Score (290-340): "))
        toefl = int(input("TOEFL Score (92-120): "))
        rating = int(input("University Rating (1-5): "))
        sop = float(input("SOP Strength (1-5): "))
        lor = float(input("LOR Strength (1-5): "))
        cgpa = float(input("CGPA (6.8-9.92): "))
        research = int(input("Research Experience (0/1): "))
        
        features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
        prediction = model.predict(features)[0]
        
        print(f"\nPredicted Admission Chance: {prediction:.3f} ({prediction*100:.1f}%)")
        
        if prediction > 0.8:
            print("Status: High chance of admission! 🎉")
        elif prediction > 0.6:
            print("Status: Good chance of admission 👍")
        elif prediction > 0.4:
            print("Status: Moderate chance of admission 🤔")
        else:
            print("Status: Low chance of admission 😔")
            
    except ValueError:
        print("Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    predict_admission()
