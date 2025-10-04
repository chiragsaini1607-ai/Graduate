import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class AdmissionPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def load_data(self, filepath='admission_data.csv'):
        return pd.read_csv(filepath)
    
    def preprocess_data(self, df):
        X = df.drop('Chance_of_Admit', axis=1)
        y = df['Chance_of_Admit']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        accuracy = r2 * 100
        return accuracy, mse, r2
    
    def save_model(self, filepath='admission_model.pkl'):
        joblib.dump(self.model, filepath)
    
    def predict_admission(self, gre, toefl, rating, sop, lor, cgpa, research):
        features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
        return self.predict(features)[0]

if __name__ == "__main__":
    predictor = AdmissionPredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    
    # Train model
    predictor.train(X_train, y_train)
    
    # Evaluate
    accuracy, mse, r2 = predictor.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.1f}%")
    print(f"R² Score: {r2:.3f}")
    print(f"MSE: {mse:.4f}")
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    chance = predictor.predict_admission(320, 110, 4, 4.5, 4.5, 8.5, 1)
    print(f"\nExample Prediction: {chance:.2f} ({chance*100:.1f}%)")
