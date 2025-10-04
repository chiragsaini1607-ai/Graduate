import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from admission_predictor import AdmissionPredictor

def evaluate_model():
    predictor = AdmissionPredictor()
    df = predictor.load_data()
    X = df.drop('Chance_of_Admit', axis=1)
    y = df['Chance_of_Admit']
    
    # Cross-validation
    cv_scores = cross_val_score(predictor.model, X, y, cv=5, scoring='r2')
    
    # Train model for feature importance
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    predictor.train(X_train, y_train)
    
    # Feature importance
    feature_names = X.columns
    importances = predictor.model.feature_importances_
    
    # Results
    print("Cross-Validation Results:")
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.3f} ({cv_scores.mean()*100:.1f}%)")
    print(f"Std CV Score: {cv_scores.std():.3f}")
    
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}")
    
    # Predictions vs Actual plot
    predictions = predictor.predict(X_test)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    
    plt.subplot(1, 2, 2)
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
