import pandas as pd
import numpy as np

np.random.seed(42)

def generate_admission_data(n_samples=1000):
    data = {
        'GRE_Score': np.random.randint(290, 341, n_samples),
        'TOEFL_Score': np.random.randint(92, 121, n_samples),
        'University_Rating': np.random.randint(1, 6, n_samples),
        'SOP': np.random.uniform(1, 5, n_samples),
        'LOR': np.random.uniform(1, 5, n_samples),
        'CGPA': np.random.uniform(6.8, 9.92, n_samples),
        'Research': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic admission chances based on features
    df['Chance_of_Admit'] = (
        0.002 * df['GRE_Score'] +
        0.003 * df['TOEFL_Score'] +
        0.06 * df['University_Rating'] +
        0.016 * df['SOP'] +
        0.017 * df['LOR'] +
        0.12 * df['CGPA'] +
        0.02 * df['Research'] -
        1.27
    )
    
    df['Chance_of_Admit'] = np.clip(df['Chance_of_Admit'], 0, 1)
    
    return df

if __name__ == "__main__":
    data = generate_admission_data()
    data.to_csv('admission_data.csv', index=False)
    print(f"Generated {len(data)} samples")
    print(data.head())
