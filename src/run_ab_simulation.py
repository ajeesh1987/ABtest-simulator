import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('models/behavior_model.pkl')

# Step 1: Generate new users
def generate_users(n=1_000_000, seed=123):
    np.random.seed(seed)
    users = pd.DataFrame({
        'user_id': range(n),
        'age_group': np.random.choice(['Gen Z', 'Millennial', 'Gen X', 'Boomer'], n, p=[0.25, 0.35, 0.25, 0.15]),
        'region': np.random.choice(['EU', 'US', 'Asia', 'Other'], n),
        'device': np.random.choice(['Mobile', 'Mobile Browser', 'Desktop'], n, p=[0.6, 0.1, 0.3]),
        'tech_savvy': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.5, 0.2]),
        'intent': np.random.choice(['Low', 'Browsing', 'High'], n, p=[0.4, 0.4, 0.2]),
        'trend_affinity': np.random.choice(['Trendy', 'Neutral', 'Price Sensitive'], n, p=[0.3, 0.4, 0.3]),
        'archetype': np.random.choice(['Impulsive', 'Loyal', 'Careful'], n, p=[0.2, 0.3, 0.5])
    })
    users['variant'] = np.random.choice(['A', 'B'], size=n)
    return users

# Step 2: Predict clicks using ML model
def predict_clicks(users):
    X = pd.get_dummies(users.drop(columns=['user_id']))  # one-hot encode
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)  # align with training
    click_probs = model.predict_proba(X)[:, 1]
    users['click_probability'] = click_probs
    users['clicked'] = np.random.rand(len(users)) < click_probs
    return users

if __name__ == "__main__":
    print("Generating users...")
    df = generate_users()

    print("Simulating A/B test with trained model...")
    df = predict_clicks(df)

    print("Saving results...")
    df.to_csv("data/ab_simulation_results.csv", index=False)
    print("✅ Simulation complete. File saved to data/ab_simulation_results.csv")
