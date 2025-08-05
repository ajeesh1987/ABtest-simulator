import pandas as pd
import numpy as np

def generate_users(n=1_000_000, seed=42):
    np.random.seed(seed)

    users = pd.DataFrame({
        'user_id': range(n),
        'age_group': np.random.choice(['Gen Z', 'Millennial', 'Gen X', 'Boomer'], n, p=[0.25, 0.35, 0.25, 0.15]),
        'region': np.random.choice(['EU', 'US', 'Asia', 'Other'], n),
        'device': np.random.choice(['Mobile', 'Desktop'], n, p=[0.7, 0.3]),
        'tech_savvy': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.3, 0.5, 0.2]),
        'intent': np.random.choice(['Low', 'Browsing', 'High'], n, p=[0.4, 0.4, 0.2]),
        'trend_affinity': np.random.choice(['Trendy', 'Neutral', 'Price Sensitive'], n, p=[0.3, 0.4, 0.3]),
        'archetype': np.random.choice(['Impulsive', 'Loyal', 'Careful'], n, p=[0.2, 0.3, 0.5])
    })

    return users

if __name__ == "__main__":
    df = generate_users()
    print(df.head())
    df.to_csv("data/simulated_users.csv", index=False)
    print("Saved to data/simulated_users.csv")
