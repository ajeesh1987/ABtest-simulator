import pandas as pd
import numpy as np

def generate_users(n=100_000, seed=42):
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

def simulate_click(row):
    prob = 0.03
    if row['intent'] == 'High':
        prob += 0.10
    elif row['intent'] == 'Browsing':
        prob += 0.05
    if row['device'] == 'Mobile':
        prob += 0.01
    if row['archetype'] == 'Impulsive':
        prob += 0.04
    elif row['archetype'] == 'Careful':
        prob -= 0.01
    if row['variant'] == 'B':
        if row['trend_affinity'] == 'Trendy':
            prob += 0.05
        if row['device'] == 'Mobile':
            prob += 0.02
    return np.random.rand() < prob

if __name__ == "__main__":
    print("Generating users...")
    users = generate_users()
    users['variant'] = np.random.choice(['A', 'B'], size=len(users))
    print("Simulating clicks...")
    users['clicked'] = users.apply(simulate_click, axis=1)
    print("Saving training data...")
    users.to_csv("data/training_data.csv", index=False)
    print("✅ Done. File saved to data/training_data.csv")
