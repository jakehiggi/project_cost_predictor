import pandas as pd
import numpy as np
import os

np.random.seed(25)
n = 500

industries = ['Biotech', 'Medtech', 'Water', 'Heavy Industry']
industry_choices = np.random.choice(industries, n)

# Generate core features
df = pd.DataFrame({
    'Project_ID': [f'PRJ_{i}' for i in range(n)],
    'Industry': industry_choices,
    'Initial_Budget': np.random.randint(80000, 500000, n),
    'Project_Duration': np.random.randint(3, 24, n),
    'Sustainability_Score': np.random.randint(20, 100, n),
    'Client_Satisfaction': np.random.randint(1, 6, n),
    'Resource_Usage_Index': np.round(np.random.uniform(1, 10, n), 2)
})

# Simulate cost overruns based on sustainability and randomness
# Base cost = budget + random overrun (positive or negative)
random_noise = np.random.normal(0, 0.15, n)  # 15% noise
sustainability_factor = (100 - df['Sustainability_Score']) / 100  # higher = worse
industry_risk = df['Industry'].map({
    'Biotech': 0.05,
    'Medtech': 0.1,
    'Water': 0.2,
    'Heavy Industry': 0.25
})

# Cost overrun = budget * (random + sustainability + industry noise)
overrun = 1 + random_noise + sustainability_factor * 0.5 + industry_risk
df['Actual_Cost'] = df['Initial_Budget'] * overrun
df['Actual_Cost'] = np.round(df['Actual_Cost'].clip(lower=20000), 2)  # No negative costs

script = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script, 'project_data.csv')

print(f"Saving generated data to {csv_path}")
df.to_csv(csv_path, index=False)

