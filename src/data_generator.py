import pandas as pd
import numpy as np

np.random.seed(42)

# Number of batches
num_batches = 500

data = {
    "Batch_ID": range(1, num_batches + 1),
    "Temperature": np.random.normal(180, 10, num_batches),
    "Pressure": np.random.normal(30, 5, num_batches),
    "pH": np.random.normal(7, 0.5, num_batches),
    "Mixing_Speed": np.random.normal(1200, 100, num_batches),
    "Energy_Used": np.random.normal(500, 50, num_batches),
}

df = pd.DataFrame(data)

# Create quality score based on closeness to optimal conditions
df["Quality_Score"] = (
    100
    - abs(df["Temperature"] - 180)
    - abs(df["Pressure"] - 30)
    - abs(df["pH"] - 7) * 5
)

df.to_csv("data/batches.csv", index=False)

print("Dataset generated successfully!")