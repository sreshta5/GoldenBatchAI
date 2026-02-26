import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("data/batches.csv")

print("Total Batches:", len(df))


# ===============================
# IDENTIFY GOLDEN BATCHES
# Golden = High Quality + Low Energy
# ===============================

golden_df = df[
    (df["Quality_Score"] > df["Quality_Score"].quantile(0.75)) &
    (df["Energy_Used"] < df["Energy_Used"].quantile(0.50))
]

print("Total Golden Batches Identified:", len(golden_df))


# ===============================
# CLUSTER HIGH PERFORMING BATCHES
# ===============================

features = golden_df[["Temperature", "Pressure", "pH", "Mixing_Speed"]]

kmeans = KMeans(n_clusters=3, random_state=42)
golden_df["Cluster"] = kmeans.fit_predict(features)

print("\nCluster Distribution:")
print(golden_df["Cluster"].value_counts())


# ===============================
# IDENTIFY BEST CLUSTER
# ===============================

best_cluster = golden_df["Cluster"].value_counts().idxmax()
print("\nBest Performing Cluster:", best_cluster)

best_cluster_data = golden_df[golden_df["Cluster"] == best_cluster]


# ===============================
# CREATE GOLDEN SIGNATURE
# ===============================

golden_signature_mean = best_cluster_data[
    ["Temperature", "Pressure", "pH", "Mixing_Speed"]
].mean()

golden_signature_std = best_cluster_data[
    ["Temperature", "Pressure", "pH", "Mixing_Speed"]
].std()

print("\nGolden Signature (Ideal Mean Parameters):")
print(golden_signature_mean)

print("\nGolden Signature (Allowed Variation - Std):")
print(golden_signature_std)


# ===============================
# SAVE RESULTS
# ===============================

golden_df.to_csv("data/golden_batches.csv", index=False)

signature_df = pd.DataFrame({
    "Mean": golden_signature_mean,
    "Std": golden_signature_std
})

signature_df.to_csv("data/golden_signature.csv")

print("\nGolden batch analysis complete!")