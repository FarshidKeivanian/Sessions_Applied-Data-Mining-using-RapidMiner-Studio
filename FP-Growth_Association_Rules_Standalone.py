import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/FarshidKeivanian/Sessions_Applied-Data-Mining-using-RapidMiner-Studio/main/Chapter05DataSet.csv"
df = pd.read_csv(url)

# Convert relevant columns into categorical data (if necessary)
for col in df.columns:
    df[col] = df[col].astype(str)  # Ensuring all data is categorical

# Convert categorical values into one-hot encoding (Transaction format)
df_encoded = pd.get_dummies(df)

# Apply FP-Growth algorithm
min_support_value = 0.2  # Adjust based on dataset
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support_value, use_colnames=True)

# Generate association rules
min_confidence_value = 0.5  # Adjust as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_value)

# Save results to CSV
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)

# Display results
print("Frequent Itemsets:")
print(frequent_itemsets.head())

print("\n Association Rules:")
print(rules.head())
