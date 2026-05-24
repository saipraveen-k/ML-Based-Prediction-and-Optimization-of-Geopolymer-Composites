import pandas as pd

# Read the CSV file
df = pd.read_csv('data/dataset.csv')

print("Dataset Shape:", df.shape)
print("\nAll Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n\nFirst few rows:")
print(df.head())

print("\n\nData Types:")
print(df.dtypes)

print("\n\nMissing Values:")
print(df.isnull().sum())

print("\n\nTarget Variable (Compressive_Strength_MPa):")
print(df['Compressive_Strength_MPa'].describe())
