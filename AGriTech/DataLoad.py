import pandas as pd

df = pd.read_csv("dataset/Crop_recommendation.csv")
print(df.head())
print(df.shape)

print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())       
# df.dropna(inplace=True) If any column has missing values, we can drop them using this line
df.drop_duplicates(inplace=True)
print(df.shape)

df.to_csv("dataset/crop_recommendation_cleaned.csv", index=False)
