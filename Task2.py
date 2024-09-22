import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno
import numpy as np  # Import NumPy
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Load the dataset (Excel file)
df = pd.read_excel('side_effect_data 1.xlsx')

# Filling missing values using SimpleImputer

# For numerical columns ('Kilo', 'Boy'), use mean for 'Kilo' and median for 'Boy'
from sklearn.impute import SimpleImputer

num_imputer_mean = SimpleImputer(strategy='mean')
df['Kilo'] = num_imputer_mean.fit_transform(df[['Kilo']]).ravel()

num_imputer_median = SimpleImputer(strategy='median')
df['Boy'] = num_imputer_median.fit_transform(df[['Boy']]).ravel()

# Handle missing 'Cinsiyet' values based on proportional distribution of existing gender data
gender_distribution = df['Cinsiyet'].value_counts(normalize=True)
missing_gender_count = df['Cinsiyet'].isnull().sum()

# Randomly fill missing 'Cinsiyet' values based on existing distribution
df.loc[df['Cinsiyet'].isnull(), 'Cinsiyet'] = np.random.choice(
    gender_distribution.index,
    size=missing_gender_count,
    p=gender_distribution.values
)

# For other categorical columns ('Il', 'Kan Grubu'), use most frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df['Il'] = cat_imputer.fit_transform(df[['Il']]).ravel()
df['Kan Grubu'] = cat_imputer.fit_transform(df[['Kan Grubu']]).ravel()

# **Do not encode 'Cinsiyet'**; keep the original gender names
# If needed, you can encode other categorical variables

# Encode 'Cinsiyet' for correlation purposes but keep original for plotting
df['Cinsiyet_encoded'] = LabelEncoder().fit_transform(df['Cinsiyet'])

# Display first 5 rows and info to verify the changes
print(df.head())
print(df.info())

# Check for remaining missing values
print(df.isnull().sum())

# Proceed with your visualizations

# Visualize the missing data matrix
plt.figure("Missing Data Matrix", figsize=(16, 10))
msno.matrix(df, sparkline=False)
plt.xticks(rotation=45, fontsize=10)
plt.gca().yaxis.set_visible(False)
plt.tight_layout()
plt.show()

# Bar plot for percentage of missing data
plt.figure("Missing Data Percentage", figsize=(14, 8))
missing_data = df.isnull().mean() * 100
missing_data.plot(kind='bar', color='salmon')
plt.title('Missing Data Percentage')
plt.xlabel('Columns')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

# Age Calculation
df['Yas'] = datetime.now().year - pd.to_datetime(df['Dogum_Tarihi']).dt.year

# Visualizing Age Distribution
plt.figure("Age Distribution", figsize=(14, 8))
df['Yas'].hist(bins=20, color='#ffb6c1')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# Gender Distribution Visualization
plt.figure("Gender Distribution", figsize=(14, 8))
df['Cinsiyet'].value_counts().plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()

# Correlation Matrix (including 'Cinsiyet_encoded')
plt.figure("Correlation Matrix", figsize=(14, 8))

# 'Cinsiyet_encoded' is already in the numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Visualize the Side Effect Distribution
plt.figure("Side Effect Distribution", figsize=(14, 8))
df['Yan_Etki'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Side Effect')
plt.ylabel('Frequency')
plt.title('Side Effect Distribution')
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

# Shorten the drug names (keep only the first two words)
df['short_drug_name'] = df['Ilac_Adi'].apply(lambda x: ' '.join(str(x).split()[:2]))

# Visualize the Drug Distribution (shortened names)
plt.figure("Drug Distribution", figsize=(20, 10))
df['short_drug_name'].value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel('Drug Name (First Two Words)')
plt.ylabel('Frequency')
plt.title('Drug Distribution')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

# Visualize the relationship between Gender and Age
plt.figure("Gender and Age Distribution", figsize=(14, 8))
sns.boxplot(x='Cinsiyet', y='Yas', data=df)
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Gender and Age Distribution')
plt.tight_layout()
plt.show()
