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
from sklearn.impute import SimpleImputer

num_imputer_mean = SimpleImputer(strategy='mean')
df['Kilo'] = num_imputer_mean.fit_transform(df[['Kilo']]).ravel()

num_imputer_median = SimpleImputer(strategy='median')
df['Boy'] = num_imputer_median.fit_transform(df[['Boy']]).ravel()

# Handle missing 'Cinsiyet' values based on proportional distribution of existing gender data
gender_distribution = df['Cinsiyet'].value_counts(normalize=True)
missing_gender_count = df['Cinsiyet'].isnull().sum()

df.loc[df['Cinsiyet'].isnull(), 'Cinsiyet'] = np.random.choice(
    gender_distribution.index,
    size=missing_gender_count,
    p=gender_distribution.values
)

# For other categorical columns ('Il', 'Kan Grubu'), use most frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df['Il'] = cat_imputer.fit_transform(df[['Il']]).ravel()
df['Kan Grubu'] = cat_imputer.fit_transform(df[['Kan Grubu']]).ravel()

# Encode 'Cinsiyet' for correlation purposes but keep original for plotting
df['Cinsiyet_encoded'] = LabelEncoder().fit_transform(df['Cinsiyet'])

# Calculate age from birthdate
df['Yas'] = datetime.now().year - pd.to_datetime(df['Dogum_Tarihi']).dt.year

# Create a figure to hold all the subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 10))  # Slightly smaller figure size

# Subplot 1: Missing Data Matrix
msno.matrix(df, sparkline=False, ax=axs[0, 0])
axs[0, 0].set_title("Missing Data Matrix")
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45, fontsize=8)
axs[0, 0].yaxis.set_visible(False)

# Subplot 2: Missing Data Percentage
missing_data = df.isnull().mean() * 100
missing_data.plot(kind='bar', color='salmon', ax=axs[0, 1])
axs[0, 1].set_title("Missing Data Percentage")
axs[0, 1].set_xlabel('Columns')
axs[0, 1].set_ylabel('Percentage (%)')
axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45, fontsize=8)

# Subplot 3: Age Distribution
df['Yas'].hist(bins=20, color='#ffb6c1', ax=axs[1, 0])
axs[1, 0].set_title("Age Distribution")
axs[1, 0].set_xlabel('Age')
axs[1, 0].set_ylabel('Frequency')

# Subplot 4: Gender Distribution
df['Cinsiyet'].value_counts().plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title("Gender Distribution")
axs[1, 1].set_xlabel('Gender')
axs[1, 1].set_ylabel('Frequency')

# Subplot 5: Correlation Matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=axs[2, 0])
axs[2, 0].set_title("Correlation Matrix")

# Subplot 6: Side Effect Distribution
df['Yan_Etki'].value_counts().plot(kind='bar', color='skyblue', ax=axs[2, 1])
axs[2, 1].set_title("Side Effect Distribution")
axs[2, 1].set_xlabel('Side Effect')
axs[2, 1].set_ylabel('Frequency')
axs[2, 1].set_xticklabels(axs[2, 1].get_xticklabels(), rotation=45, fontsize=8)

# Set the overall title and adjust the layout
fig.suptitle('Data Analysis Overview', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to reduce blank space
plt.show()
