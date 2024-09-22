import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno

# Load the dataset (Excel file)
df = pd.read_excel('side_effect_data 1.xlsx')

# Calculate age from birthdate
df['age'] = datetime.now().year - pd.to_datetime(df['Dogum_Tarihi']).dt.year

# Shorten the drug names (keep only the first two words)
df['short_drug_name'] = df['Ilac_Adi'].apply(lambda x: ' '.join(x.split()[:2]))

# Ensure correct numerical columns are selected
numeric_df = df.select_dtypes(include=['float64', 'int64', 'datetime64'])

# Set up a figure with multiple subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # Adjust the figure size to fit all graphs
fig.suptitle('Side Effect Data Analysis', fontsize=18)

# Missing Data Matrix
msno.matrix(df, sparkline=False, ax=axes[0, 0])
axes[0, 0].set_title('Missing Data Matrix', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=90, labelsize=6)

# Missing Data Percentage
missing_data = df.isnull().mean() * 100
missing_data.plot(kind='bar', color='salmon', ax=axes[0, 1])
axes[0, 1].set_title('Missing Data Percentage', fontsize=10)
axes[0, 1].set_xlabel('Columns', fontsize=8)
axes[0, 1].set_ylabel('Percentage (%)', fontsize=8)
axes[0, 1].tick_params(axis='x', rotation=90, labelsize=6)

# Age Distribution
df['age'].hist(bins=20, color='#ffb6c1', ax=axes[0, 2])
axes[0, 2].set_title('Age Distribution', fontsize=10)
axes[0, 2].set_xlabel('Age', fontsize=8)
axes[0, 2].set_ylabel('Frequency', fontsize=8)

# Gender Distribution
df['Cinsiyet'].value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Gender Distribution', fontsize=10)
axes[1, 0].set_xlabel('Gender', fontsize=8)
axes[1, 0].set_ylabel('Frequency', fontsize=8)

# Correlation Matrix
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix', fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=90, labelsize=6)
axes[1, 1].tick_params(axis='y', rotation=0, labelsize=6)

# Side Effect Distribution
df['Yan_Etki'].value_counts().plot(kind='bar', color='skyblue', ax=axes[1, 2])
axes[1, 2].set_title('Side Effect Distribution', fontsize=10)
axes[1, 2].set_xlabel('Side Effect', fontsize=8)
axes[1, 2].set_ylabel('Frequency', fontsize=8)
axes[1, 2].tick_params(axis='x', rotation=90, labelsize=6)

# Drug Distribution (shortened names) with smaller text
df['short_drug_name'].value_counts().plot(kind='bar', color='lightgreen', ax=axes[2, 0])
axes[2, 0].set_title('Drug Distribution', fontsize=10)
axes[2, 0].set_xlabel('Drug Name (First Two Words)', fontsize=8)
axes[2, 0].set_ylabel('Frequency', fontsize=8)
axes[2, 0].tick_params(axis='x', rotation=90, labelsize=5)  # Reduce font size for labels

# Gender and Age Distribution
sns.boxplot(x='Cinsiyet', y='age', data=df, ax=axes[2, 1])
axes[2, 1].set_title('Gender and Age Distribution', fontsize=10)
axes[2, 1].set_xlabel('Cinsiyet', fontsize=8)
axes[2, 1].set_ylabel('Age', fontsize=8)

# Hide the unused subplot
axes[2, 2].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusting rect to leave space for the main title
plt.show()
