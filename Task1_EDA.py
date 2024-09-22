import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno

# Load the dataset (Excel file)
df = pd.read_excel('side_effect_data 1.xlsx')

# Display the first 5 rows
print(df.head())

# Display information about the data types and columns
print(df.info())

# Check for missing values (corrected function)
print(df.isnull().sum())

# Visualize the missing data matrix
plt.figure("Missing Data Matrix", figsize=(16, 10))  # Adjust the figure size
msno.matrix(df, sparkline=False)
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and reduce font size
plt.gca().yaxis.set_visible(False)  # Hide the left y-axis
plt.subplots_adjust(left=0.05, right=0.95)  # Adjust left and right margins
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Missing Data Matrix')
plt.show()

# Visualize the percentage of missing data
plt.figure("Missing Data Percentage", figsize=(14, 8))  # Set figure title and size
missing_data = df.isnull().mean() * 100
missing_data.plot(kind='bar', color='salmon')
plt.title('Missing Data Percentage')
plt.xlabel('Columns')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and reduce font size
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Missing Data Percentage')
plt.show()

# Calculate age from birthdate
df['age'] = datetime.now().year - pd.to_datetime(df['Dogum_Tarihi']).dt.year

# Visualize the age distribution
plt.figure("Age Distribution", figsize=(14, 8))  # Set figure title and size
df['age'].hist(bins=20, color='#ffb6c1')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Age Distribution')
plt.show()

# Visualize the gender distribution
plt.figure("Gender Distribution", figsize=(14, 8))  # Set figure title and size
df['Cinsiyet'].value_counts().plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Gender Distribution')
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Gender Distribution')
plt.show()

# Visualize the correlation matrix (numeric columns only)
plt.figure("Correlation Matrix", figsize=(14, 8))  # Set figure title and size
numeric_df = df.select_dtypes(include=['float64', 'int64', 'datetime64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Correlation Matrix')
plt.show()

# Visualize the side effect distribution
plt.figure("Side Effect Distribution", figsize=(14, 8))  # Set figure title and size
df['Yan_Etki'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Side Effect')
plt.ylabel('Frequency')
plt.title('Side Effect Distribution')
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels and reduce font size
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Side Effect Distribution')
plt.show()

# Shorten the drug names (keep only the first two words)
df['short_drug_name'] = df['Ilac_Adi'].apply(lambda x: ' '.join(x.split()[:2]))

# Visualize the drug distribution (shortened names)
plt.figure("Drug Distribution", figsize=(20, 10))  # Set figure title and size
df['short_drug_name'].value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel('Drug Name (First Two Words)')
plt.ylabel('Frequency')
plt.title('Drug Distribution')
plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels and reduce font size
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Drug Distribution')
plt.show()

# Visualize the relationship between gender and age
plt.figure("Gender and Age Distribution", figsize=(14, 8))  # Set figure title and size
sns.boxplot(x='Cinsiyet', y='age', data=df)
plt.title('Gender and Age Distribution')
plt.tight_layout()  # Avoid label overlapping
plt.get_current_fig_manager().set_window_title('Gender and Age Distribution')
plt.show()
