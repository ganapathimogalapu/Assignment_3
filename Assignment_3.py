
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"C:\Ganapathi\UCM_Course_work\Neural_Networks&Deep_learning\Assignment_3\data.csv"
df = pd.read_csv(file_path)

# Show basic statistical description
stats_description = df.describe()

# Check for null values
null_values = df.isnull().sum()

# Replace null values with the mean
df.fillna(df.mean(), inplace=True)

# Select two columns and aggregate data
aggregated_data = df.agg({'Duration': ['min', 'max', 'count', 'mean'], 
                          'Calories': ['min', 'max', 'count', 'mean']})

# Filter rows with Calories between 500 and 1000
filtered_df_1 = df[(df['Calories'] >= 500) & (df['Calories'] <= 1000)]

# Filter rows with Calories > 500 and Pulse < 100
filtered_df_2 = df[(df['Calories'] > 500) & (df['Pulse'] < 100)]

# Create df_modified without "Maxpulse" column
df_modified = df.drop(columns=['Maxpulse'], errors='ignore')

# Delete "Maxpulse" column from main df
df.drop(columns=['Maxpulse'], inplace=True, errors='ignore')

# Convert Calories column to int
df['Calories'] = df['Calories'].astype(int)

# Create a scatter plot for Duration and Calories
plt.scatter(df['Duration'], df['Calories'])
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title('Scatter Plot of Duration vs Calories')
plt.show()

# Display results
stats_description, null_values, aggregated_data, filtered_df_1.head(), filtered_df_2.head(), df.head(), df_modified.head()
