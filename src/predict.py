
import hvplot


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas

# %matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 80)

df = pd.read_csv('/content/Employee Attrition.csv')
df.head()

df.info()

df.describe()

for column in df.columns:
    print(f"{column}: Number of unique values {df[column].nunique()}")

df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

df.info()

object_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 30:
        object_col.append(column)
        print(f"{column} : {df[column].unique()}")
        print(df[column].value_counts())
        print("====================================")
object_col.remove('Attrition')

len(object_col)

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
df["Attrition"] = label.fit_transform(df.Attrition)

disc_col = []
for column in df.columns:
    if df[column].dtypes != object and df[column].nunique() < 30:
        print(f"{column} : {df[column].unique()}")
        disc_col.append(column)
        print("====================================")
disc_col.remove('Attrition')

cont_col = []
for column in df.columns:
    if df[column].dtypes != object and df[column].nunique() > 30:
        print(f"{column} : Minimum: {df[column].min()}, Maximum: {df[column].max()}")
        cont_col.append(column)
        print("====================================")

df.hvplot.hist(y='DistanceFromHome', by='Attrition', subplots=False, width=600, height=300, bins=30)

df.hvplot.hist(y='Education', by='Attrition', subplots=False, width=600, height=300)

df.hvplot.hist(y='EnvironmentSatisfaction', by='Attrition', subplots=False, width=600, height=300)

df.hvplot.hist(y='RelationshipSatisfaction', by='Attrition', subplots=False, width=600, height=300)

df.hvplot.hist(y='Age', by='Attrition', subplots=False, width=600, height=300, bins=35)

# Convert categorical columns to numerical using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now calculate the correlation matrix and plot the heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(df_encoded.corr(), annot=True, cmap="RdYlGn", annot_kws={"size": 15})

# Convert categorical columns to numerical using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now calculate the correlation matrix and plot the heatmap
# Use the encoded DataFrame (df_encoded) instead of the original DataFrame (df)
col = df_encoded.corr().nlargest(20, "Attrition").Attrition.index
plt.figure(figsize=(15, 15))
sns.heatmap(df_encoded[col].corr(), annot=True, cmap="RdYlGn", annot_kws={"size":10})

# Convert categorical columns to numerical using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now calculate the correlation with the encoded DataFrame (df_encoded)
df_encoded.drop('Attrition', axis=1).corrwith(df_encoded.Attrition).hvplot.barh()

dummy_col = [column for column in df.drop('Attrition', axis=1).columns if df[column].nunique() < 20]
data = pd.get_dummies(df, columns=dummy_col, drop_first=True, dtype='uint8')
data.info()

print(data.shape)

# Remove duplicate Features
data = data.T.drop_duplicates()
data = data.T

# Remove Duplicate Rows
data.drop_duplicates(inplace=True)

print(data.shape)

data.shape

data.drop('Attrition', axis=1).corrwith(data.Attrition).sort_values().plot(kind='barh', figsize=(10, 30))

categorical_features = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = data.drop('Attrition', axis=1)
y = data['Attrition']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Import confusion_matrix

y_pred = log_model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred) # Now confusion_matrix is defined
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to take simplified user input and make a prediction
def predict_attrition_simple():
    print("Please provide the following employee details:")

    # Dictionary to store user inputs
    user_data = {}

    # Simple input for key features
    user_data['Age'] = int(input("Age: "))
    user_data['MonthlyIncome'] = int(input("Monthly Income: "))
    user_data['DistanceFromHome'] = int(input("Distance from Home: "))
    user_data['JobSatisfaction'] = int(input("Job Satisfaction (1-4): "))
    user_data['OverTime_Yes'] = int(input("Overtime (1 if Yes, 0 if No): "))

    # Set default values for other features
    defaults = {
        'DailyRate': 800,
        'Education': 3,
        'EnvironmentSatisfaction': 3,
        'HourlyRate': 50,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'NumCompaniesWorked': 1,
        'PercentSalaryHike': 12,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': 10,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': 3,
        'YearsAtCompany': 5,
        'YearsInCurrentRole': 2,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 2,
        'BusinessTravel_Travel_Frequently': 0,
        'BusinessTravel_Travel_Rarely': 1,
        'Department_Human Resources': 0,
        'Department_Research & Development': 1,
        'EducationField_Life Sciences': 1,
        'EducationField_Marketing': 0,
        'EducationField_Medical': 0,
        'EducationField_Other': 0,
        'EducationField_Technical Degree': 0,
        'Gender_Male': 1,
        'JobRole_Manager': 0,
        'JobRole_Research Director': 0,
        'JobRole_Sales Executive': 0,
        'MaritalStatus_Married': 0,
        'MaritalStatus_Single': 1
    }

    # Add default values to user data
    user_data.update(defaults)

    # Convert the user input into a DataFrame
    user_df = pd.DataFrame([user_data])

    # Ensure user_df has the same columns as X_train (adding missing columns as 0)
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # Scale the user input
    user_df = scaler.transform(user_df)

    # Predict using the trained model
    prediction = log_model.predict(user_df)
    result = "Yes" if prediction[0] == 1 else "No"
    print(f"\nPrediction: The employee is likely to leave (Attrition: {result})")

# Run the simplified prediction function
predict_attrition_simple()