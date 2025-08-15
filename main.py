import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
# Generate synthetic data representing feature usage and churn
num_users = 500
data = {
    'feature_a': np.random.randint(0, 10, num_users),  # Feature A usage
    'feature_b': np.random.randint(0, 5, num_users),   # Feature B usage
    'feature_c': np.random.randint(0, 15, num_users),  # Feature C usage
    'days_active': np.random.randint(1, 31, num_users), # Days active in the last month
    'churned': np.random.binomial(1, 0.2, num_users)   # Churn (1=churned, 0=not churned)
}
df = pd.DataFrame(data)
# --- 2. Data Analysis and Preprocessing ---
# No significant cleaning needed for synthetic data, but this section is crucial for real-world datasets.
# Feature Engineering (Example: Interaction between features)
df['feature_a_b'] = df['feature_a'] * df['feature_b']
# --- 3. Model Training ---
# Split data into training and testing sets
X = df[['feature_a', 'feature_b', 'feature_c', 'days_active', 'feature_a_b']]
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Logistic Regression model (you can experiment with other models)
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
# --- 4. Visualization ---
# Feature Importance (Example using coefficients from Logistic Regression)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Churn Prediction')
plt.xlabel('Importance (Coefficient)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
#Churn vs Days Active
plt.figure(figsize=(10,6))
sns.boxplot(x='churned', y='days_active', data=df)
plt.title('Days Active vs Churn')
plt.xlabel('Churned (0=No, 1=Yes)')
plt.ylabel('Days Active')
plt.tight_layout()
plt.savefig('churn_vs_days_active.png')
print("Plot saved to churn_vs_days_active.png")