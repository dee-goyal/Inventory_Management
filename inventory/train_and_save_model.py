import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('stock_depletion_data.csv')  # Update the path to your dataset

# Handle missing values or outliers if necessary
data.dropna(inplace=True)

# Calculate depletion rate, handling division by zero
data['Depletion Rate'] = data['Sales Volume'] / data['Stock Level'].replace(0, 1)

# Selecting relevant features
features = [
    'Stock Level', 'Stock-Out Occurrence', 'Stock-Out Duration', 'Promotion',
    'Seasonality', 'External Events', 'Product Price', 'Order Quantity'
]
target = 'Depletion Rate'

# Remove rows where the depletion rate is infinity or too large
data = data.replace([float('inf'), -float('inf')], float('nan')).dropna(subset=[target])

# One-hot encoding for categorical variables
categorical_features = ['Promotion', 'Seasonality', 'External Events']

# Splitting the data into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model_pipeline, 'depletion_rate_model.pkl')

# Get feature importances from the trained model
feature_importances = model_pipeline.named_steps['model'].feature_importances_

# Map feature importances to feature names
feature_names = (categorical_features +
                 [f for f in features if f not in categorical_features])
importances = sorted(zip(feature_importances, feature_names), reverse=True)

# Print feature importances
print("Feature Importances:")
for importance, name in importances:
    print(f"{name}: {importance}")
