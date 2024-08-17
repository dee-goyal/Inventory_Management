import pandas as pd
import joblib

# Load the dataset and model
data = pd.read_csv('inventory/stock_depletion_data.csv')
model = joblib.load('inventory/depletion_rate_model.pkl')  # Ensure the model path is correct

def predict_depletion_rate(store_id, product_id):
    # Debugging: Print input values
    print(f"Predicting for store_id: {store_id}, product_id: {product_id}")

    # Filter the dataset for the specific Store ID and Product ID
    filtered_data = data[(data['Store ID'] == store_id) & (data['Product ID'] == product_id)]
    
    if filtered_data.empty:
        return None, "No data found for the provided Store ID and Product ID."

    # Ensure the 'Stock Level' and other columns are included as required
    features = filtered_data[['Stock Level', 'Stock-Out Occurrence', 'Stock-Out Duration', 'Promotion',
                              'Seasonality', 'External Events', 'Product Price', 'Order Quantity']]

    # Prepare features for prediction using the same preprocessing as during training
    # Note: The model includes preprocessing steps, so you should provide raw features
    try:
        predicted_rate = model.predict(features)[0]
    except Exception as e:
        return None, f"Prediction failed: {e}"

    # Assume Stock Level is required for other calculations
    current_stock_level = filtered_data['Stock Level'].iloc[0]
    
    return current_stock_level, predicted_rate
