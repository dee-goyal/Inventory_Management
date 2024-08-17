from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv('warehouse/stock_depletion_data.csv')  # Adjust the path as needed
model = joblib.load('warehouse/stock_prediction_model.pkl')  # Adjust the path as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inventory', methods=['GET', 'POST'])
def inventory():
    # Inventory function implementation
    pass

@app.route('/warehouse', methods=['GET', 'POST'])
def warehouse():
    result = None

    if request.method == 'POST':
        store_id = request.form.get('store_id')
        product_id = request.form.get('product_id')

        if store_id and product_id:
            try:
                store_id = int(store_id)
                product_id = int(product_id)
                result = warehouse_function(store_id, product_id)
            except ValueError:
                result = "Invalid input: Store ID and Product ID should be integers."
        else:
            result = "Both Store ID and Product ID are required."

    return render_template('warehouse.html', result=result)

def warehouse_function(store_id, product_id):
    global data, model  # Ensure these are accessible

    # Check if data is loaded
    if data is None or data.empty:
        return "Data is not loaded or is empty."

    # Check if model is loaded
    if model is None:
        return "Model is not loaded."

    # Filter the dataset
    filtered_data = data[(data['Store ID'] == store_id) & (data['Product ID'] == product_id)]
    
    if filtered_data.empty:
        return "No data found for the provided Store ID and Product ID."

    # Ensure all the columns used in training are present
    required_columns = [
        'Product ID', 'Store ID', 'Sales Volume', 'Sales Revenue',
        'Stock Level', 'Stock-Out Occurrence', 'Promotion',
        'Product Price', 'Reorder Point', 'Lead Time'
    ]

    # Handle missing columns
    missing_columns = [col for col in required_columns if col not in filtered_data.columns]
    if missing_columns:
        return f"Missing columns: {', '.join(missing_columns)}"

    # Prepare features for prediction
    features = filtered_data[required_columns]
    
    try:
        # Predict using the loaded model
        prediction = model.predict(features)[0]
        return f"Predicted Order Quantity: {prediction}"
    except Exception as e:
        # Log the exception to debug later
        print(f"Prediction failed with exception: {e}")
        return f"Prediction failed: {e}"
