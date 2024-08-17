from flask import Flask, render_template, request
from inventory.inventory import predict_depletion_rate
from warehouse.warehouse import warehouse_function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inventory', methods=['GET', 'POST'])
def inventory():
    current_stock_level = None
    depletion_rate = None
    monthly_depletion = None

    if request.method == 'POST':
        store_id = request.form.get('store_id')
        product_id = request.form.get('product_id')

        if store_id and product_id:
            try:
                store_id = int(store_id)
                product_id = int(product_id)

                current_stock_level, depletion_rate = predict_depletion_rate(store_id, product_id)

                if depletion_rate is None:
                    depletion_rate = "No data found for the provided Store ID and Product ID."
                else:
                    monthly_depletion = depletion_rate * 30

            except ValueError:
                depletion_rate = "Invalid input: Store ID and Product ID should be integers."
        else:
            depletion_rate = "Both Store ID and Product ID are required."

    return render_template('inventory.html', 
                           current_stock_level=current_stock_level,
                           depletion_rate=depletion_rate,
                           monthly_depletion=monthly_depletion)

@app.route('/warehouse', methods=['GET', 'POST'])
def warehouse():
    result = None

    if request.method == 'POST':
        store_id = request.form.get('store_id')
        product_id = request.form.get('product_id')

        # Debugging output
        print(f"Received Store ID: {store_id}, Product ID: {product_id}")

        if store_id and product_id:
            try:
                store_id = int(store_id)
                product_id = int(product_id)
                result = warehouse_function(store_id, product_id)
                print(f"Result from warehouse_function: {result}")  # Debugging output
            except ValueError:
                result = "Invalid input: Store ID and Product ID should be integers."
                print(result)  # Debugging output
        else:
            result = "Both Store ID and Product ID are required."
            print(result)  # Debugging output

    return render_template('warehouse.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
