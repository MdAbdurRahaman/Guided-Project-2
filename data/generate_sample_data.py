import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ecommerce_data(num_records=1200, output_path="data/raw_data.csv"):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 150 distinct customers
    customer_ids = np.random.choice(range(10001, 10151), size=num_records)
    
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=int(np.random.exponential(scale=60))) for _ in range(num_records)]
    # cap dates to today
    dates = [min(d, datetime(2026, 6, 1)) for d in dates]
    
    stock_codes = [f"ITEM_{np.random.randint(100, 150)}" for _ in range(num_records)]
    descriptions = [f"Product Description {code}" for code in stock_codes]
    quantities = np.random.randint(1, 25, size=num_records)
    unit_prices = np.round(np.random.uniform(2.5, 250.0, size=num_records), 2)
    invoice_numbers = [f"INV{50000 + i // 3}" for i in range(num_records)]
    countries = np.random.choice(["United Kingdom", "Germany", "France", "USA", "Spain"], size=num_records, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    
    # Inject some noise/cancellations
    for idx in range(0, num_records, 40):
        quantities[idx] = -1 * quantities[idx]
        invoice_numbers[idx] = f"C{invoice_numbers[idx]}"
        
    df = pd.DataFrame({
        "InvoiceNo": invoice_numbers,
        "StockCode": stock_codes,
        "Description": descriptions,
        "Quantity": quantities,
        "InvoiceDate": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        "UnitPrice": unit_prices,
        "CustomerID": customer_ids,
        "Country": countries
    })
    
    df.to_csv(output_path, index=False)
    print(f"Sample raw dataset with {len(df)} transactions generated at {output_path}")

if __name__ == "__main__":
    generate_ecommerce_data()
