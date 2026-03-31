import joblib
import pandas as pd
import os
from datetime import datetime

MODEL_PATH = 'medicine_model.pkl'
COUNTRY_LE_PATH = 'country_encoder.pkl'
PRODUCT_LE_PATH = 'product_encoder.pkl'

def run_cli():
    print("\n--- pharmacy-sales-predictor CLI Tool ---")
   
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found in this folder!")
        return

    print("Loading model and encoders...")
    model = joblib.load(MODEL_PATH)
    le_country = joblib.load(COUNTRY_LE_PATH)
    le_product = joblib.load(PRODUCT_LE_PATH)
    print("System Ready!\n")

    try:
        print(f"Markets: {list(le_country.classes_)}")
        country = input("Enter Market: ").strip()
        
        boxes = int(input("Enter Boxes Shipped: "))
        amount = float(input("Enter Amount ($): "))
        
        date_str = datetime.now()
        month = date_str.month
        day_of_week = date_str.weekday()

        country_enc = le_country.transform([country])[0]

        input_df = pd.DataFrame([[country_enc, month, day_of_week, boxes, amount]], 
                                columns=['Country_Encoded', 'Month', 'DayOfWeek', 'Boxes Shipped', 'Amount ($)'])

        prediction = model.predict(input_df)
        result = le_product.inverse_transform(prediction)[0]

        print("\n" + "="*30)
        print(f"PREDICTED PRODUCT: {result}")
        print("="*30 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_cli()
