import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = 'E:/Vityarthi project 2nd sem/archive/pharmacy_otc_sales_data.csv'
df = pd.read_csv(file_path)

product_knowledge_base = {
    'Paracetamol': {'Category': 'Pain Relief', 'Form': 'Tablet', 'Demographic': 'Adult'},
    'Cough Syrup': {'Category': 'Cold & Flu', 'Form': 'Liquid', 'Demographic': 'All'},
    'Vitamin C':   {'Category': 'Supplement', 'Form': 'Gummy', 'Demographic': 'Child'},
    'Ibuprofen':   {'Category': 'Pain Relief', 'Form': 'Tablet', 'Demographic': 'Adult'},
}

def get_category(prod): return product_knowledge_base.get(prod, {}).get('Category', 'Unknown_Cat')
def get_form(prod):     return product_knowledge_base.get(prod, {}).get('Form', 'Unknown_Form')
def get_demo(prod):     return product_knowledge_base.get(prod, {}).get('Demographic', 'Unknown_Demo')

df['Product_Category'] = df['Product'].apply(get_category)
df['Product_Form'] = df['Product'].apply(get_form)
df['Target_Demographic'] = df['Product'].apply(get_demo)

df.to_csv('E:/Vityarthi project 2nd sem/archive/enriched_pharmacy_data.csv', index=False)
print("Enriched dataset saved with new columns!")

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Price_Per_Box'] = np.where(df['Boxes Shipped'] == 0, 0, df['Amount ($)'] / df['Boxes Shipped'])
encoders = {}
categorical_cols = ['Country', 'Product', 'Product_Category', 'Product_Form', 'Target_Demographic']

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col + '_Encoded'] = encoders[col].fit_transform(df[col])
X = df[['Country_Encoded', 'Month', 'DayOfWeek', 'Boxes Shipped', 'Amount ($)', 
        'Price_Per_Box', 'Product_Category_Encoded', 'Product_Form_Encoded', 'Target_Demographic_Encoded']]
y = df['Product_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 50)
print(f"New Model Accuracy with Enriched Data: {accuracy * 100:.2f}%")
print("-" * 50)
