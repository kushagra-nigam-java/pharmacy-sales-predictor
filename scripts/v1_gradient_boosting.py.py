import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

df = pd.read_csv('pharmacy_otc_sales_data.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

le_country = LabelEncoder()
le_product = LabelEncoder()

df['Country_Encoded'] = le_country.fit_transform(df['Country'])
df['Product_Encoded'] = le_product.fit_transform(df['Product'])

X = df[['Country_Encoded', 'Month', 'DayOfWeek', 'Boxes Shipped', 'Amount ($)']]
y = df['Product_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("-" * 30)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("-" * 30)

print("Saving model components...")
joblib.dump(model, 'medicine_model.pkl')
joblib.dump(le_country, 'country_encoder.pkl')
joblib.dump(le_product, 'product_encoder.pkl')
print("All files saved: medicine_model.pkl, country_encoder.pkl, product_encoder.pkl")

#  TERMINAL CONFUSION MATRIX 
print("\n" + "="*30)
print("TEXT-BASED CONFUSION MATRIX")
print("="*30)

matrix_df = pd.crosstab(
    le_product.inverse_transform(y_test), 
    le_product.inverse_transform(y_pred), 
    rownames=['Actual'], 
    colnames=['Predicted'],
    margins=True 
)

print(matrix_df)
print("="*30)
