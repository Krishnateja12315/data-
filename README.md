import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your sales data
df = pd.read_csv('sales_data.csv')

# Feature Engineering
df['Total_Weight'] = df['Quantity_Sold'] * df['Unit_Weight']

# Prepare features and target
X = df[['Quantity_Sold', 'Unit_Weight']]  # Add more features if needed
y = df['Total_Weight']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error
print("MSE:", mean_squared_error(y_test, y_pred))
