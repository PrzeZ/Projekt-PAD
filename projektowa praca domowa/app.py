import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("messy_data.csv", names=['carat', 'clarity', 'color', 'cut', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price'])
df.drop(index=0, inplace=True)

# zamiana dziwnych i pustych wartości na NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# konwersja na typ float
df['carat'] = df['carat'].astype(float)
df['x dimension'] = df['x dimension'].astype(float)
df['y dimension'] = df['y dimension'].astype(float)
df['z dimension'] = df['z dimension'].astype(float)
df['depth'] = df['depth'].astype(float)
df['table'] = df['table'].astype(float)
df['price'] = df['price'].astype(float)

# zamiana na duże litery
df['cut'] = df['cut'].str.upper()
df['clarity'] = df['clarity'].str.upper()
df['color'] = df['color'].str.upper()

# zamiana NaN na średnią tam, gdzie się da
df['carat'].fillna(df["carat"].mean(), inplace=True)
df['x dimension'].fillna(df["x dimension"].mean(), inplace=True)
df['y dimension'].fillna(df["y dimension"].mean(), inplace=True)
df['z dimension'].fillna(df["z dimension"].mean(), inplace=True)
df['depth'].fillna(df["depth"].mean(), inplace=True)
df['table'].fillna(df["table"].mean(), inplace=True)
df['price'].fillna(df["price"].mean(), inplace=True)
# Drop rows with missing values
df = df.dropna()

# Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['clarity', 'color', 'cut'], drop_first=True)  # add drop_first=True to avoid dummy variable trap

# Split the data into features and target variable
X = df.drop(columns=['price'])
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.write("Mean Squared Error:", mse)

# Add a selectbox to choose the column for x-axis
selected_column = st.selectbox("Select column for X-axis", X.columns)

# Plot results using Plotly Express
fig = px.scatter(x=y_test, y=X_test[selected_column], labels={'x': 'Price', 'y': selected_column}, title=selected_column + ' vs Price')

# Add regression line
fig.add_scatter(x=y_pred, y=X_test[selected_column], mode='markers', name='Predicted Price')

st.plotly_chart(fig)
