import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

# Tytuł
st.title('Przewidywanie Ceny Diamentów (Projekt PAD)')
st.write("""
Aplikacja używa modelu regresji liniowej aby przewidywać ceny diamentów na podstawie różnych zmiennych
Możesz wybrać inne zmienne i sprawdzić jak są powiązane z ceną diamentu.
""")

# Ładowanie danych
st.subheader('1. Załadowanie Danych')
df = pd.read_csv("messy_data.csv", names=['carat', 'clarity', 'color', 'cut', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price'])
df.drop(index=0, inplace=True)

# Pokazanie niewyczyszczonych danych
st.write("""
Poniżej załadowane dane przed czyszczeniem:
""")
st.dataframe(df)

# Zamiana pustych wartości na NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Konwersja numerycznych kolumn na float
numeric_cols = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Uzupełnienie NaN wartością średnią
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Zamiana tekstów na duże litery
categorical_cols = ['clarity', 'color', 'cut']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.upper())

# Zamiana zmiennych kategorycznych na dummy variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Pokazanie ramki danych
st.subheader('2. Czyszczenie danych')
st.write("""
Poniżej wyczyszczone i przetworzone dane, gotowe do modelowania:
""")
st.dataframe(df)

# Rozdzielenie ramki danych na zmienne i target
X = df.drop(columns=['price'])
y = df['price']

# Rozdzielenie na zestaw do uczenia i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Selekcja postępująca
st.subheader('3. Wybór zmiennych za pomocą selekcji postępującej')
st.write("""
Używamy selekcji postępującej aby wybrać ważne zmienne.
""")

# Model regresji
model = LinearRegression()

# Selektor zmiennych
sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward')

# Dopasowanie selekcji
sfs.fit(X_train, y_train)

# Pobierz indeksy
selected_feature_indices = sfs.get_support(indices=True)

# Pobierz nazwy kolumn
selected_features = X.columns[selected_feature_indices]

# Podziel na dane treningowe i testowe
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]

# Dopasuj model
model.fit(X_train_selected, y_train)

# Szacuj na zestawie testowym
y_pred_selected = model.predict(X_test_selected)

# Policz błąd
mse_selected = mean_squared_error(y_test, y_pred_selected)
st.write("Błąd kwadratowy:", mse_selected)

# Wizualizacja
st.subheader('4. Wizualizacja')
st.write("""
Poniżej wykres który pokazuje zależność:
""")

# Selectbox
selected_column = st.selectbox("Wybierz kolumnę dla X", X.columns, index=0)

# Plotly Express
fig = px.scatter(x=y_test, y=X_test[selected_column], labels={'x': 'Price', 'y': selected_column}, title=f'{selected_column} vs Price')

# Checkbox
show_regression = st.checkbox('Pokaż Regresję', value=False)

if show_regression:
    fig.add_scatter(x=y_pred_selected, y=X_test[selected_column], mode='markers', name='Przewidywana Cena')

st.plotly_chart(fig)