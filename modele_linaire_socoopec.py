import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def entrainer_modele(data):
    X = data[["temperature", "vibration"]]
    y = data["indice_panne"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, mae, r2

if __name__ == "__main__":
    data = pd.DataFrame({
        "temperature": [23, 25, 28, 22, 30, 27, 26, 29],
        "vibration": [0.1, 0.2, 0.25, 0.18, 0.3, 0.28, 0.24, 0.31],
        "indice_panne": [0, 0, 1, 0, 1, 1, 0, 1]
    })

    modele, rmse, mae, r2 = entrainer_modele(data)
    print("Modèle entraîné avec succès !")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    # Prédiction pour une nouvelle machine
    nouvelle_machine = np.array([[28, 0.27]])
    prediction = modele.predict(nouvelle_machine)
    print("Indice de panne prédit :", prediction[0])
