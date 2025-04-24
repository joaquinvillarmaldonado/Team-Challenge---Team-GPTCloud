from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    return '''
        Bienvenido a la API de predicción del modelo de lluvia.
        Para hacer una predicción, usa el endpoint /api/v1/predict con un método GET o POST.
        Para la predicción con GET, los parámetros son: DAY, MONTH, YEAR, TMAX, TMIN.
        Para hacer un retrain del modelo, usa el endpoint /api/v1/retrain.
    '''

# Enruta la funcion al endpoint /api/v1/predict (con GET)
@app.route('/api/v1/predict', methods=['GET'])
def predict_get():
    # Obtener parámetros de la URL
    day = request.args.get('DAY', type=int)
    month = request.args.get('MONTH', type=int)
    year = request.args.get('YEAR', type=int)
    tmax = request.args.get('TMAX', type=float)
    tmin = request.args.get('TMIN', type=float)

    # Validar que los parámetros estén presentes
    if None in [day, month, year, tmax, tmin]:
        return jsonify({'message': 'Faltan parámetros necesarios para realizar la predicción.'}), 400

    # Preparar los datos de entrada para la predicción
    input_data = np.array([[day, month, year, tmax, tmin]])

    # Cargar el modelo
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Hacer la predicción
    prediction = model.predict(input_data)

    return jsonify({'predicted_rain': int(prediction[0])})

# Enruta la funcion al endpoint /api/v1/predict (con POST)
@app.route('/api/v1/predict', methods=['POST'])
def predict_post():
    # Obtener los parámetros desde el cuerpo de la solicitud
    data = request.get_json()

    day = data.get('DAY')
    month = data.get('MONTH')
    year = data.get('YEAR')
    tmax = data.get('TMAX')
    tmin = data.get('TMIN')

    # Validar que los parámetros estén presentes
    if None in [day, month, year, tmax, tmin]:
        return jsonify({'message': 'Faltan parámetros necesarios para realizar la predicción.'}), 400

    # Preparar los datos de entrada para la predicción
    input_data = np.array([[day, month, year, tmax, tmin]])

    # Cargar el modelo
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Hacer la predicción
    prediction = model.predict(input_data)

    return jsonify({'predicted_rain': int(prediction[0])})

# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

# Tercer endpoint (comentado inicialmente)
# @app.route('/api/v1/test', methods=['GET'])
# def test():
#     return jsonify({'message': 'Este es un endpoint de prueba.'})

if __name__ == '__main__':
    app.run(debug=True)
