from flask import Flask, jsonify, request
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# Crear la aplicación Flask
app = Flask(__name__)

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    return '''
        Bienvenido a la API de predicción del modelo de lluvia del Team GPTCloud.<br>
        Para hacer una predicción, usa el endpoint /api/v1/predict con un método GET o POST.<br>
        Para la predicción con GET, los parámetros son: DAY, MONTH, YEAR, TMAX, TMIN.<br>
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
    # Verificar si el archivo de datos existe
    if os.path.exists("data/seattleWeather_1948-2017.csv"):
        # Cargar los datos del archivo CSV
        data = pd.read_csv('data/seattleWeather_1948-2017.csv')

        # Eliminar cualquier fila con valores nulos o vacíos
        data = data.dropna(subset=['DAY', 'MONTH', 'YEAR', 'TMAX', 'TMIN', 'RAIN'])

        # Filtrar las columnas relevantes para el modelo
        X = data[['DAY', 'MONTH', 'YEAR', 'TMAX', 'TMIN']]  # Características (features)
        y = data['RAIN']  # Etiqueta (target)

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inicializar y entrenar el modelo de regresión logística
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Realizar predicciones sobre el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calcular las métricas de evaluación
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred)

        # Reentrenar el modelo con todos los datos
        model.fit(X, y)

        # Guardar el modelo entrenado
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Devolver las métricas de evaluación
        return jsonify({
            'message': 'Modelo reentrenado exitosamente.',
            'metrics': {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'roc_auc': auc_roc
            }
        })
    else:
        return jsonify({'message': 'No se encontraron datos nuevos para reentrenar el modelo.'}), 404

# Tercer endpoint comentado (para prueba)
# @app.route('/api/v1/test', methods=['GET'])
# def test():
#     return jsonify({'message': 'Este es un endpoint de prueba.'})

if __name__ == '__main__':
    app.run(debug=True)