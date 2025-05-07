from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from flask_cors import CORS  # Untuk handle CORS

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua route

# Load model dan scaler
model = joblib.load("model_xgboost.pkl")
scaler = joblib.load("scaler.pkl")

# Koneksi ke MongoDB (untuk logging)
client = MongoClient("mongodb://localhost:27017/")
db = client["staff_db"]

# Kolom yang diharapkan
expected_columns = [
    'department', 'education', 'gender', 'recruitment_channel',
    'no_of_trainings', 'age', 'previous_year_rating',
    'length_of_service', 'KPIs_met >80%', 'awards_won?',
    'avg_training_score'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.get_json()
        
        # Validasi data
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Konversi ke tipe data yang tepat dan mapping nama feature
        input_data = [
            int(data.get('department', 0)),
            int(data.get('education', 0)),
            int(data.get('gender', 0)),
            int(data.get('recruitment_channel', 0)),
            int(data.get('no_of_trainings', 0)),
            int(data.get('age', 0)),
            int(data.get('previous_year_rating', 0)),
            int(data.get('length_of_service', 0)),
            int(data.get('KPIs_met >80%', 0)),
            int(data.get('awards_won', 0)),
            float(data.get('avg_training_score', 0))
        ]
        
        # Buat DataFrame dengan nama kolom yang benar
        sample_df = pd.DataFrame([input_data], columns=expected_columns)
        
        # Scaling
        sample_scaled = scaler.transform(sample_df)
        
        # Prediksi
        pred = model.predict(sample_scaled)
        
        # Hasil prediksi
        result = "promoted" if pred[0] == 1 else "not_promoted"
        
        # Simpan ke MongoDB, tapi salin data agar tidak terkontaminasi _id
        data_to_save = data.copy()
        data_to_save['prediction'] = result
        inserted = db['predictions'].insert_one(data_to_save)

        # Siapkan response tanpa ObjectId
        data['prediction'] = result
        data['id'] = str(inserted.inserted_id)

        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'data': data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
