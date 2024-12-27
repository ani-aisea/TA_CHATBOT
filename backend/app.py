from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from datetime import datetime


# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Mengizinkan CORS untuk semua rute
CORS(app)

@app.route('/api/user/profile', methods=['POST'])
def save_user_profile():
    data = request.json
    print("Data diterima:", data)  # Tambahkan log untuk debugging
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    # Simpan profil ke database
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_profiles (name, email) VALUES (?, ?)", (data['name'], data['email']))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Profil berhasil disimpan'}), 200

# Endpoint untuk menyimpan profil admin
@app.route('/api/admin/profile', methods=['POST'])
def save_admin_profile():
    data = request.json
    print("Data diterima:", data)
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({'status': 'error', 'message': 'Nama dan email diperlukan'}), 400
    # Logika penyimpanan data ke database
    return jsonify({'status': 'success', 'message': 'Profil berhasil disimpan'}), 200

# Endpoint untuk menyimpan profil psikolog
@app.route('/api/psychologist/profile', methods=['POST'])
def save_psychologist_profile():
    data = request.json
    print("Data diterima untuk psychologist:", data)
    if not data or not data.get('name') or not data.get('email'):
        return jsonify({'status': 'error', 'message': 'Nama dan email diperlukan'}), 400
    # Logika penyimpanan data ke database (opsional)
    return jsonify({'status': 'success', 'message': 'Profil psikolog berhasil disimpan'}), 200

 # Simpan profil ke database
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_profiles (name, email) VALUES (?, ?)", (data['name'], data['email']))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Profil berhasil disimpan'}), 200

# Load datasets
sheet3_path = "backend/data/Sheet3.csv"
worksheet_path = "backend/data/Worksheet.csv"

sheet3 = pd.read_csv(sheet3_path)
worksheet = pd.read_csv(worksheet_path)

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Preprocess text
def preprocess_text(text):
    return ' '.join(text.lower().strip().split())

# Match input with keywords from Sheet3 using TF-IDF
def match_keyword_tfidf(user_input, sheet3):
    keywords = sheet3['Keyword'].dropna().tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(keywords)
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    max_index = cosine_similarities.argmax()
    if cosine_similarities[max_index] > 0.3:  # Threshold for matching
        return sheet3.iloc[max_index]['Response']
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    user_message = preprocess_text(data['message'])
    
    # Match input with Sheet3 using TF-IDF
    response = match_keyword_tfidf(user_message, sheet3)

    if response:
        bot_response = response
    else:
        bot_response = "Aku tidak memahami, tetapi aku bisa bertanya lebih lanjut jika kamu bersedia."

    return jsonify({'response': bot_response}), 200

@app.route('/api/chat/analysis', methods=['POST'])
def chat_analysis():
    data = request.json
    if not data or 'responses' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    user_responses = data['responses']

    # Summarize results
    disorder_scores = {}
    detailed_responses = []
    for i, response in enumerate(user_responses):
        if response:
            disorder = worksheet.iloc[i]['DISORDER']
            question = worksheet.iloc[i]['PERTANYAAN']
            detailed_responses.append({"Question": question, "Disorder": disorder, "Response": "Ya"})
            if disorder in disorder_scores:
                disorder_scores[disorder] += 1
            else:
                disorder_scores[disorder] = 1

    if disorder_scores:
        disorder = max(disorder_scores, key=disorder_scores.get)
        solution = worksheet[worksheet['DISORDER'] == disorder]['INITIAL SOLUTION'].dropna().iloc[0]
        return jsonify({'disorder': disorder, 'solution': solution, 'details': detailed_responses}), 200
    else:
        return jsonify({'message': 'Tidak ada masalah yang terdeteksi'}), 200


# Endpoint untuk chatbot dengan logika respon
@app.route('/api/chat/response', methods=['POST'])
def chat_response():
    data = request.json
    user_id = data.get('user_id', 'unknown')  # Tambahkan user_id jika dikirimkan dari frontend
    message = data.get('message', '').lower()

    if "lelah" in message:
        response = "Saya memahami bahwa Anda merasa lelah. Mungkin Anda perlu istirahat atau berbicara dengan seseorang."
    elif "bahagia" in message:
        response = "Senang mendengar Anda bahagia! Tetap semangat ya."
    else:
        response = "Maaf, saya belum memahami pesan Anda. Bisa dijelaskan lebih lanjut?"

    # Simpan pesan ke database
    save_chat_to_db(user_id, message, response)

    return jsonify({"response": response})

# Fungsi untuk menyimpan chat ke database
def save_chat_to_db(user_id, user_message, bot_response):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_logs (user_id, user_message, bot_response, timestamp) VALUES (?, ?, ?, datetime('now'))",
        (user_id, user_message, bot_response)
    )
    conn.commit()
    conn.close()

# Endpoint untuk mendapatkan log chat
@app.route('/api/chat/logs', methods=['GET'])
def get_chat_logs():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()

    return jsonify({"logs": logs})


if __name__ == '__main__':
    app.run(debug=False)
