import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load datasets
sheet3_path = r"C:\\Users\\anich\\chatbot_project\\backend\\data\\Sheet3.csv"
worksheet_path = r"C:\\Users\\anich\\chatbot_project\\backend\\data\\Worksheet.csv"

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

# Ask questions from a given column
def ask_questions(questions):
    responses = []
    for question in questions:
        print(f"{question} (ya/tidak)")
        response = input().strip().lower()
        responses.append(response == 'ya')
    return responses

# Summarize results and determine disorder
def summarize_results(user_responses, worksheet):
    disorder_scores = {}
    for i, response in enumerate(user_responses):
        if response:
            disorder = worksheet.iloc[i]['DISORDER']
            if disorder in disorder_scores:
                disorder_scores[disorder] += 1
            else:
                disorder_scores[disorder] = 1
    return max(disorder_scores, key=disorder_scores.get), disorder_scores

# Main chatbot logic
while True:
    print("Halo! Ceritakan bagaimana perasaan Anda hari ini.")
    user_input = input("Anda: ").strip()
    if user_input.lower() in ['exit', 'keluar']:
        print("Terima kasih telah berbagi. Semoga hari Anda membaik!")
        break

    # Preprocess input
    user_input = preprocess_text(user_input)

    # Match input with Sheet3 using TF-IDF
    response = match_keyword_tfidf(user_input, sheet3)
    if response:
        print(f"Chatbot: {response}")
    else:
        print("Chatbot: Aku ingin memastikan keadaanmu, apa kamu bisa menjawab beberapa pertanyaan dari ku?")
        consent = input("Anda: ").strip().lower()
        if consent == 'ya':
            # Ask initial questions
            initial_questions = worksheet['PERTANYAAN'].dropna().tolist()
            user_responses = ask_questions(initial_questions)

            # Check if more than 50% are 'ya'
            if sum(user_responses) > len(initial_questions) / 2:
                print("Chatbot: Aku cukup khawatir dengan keadaanmu, apa kamu bisa menjawab pertanyaan lanjutan agar aku bisa memastikan keadaanmu?")
                follow_up_consent = input("Anda: ").strip().lower()
                if follow_up_consent == 'ya':
                    # Ask follow-up questions
                    follow_up_questions = worksheet['PERTANYAAN LANJUTAN'].dropna().tolist()
                    follow_up_responses = ask_questions(follow_up_questions)

                    # Summarize results
                    disorder, scores = summarize_results(follow_up_responses, worksheet)

                    # Save results to psychologist database (placeholder)
                    print("\n[Data telah dikirim ke database psikolog untuk validasi]")
                    print(f"Disorder Detected: {disorder}")
                    print(f"Scores: {scores}")

                    # Provide solutions
                    solution = worksheet[worksheet['DISORDER'] == disorder]['INITIAL SOLUTION'].dropna().iloc[0]
                    triggers = worksheet[worksheet['DISORDER'] == disorder]['FAKTOR PEMICU'].dropna().iloc[0]
                    print(f"Chatbot: Berdasarkan analisis, ini beberapa solusi awal untukmu:\n{solution}\n\nFaktor pemicu:\n{triggers}")
                else:
                    print("Chatbot: Terima kasih telah berbagi. Jika ada sesuatu yang ingin kamu ceritakan lagi, aku di sini untuk membantu.")
            else:
                print("Chatbot: Gejala yang kamu alami ringan. Cobalah untuk beristirahat dan menjaga pola hidup sehat.")

def save_chat_to_db(user_id, message, response):
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_logs (user_id, message, response, timestamp) 
        VALUES (?, ?, ?, datetime('now'))
    """, (user_id, message, response))
    conn.commit()
    conn.close()

@app.route('/api/admin/profile', methods=['POST'])
def save_admin_profile():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    if not name or not email:
        return jsonify({"status": "error", "message": "Nama dan email harus diisi"}), 400
    # Simpan data profil admin ke database
    return jsonify({"status": "success", "message": "Profil admin berhasil disimpan"})

@app.route('/api/user/activity', methods=['GET'])
def user_activity():
    # Contoh data dummy
    activity_data = {
        "total_chat": 10,
        "total_login": 5,
        "last_login": "2024-12-26",
        "counseling_history": ["Sesi 1 - Baik", "Sesi 2 - Sangat Baik"]
    }
    return jsonify(activity_data)

@app.route('/api/admin/user_data', methods=['GET'])
def user_data():
    # Contoh data dummy
    users = [
        {"name": "User A", "email": "usera@example.com", "login_count": 10},
        {"name": "User B", "email": "userb@example.com", "login_count": 5}
    ]
    return jsonify(users)
