import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datetime import datetime

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

# Ask questions from a given column without repetition
def ask_questions(questions, asked_questions):
    responses = []
    for question in questions:
        if question not in asked_questions:
            print(f"{question} (ya/tidak)")
            response = input().strip().lower()
            responses.append(response == 'ya')
            asked_questions.add(question)
    return responses

# Summarize results and determine disorder
def summarize_results(user_responses, worksheet):
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
    return max(disorder_scores, key=disorder_scores.get), disorder_scores, detailed_responses

# Save conversation to dataset
def save_conversation(session_start_time, user_input, chatbot_response, detailed_responses, disorder, scores, solution, conversation_log):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conversation_log.append({
        "Timestamp": timestamp,
        "Session Start": session_start_time,
        "User Input": user_input,
        "Chatbot Response": chatbot_response,
        "Questions": [response['Question'] for response in detailed_responses],
        "Answers": [response['Response'] for response in detailed_responses],
        "Disorder Detected": disorder,
        "Scores": scores,
        "Solution": solution,
        "Dropdown Valid/Tidak Valid": "",
        "Kritik dan Saran": ""
    })

# Main chatbot logic
conversation_log = []
asked_questions = set()

session_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
conversation_log.append({"Timestamp": session_start_time, "Session Start": "=== Sesi Dimulai ===", "User Input": "", "Chatbot Response": "", "Questions": "", "Answers": "", "Disorder Detected": "", "Scores": "", "Solution": "", "Dropdown Valid/Tidak Valid": "", "Kritik dan Saran": ""})

print("Halo! Ceritakan bagaimana perasaan Anda hari ini.")
while True:
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
        save_conversation(session_start_time, user_input, response, [], "", {}, "", conversation_log)
    else:
        print("Chatbot: Aku ingin memastikan keadaanmu, apa kamu bisa menjawab beberapa pertanyaan dari ku? (ketik Ya)")
        consent = input("Anda: ").strip().lower()
        if consent == 'ya':
            # Ask initial questions
            initial_questions = worksheet['PERTANYAAN'].dropna().tolist()
            user_responses = ask_questions(initial_questions, asked_questions)

            # Log jumlah jawaban "Ya" untuk debugging
            print(f"Jumlah jawaban 'Ya': {sum(user_responses)}, Batas: {len(initial_questions) / 2}")

            # Check if more than 50% are 'ya'
            if sum(user_responses) > len(initial_questions) / 2:
                print("Chatbot: Aku cukup khawatir dengan keadaanmu, apa kamu bisa menjawab pertanyaan lanjutan agar aku bisa memastikan keadaanmu? (ketik Ya)")
                follow_up_consent = input("Anda: ").strip().lower()
                if follow_up_consent == 'ya':
                    # Ask follow-up questions
                    follow_up_questions = worksheet['PERTANYAAN LANJUTAN'].dropna().tolist()
                    follow_up_responses = ask_questions(follow_up_questions, asked_questions)

                    # Summarize results
                    disorder, scores, detailed_responses = summarize_results(follow_up_responses, worksheet)

                    # Save results to psychologist database (placeholder)
                    print("\n[Data telah dikirim ke database psikolog untuk validasi]")

                    # Provide solutions
                    solution = worksheet[worksheet['DISORDER'] == disorder]['INITIAL SOLUTION'].dropna().iloc[0]
                    triggers = worksheet[worksheet['DISORDER'] == disorder]['FAKTOR PEMICU'].dropna().iloc[0]
                    print(f"Chatbot: Berdasarkan analisis, ini beberapa solusi awal untukmu:\n{solution}\n\nFaktor pemicu:\n{triggers}")

                    print("Chatbot: Data didapatkan, apa kamu ingin mengakhiri percakapan ini? Jika ya, ketik exit.")

                    save_conversation(session_start_time, user_input, "Chatbot memberikan solusi", detailed_responses, disorder, scores, solution, conversation_log)
                else:
                    print("Chatbot: Terima kasih telah berbagi. Jika ada sesuatu yang ingin kamu ceritakan lagi, aku di sini untuk membantu.")
                    save_conversation(session_start_time, user_input, "Terima kasih telah berbagi", [], "", {}, "", conversation_log)
            else:
                print("Chatbot: Gejala yang kamu alami ringan. Cobalah untuk beristirahat dan menjaga pola hidup sehat.")
                save_conversation(session_start_time, user_input, "Gejala ringan, saran untuk menjaga pola hidup sehat", [], "", {}, "", conversation_log)

                # Tambahkan opsi untuk melanjutkan ke pertanyaan lanjutan secara manual
                print("Chatbot: Jika kamu ingin menjawab pertanyaan lanjutan, ketik 'Ya'. Jika tidak, ketik 'Tidak'.")
                continue_to_follow_up = input("Anda: ").strip().lower()
                if continue_to_follow_up == 'ya':
                    follow_up_questions = worksheet['PERTANYAAN LANJUTAN'].dropna().tolist()
                    follow_up_responses = ask_questions(follow_up_questions, asked_questions)

                    # Summarize results
                    disorder, scores, detailed_responses = summarize_results(follow_up_responses, worksheet)

                    # Save results to psychologist database (placeholder)
                    print("\n[Data telah dikirim ke database psikolog untuk validasi]")

                    # Provide solutions
                    solution = worksheet[worksheet['DISORDER'] == disorder]['INITIAL SOLUTION'].dropna().iloc[0]
                    triggers = worksheet[worksheet['DISORDER'] == disorder]['FAKTOR PEMICU'].dropna().iloc[0]
                    print(f"Chatbot: Berdasarkan analisis, ini beberapa solusi awal untukmu:\n{solution}\n\nFaktor pemicu:\n{triggers}")

                    print("Chatbot: Data didapatkan, apa kamu ingin mengakhiri percakapan ini? Jika ya, ketik exit.")

                    save_conversation(session_start_time, user_input, "Chatbot memberikan solusi", detailed_responses, disorder, scores, solution, conversation_log)

# Save conversation logs to a new dataset
conversation_log_df = pd.DataFrame(conversation_log)
conversation_log_df.to_csv("conversation_log.csv", index=False)
print("Percakapan telah disimpan dalam conversation_log.csv")
