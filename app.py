import spacy
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

nlp = spacy.load("en_core_web_sm")

def load_knowledge_base(file_path):
    knowledge_df = pd.read_csv(file_path)
    knowledge_base = {}
    for index, row in knowledge_df.iterrows():
        knowledge_base[row['question'].lower()] = row['answer']
    return knowledge_base

csv_file_path = 'data/basic_example.csv'
knowledge_base = load_knowledge_base(csv_file_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

def find_answer(user_query):
    user_query_doc = nlp(user_query.lower())
    best_match = None
    highest_similarity = 0

    for question, answer in knowledge_base.items():
        question_doc = nlp(question)
        similarity = user_query_doc.similarity(question_doc)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    if highest_similarity > 0.7: 
        return best_match
    else:
        return "I'm sorry, I couldn't find a relevant answer. Let me connect you with a support agent."

@app.route('/api/query', methods=['POST'])
def query_support():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response = find_answer(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
