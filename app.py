import spacy
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

nlp = spacy.load("en_core_web_sm")

def load_knowledge_base(file_path):
    knowledge_df = pd.read_csv(file_path)
    knowledge_df.columns = [col.lower() for col in knowledge_df.columns]
    if 'answer' in knowledge_df.columns:
        knowledge_df['answer'].fillna("No answer provided", inplace=True)
    knowledge_base = {
        row['question'].lower(): row['answer']
        for _, row in knowledge_df.iterrows() if 'question' in row and 'answer' in row
    }
    return knowledge_base

csv_file_path = 'data/basic_example.csv'
knowledge_base = load_knowledge_base(csv_file_path)

app = Flask(__name__)
CORS(app)

def extract_intent_and_entities(user_query):
    """
    Analyze the user query using NLP techniques to extract intent and key entities.
    """
    doc = nlp(user_query.lower())
    entities = {ent.label_: ent.text for ent in doc.ents}  # Named entities
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]  # Filter out stopwords and non-alphabetic tokens
    
    return {
        "intent": "general_query",  # Placeholder intent for now
        "entities": entities,
        "keywords": keywords
    }

def find_answer(user_query):
    user_query_doc = nlp(user_query.lower())
    intent_data = extract_intent_and_entities(user_query)

    best_match = None
    highest_similarity = 0

    for question, answer in knowledge_base.items():
        question_doc = nlp(question)
        similarity = user_query_doc.similarity(question_doc)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    if highest_similarity > 0.7:
        return best_match, intent_data
    else:
        return (
            "I'm sorry, I couldn't find a relevant answer. Let me connect you with a support agent.",
            intent_data
        )

@app.route('/api/query', methods=['POST'])
def query_support():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response, intent_data = find_answer(user_query)
    return jsonify({
        "response": response,
        "intent": intent_data.get("intent"),
        "entities": intent_data.get("entities"),
        "keywords": intent_data.get("keywords")
    })

if __name__ == "__main__":
    app.run(debug=True)
