from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv

def create_app():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Load the CSV data into memory
    issues_data = []
    try:
        with open('data/tech_support_dataset.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                issues_data.append(row)
    except FileNotFoundError:
        print("Warning: 'tech_support_dataset' file not found. Defaulting to empty dataset.")
        
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index(): 
        return render_template('chat.html')

    @app.route('/api/query', methods=['POST'])
    def chat():
        data = request.get_json()  
        if 'query' not in data:
            return "Error: No query key found in the request", 400

        user_input = data['query']
        response = get_response(user_input)
        return jsonify({"response": response})

    def get_response(user_input):
        # Attempt to find a response based on the CSV dataset
        issue_response = find_issue_response(user_input)
        if issue_response:
            return issue_response
        else:
            # If no match is found, generate a response using DialoGPT
            return get_Chat_response(user_input)

    def find_issue_response(user_input):
        best_match = None
        highest_score = 0
        
        # Check each row in the issues_data to see if the user input matches an issue
        for issue in issues_data:
            issue_description = issue['Customer_Issue'].lower()
            match_score = calculate_match_score(user_input.lower(), issue_description)
            
            # Select the issue with the highest match score
            if match_score > highest_score:
                highest_score = match_score
                best_match = issue
        
        if best_match and highest_score > 0.5:  # Threshold for a match
            return generate_technical_response(best_match)
        return None

    def calculate_match_score(user_input, issue_description):
        # Calculate match score based on keyword overlap or other metrics
        user_input_words = set(user_input.split())
        issue_description_words = set(issue_description.split())
        common_words = user_input_words.intersection(issue_description_words)
        return len(common_words) / len(issue_description_words)

    def generate_technical_response(issue):
        # Generate a detailed technical response based on the best match
        return f" {issue['Tech_Response']}"

    def get_Chat_response(text):
        chat_history_ids = torch.tensor([])  # Initialize chat history for each conversation
        for step in range(5):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generate a response while limiting the total chat history to 1000 tokens
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
