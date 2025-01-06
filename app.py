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
        with open('data/tech_support_dataset', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                issues_data.append(row)
    except FileNotFoundError:
        print("Warning: 'tech_support_dataset' file not found. Defaulting to empty dataset.")
        # Optionally, you can set some default data or leave issues_data empty

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
        # Check if the user's input matches any issue in the CSV
        issue_response = find_issue_response(user_input)
        if issue_response:
            return issue_response  # Return response from CSV if matched
        else:
            # If no match is found, generate a response using DialoGPT
            return get_Chat_response(user_input)

    def find_issue_response(user_input):
        # Check each row in the issues_data to see if the user input matches an issue
        for issue in issues_data:
            if user_input.lower() in issue['Customer_Issue'].lower():
                return f"Tech Response: {issue['Tech_Response']}\nResolution Time: {issue['Resolution_Time']}\nIssue Category: {issue['Issue_Category']}\nIssue Status: {issue['Issue_Status']}"
        # Return None if no match is found
        return None

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


#from flask import Flask, render_template, request, jsonify
#from flask_cors import CORS
#from transformers import AutoModelForCausalLM, AutoTokenizer
#import torch
#
#def create_app():
#    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
#    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
#
#    app = Flask(__name__)
#    CORS(app)
#
#    @app.route("/")
#    def index(): 
#        return render_template('chat.html')
#
#    @app.route('/api/query', methods=['POST'])
#    def chat():
#        data = request.get_json()  
#        if 'query' not in data:
#            return "Error: No query key found in the request", 400
#
#        user_input = data['query']
#        response = get_Chat_response(user_input)
#        return jsonify({"response": response})
#
#    def get_Chat_response(text):
#        chat_history_ids = torch.tensor([])  # Initialize chat history for each conversation
#        for step in range(5):
#            # encode the new user input, add the eos_token and return a tensor in Pytorch
#            new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
#
#            # append the new user input tokens to the chat history
#            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
#
#            # generate a response while limiting the total chat history to 1000 tokens
#            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#
#            # pretty print last output tokens from the bot
#        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
#
#    return app
#
#if __name__ == '__main__':
#    app = create_app()
#    app.run()
#