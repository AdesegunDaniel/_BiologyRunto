from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
import os
from flask_cors import CORS


model_name = "/home/ec2-user/.cache/huggingface/hub/models--AdesegunDaniel--BiologyRunto/snapshots/2fe1fcbd72d1f55334ac18eed9888878ec6f483a"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def respond(input_text=""):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=100, num_beams=3, early_stopping=True)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response


app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reply', methods=['POST'])
def reply():
    user_input = request.json['user_text']
    bot_response = respond(user_input)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)
