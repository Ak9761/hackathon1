from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from googletrans import Translator  # Import the googletrans library

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the BART summarization pipeline once on startup
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn"
)

# Initialize Google Translator
translator = Translator()

# Function to handle translation
def translate_text(text, target_lang='en'):
    translated = translator.translate(text, dest=target_lang)
    return translated.text

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    print("Received request:", data)
    input_text = data.get('text', '')
    language = data.get('language', 'en')  # Default to English if no language is provided

    # Translate the input text to English before summarizing
    translated_text = translate_text(input_text, 'en')

    # Run the BART summarizer on the translated text
    summary_list = summarizer(
        translated_text,
        max_length=1500,
        min_length=25,
        do_sample=False
    )

    summary_text = summary_list[0]['summary_text']

    # Translate the summary back to the requested language
    final_summary = translate_text(summary_text, language)

    return jsonify({'summary': final_summary})

if __name__ == '__main__':
    app.run(debug=True)

