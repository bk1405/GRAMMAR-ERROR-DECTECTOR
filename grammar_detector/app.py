from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the model and tokenizer
model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def correct_grammar(text, iterations=2):
    corrected = text
    for _ in range(iterations):  # Apply multiple passes for better correction
        input_text = "grammar: " + corrected
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded == corrected:
            break
        corrected = decoded
    return corrected

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    corrected_text = correct_grammar(input_text)
    return render_template('result.html', original_text=input_text, corrected_text=corrected_text)

if __name__ == '__main__':
    app.run(debug=True)





