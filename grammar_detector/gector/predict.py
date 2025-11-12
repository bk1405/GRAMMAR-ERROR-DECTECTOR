from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer from Hugging Face
model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def correct_grammar(text: str, iterations: int = 2) -> str:
    """
    Corrects the grammar of the given text using T5 model.
    It applies multiple passes to improve correction accuracy.

    Parameters:
        text (str): The input sentence with potential grammar errors.
        iterations (int): Number of correction passes to apply.

    Returns:
        str: The grammar-corrected version of the input text.
    """
    corrected = text
    for _ in range(iterations):
        input_text = "grammar: " + corrected
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            num_return_sequences=1
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.strip() == corrected.strip():
            break
        corrected = decoded.strip()
    return corrected

# For testing independently
if __name__ == "__main__":
    test_sentences = [
        "She not like mango.",
        "I no understand this.",
        "He going school everyday.",
        "They was playing cricket.",
        "My brother have car.",
        "It raining outside.",
        "This is goodest one."
    ]

    for sentence in test_sentences:
        corrected = correct_grammar(sentence)
        print(f"Original : {sentence}")
        print(f"Corrected: {corrected}\n")



