
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
# Load the model using the Hugging Face library
model = TFAutoModelForSequenceClassification.from_pretrained("../ArticleTagModel1", from_pt=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("../ArticleTagModel1")

# Prediction function
def predict_text(input_text):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True)

    # Perform prediction
    logits = model(inputs["input_ids"]).logits
    predicted_label = tf.argmax(logits, axis=1).numpy()[0]
    return predicted_label

# Example usage
new_text = "I love being a cricket."
prediction = predict_text(new_text)
print(f"Predicted Label: {prediction}")