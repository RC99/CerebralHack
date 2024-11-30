from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

app = Flask(__name__)

# Set your OpenAI API key securely
client = OpenAI(api_key="") # Replace with your actual OpenAI API key

# Load the models and tokenizers
model1 = TFAutoModelForSequenceClassification.from_pretrained("../ArticleTagModel1/ArticleTag", from_pt=False)
tokenizer1 = AutoTokenizer.from_pretrained("../ArticleTagModel1/ArticleTag")

model2 = TFAutoModelForSequenceClassification.from_pretrained("../ArticleTagModel1", from_pt=False)
tokenizer2 = AutoTokenizer.from_pretrained("../ArticleTagModel1")

# Manual label mapping
label_map1 = {0: "Positive", 1: "Negative", 2: "Neutral"}
label_map2 = {0: "Sports", 1: "Technology", 2: "Stock Business"}

# Prediction function
def predict_text(input_text):
    # Prediction for model 1 (ArticleTag)
    inputs1 = tokenizer1(input_text, return_tensors="tf", truncation=True, padding=True)
    logits1 = model1(inputs1["input_ids"]).logits
    predicted_label1 = tf.argmax(logits1, axis=1).numpy()[0]
    
    # Prediction for model 2 (New label)
    inputs2 = tokenizer2(input_text, return_tensors="tf", truncation=True, padding=True)
    logits2 = model2(inputs2["input_ids"]).logits
    predicted_label2 = tf.argmax(logits2, axis=1).numpy()[0]
    
    return label_map1[predicted_label1], label_map2[predicted_label2]

# Dictionary to store blogs
blogs = {
    "Virat Kohli": [["The Journey of Virat Kohli: From a Young Talent to Cricket Legend", """Hey everyone,

I'm Virat Kohli, and today I want to take you on a journey through my life and career in cricket. Born on November 5, 1988, in Delhi, I grew up with a bat in hand and a dream in my heart. Cricket has always been my passion, and I knew from a young age that I wanted to make it my life.

I still remember captaining the Indian team to victory in the Under-19 World Cup in 2008. That moment was surreal; it marked the beginning of my journey with the national team. I made my debut against Sri Lanka later that year, and I felt a mix of excitement and nerves. Playing for India was a dream come true.

Throughout my career, I've had the privilege of representing India in countless matches. There have been highs and lows, but my determination to push boundaries has always driven me. I've always believed in chasing targets and never backing down. I take immense pride in being the fastest player to score 10,000 runs in ODIs and leading the team to our historic series win in Australia in 2018-2019.

Leadership has been an incredible journey for me. I strive to inspire my teammates to give their best and play with passion. Seeing my team perform at their peak and achieving our goals together is what I cherish the most.

I've always tried to be consistent, and my records—like having the most centuries in ODIs—reflect my hard work and dedication. But it's not just about personal achievements; it's about how we can come together as a team and make our fans proud.

Thank you all for being part of this journey with me. Your support means the world, and I promise to keep striving for excellence, both on and off the field.

Cheers,

Virat Kohli
""", "Positive", "Sports"]]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/atheletes')
def atheletes():
    return render_template('atheletes.html')

@app.route('/ask_athlete', methods=['POST'])
def ask_athlete():
    data = request.json
    question = data.get('question')
    if question:
        prompt = f"The user is asking a question about athletes: {question}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about athletes."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer}), 200
    else:
        return jsonify({"error": "Question is required!"}), 400

@app.route('/ask', methods=['GET'])
def ask_page():
    return render_template('ask.html')

@app.route('/blog', methods=['GET'])
def blog_page():
    return render_template('blog.html')

@app.route('/get_blogs', methods=['GET'])
def get_blogs():
    return jsonify(blogs)

@app.route('/blog', methods=['POST'])
def submit_blog():
    data = request.json
    author = data.get('author')
    title = data.get('title')
    hashtag = data.get('hashtags')
    content = data.get('content')
    
    if author and title and content:
        # Predict the labels
        label1, label2 = predict_text(content)
        
        # Add blog to the dictionary with the predicted labels
        if author in blogs:
            blogs[author].append([title, content, label1, label2])
        else:
            blogs[author] = [[title, content, label1, label2]]
        
        print(blogs)
        return jsonify({"message": "Blog added successfully!", "label1": label1, "label2": label2}), 200
    else:
        return jsonify({"error": "Author, title, and content are required!"}), 400

@app.route('/ask_blog', methods=['POST'])
def ask_blog():
    data = request.json
    author = data.get('author')
    title = data.get('title')
    question = data.get('question')
    if author and title and question:
        if author in blogs:
            filtered_blogs = [content for t, content, _, _ in blogs[author] if t == title]
            if filtered_blogs:
                blog_content = " ".join(filtered_blogs)
                prompt = f"The user is asking a question about the blog titled '{title}' written by {author}: {blog_content}. The question is: {question}"
            else:
                prompt = f"The user is asking a question about the title '{title}' but no such blog exists by {author}. The question is: {question}"
        else:
            prompt = f"The user is asking a question: {question}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on blog content."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer}), 200
    else:
        return jsonify({"error": "Author, title, and question are required!"}), 400

@app.route('/explore', methods=['GET'])
def explore_page():
    return render_template('explore.html', blogs=blogs)

if __name__ == '__main__':
    app.run(debug=True)