from flask import Flask, request, jsonify, render_template, redirect, url_for
from sentiment import analyze_sentiment
from utils import save_entry, load_entries
from joblib import load

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

# Web interface route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('entry')
        if text and text.strip():
            score, label = analyze_sentiment(text)
            save_entry(text, label, score)
        return redirect(url_for('index'))
    
    entries = load_entries()
    return render_template('index.html', entries=entries)

# API routes
@app.route('/journal', methods=['POST'])
def journal_entry():
    data = request.json
    entry_text = data.get('text', '')
    
    if not entry_text.strip():
        return jsonify({"error": "Entry text cannot be empty"}), 400

    sentiment_score, sentiment_label = analyze_sentiment(entry_text)
    save_entry(entry_text, sentiment_label, sentiment_score)

    return jsonify({
        "text": entry_text,
        "sentiment": sentiment_label,
        "score": sentiment_score
    })

@app.route('/summary', methods=['GET'])
def summary():
    entries = load_entries()
    return jsonify({'entries': entries})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message.strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    # Use an AI model to generate a response
    ai_response = generate_chat_response(user_message)

    # Save the chat to the journal
    save_entry(user_message, "chat", 0)  # Label as "chat" with neutral score
    save_entry(ai_response, "ai_response", 0)

    return jsonify({"user_message": user_message, "ai_response": ai_response})

emotion_model = load('emotion_model.pkl')
vectorizer = load('vectorizer.pkl')
def analyze_emotional_patterns(entries):
    texts = [entry['text'] for entry in entries]
    X = vectorizer.transform(texts)
    predictions = emotion_model.predict(X)
    return predictions

@app.route('/report', methods=['GET'])
def report():
    entries = load_entries()
    patterns = analyze_emotional_patterns(entries)

    # Generate a summary report
    report = {
        "total_entries": len(entries),
        "positive_count": patterns.count("positive"),
        "negative_count": patterns.count("negative"),
        "neutral_count": patterns.count("neutral"),
    }
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)