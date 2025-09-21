from flask import Blueprint, request, jsonify
from ..sentiment import analyze_sentiment
from ..utils import save_entry, load_entries

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.route('/journal', methods=['POST'])
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
        "score": sentiment_score,
        "message": "Entry saved successfully"
    })

@api_bp.route('/summary', methods=['GET'])
def summary():
    entries = load_entries()
    return jsonify({'entries': entries})