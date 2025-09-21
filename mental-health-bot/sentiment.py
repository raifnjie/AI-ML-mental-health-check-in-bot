from typing import Dict, Tuple

class SentimentAnalyzer:
    def __init__(self, use_transformers: bool = False):
        self.use_transformers = use_transformers
        if use_transformers:
            from transformers import pipeline
            self.pipe = pipeline("sentiment-analysis")
        else:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict:
        text = text or ""
        if self.use_transformers:
            res = self.pipe(text)[0]  # {'label':'POSITIVE','score':0.98}
            label = res['label']
            # map POSITIVE -> +score, NEGATIVE -> -score so score ~ [-1,1]
            score = res['score'] if label.upper().startswith("POS") else -res['score']
            return {"label": label, "score": float(score)}
        else:
            vs = self.analyzer.polarity_scores(text)
            # vs['compound'] is in [-1,1]
            label = "POSITIVE" if vs['compound'] >= 0 else "NEGATIVE"
            return {"label": label, "score": float(vs['compound'])}

# Global analyzer instance
_analyzer = SentimentAnalyzer()

def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Analyze sentiment of text and return score and label.
    This function is called by the routes.
    """
    result = _analyzer.analyze(text)
    return result['score'], result['label']