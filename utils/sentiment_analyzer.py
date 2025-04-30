from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class EmotionAnalyzer:
    def __init__(self):
        # Initialize emotion detection pipeline
        self.emotion_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=3
        )
        
        # Initialize zero-shot classifier for activities
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Initialize sentiment analysis pipeline
        self.sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Activity categories for zero-shot classification
        self.activity_categories = [
            "working", "resting", "entertainment", 
            "socializing", "exercising", "eating",
            "studying", "hobbies", "traveling",
            "shopping", "cleaning", "creating",
            "outdoor activities", "self-care"
        ]

    def analyze_text(self, text: str) -> dict:
        # Get emotion predictions
        emotion_results = self.emotion_classifier(text)
        
        # Get sentiment score
        sentiment_result = self.sentiment_classifier(text)[0]
        sentiment_score = float(sentiment_result['score'])
        if sentiment_result['label'] == 'NEGATIVE':
            sentiment_score = -sentiment_score
        
        # Get activity predictions
        activity_results = self.zero_shot_classifier(
            text,
            self.activity_categories,
            multi_label=True
        )

        # Format emotion details
        emotions = {
            pred['label']: float(pred['score'])
            for pred in emotion_results[0]
        }

        # Create detailed analysis
        analysis = {
            "sentiment_score": sentiment_score,  # Added sentiment score
            "emotion_details": {
                "primary_emotion": max(emotions.items(), key=lambda x: x[1])[0],
                "intensity": max(emotions.values()),
                "secondary_emotions": emotions,
                "confidence": float(np.mean(list(emotions.values()))),
                "mood_indicators": self._get_mood_indicators(emotions)
            },
            "activities": [
                label for label, score in zip(
                    activity_results['labels'],
                    activity_results['scores']
                )
                if score > 0.5
            ]
        }
        
        return analysis

    def _get_mood_indicators(self, emotions: dict) -> list:
        indicators = []
        
        # Group emotions into mood categories
        positive_emotions = ['joy', 'admiration', 'approval', 'gratitude', 'relief', 'love']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'disappointment']
        
        # Calculate overall mood
        positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
        
        if positive_score > negative_score:
            indicators.append("positive")
        elif negative_score > positive_score:
            indicators.append("negative")
        else:
            indicators.append("neutral")
        
        # Add specific mood indicators based on strongest emotions
        top_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        indicators.append(f"predominantly_{top_emotion}")
        
        return indicators