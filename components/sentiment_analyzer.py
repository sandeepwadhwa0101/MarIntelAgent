from typing import List, Dict
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model."""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device="cpu"  # Explicitly set to CPU to avoid CUDA errors
            )
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {str(e)}")
            raise

    def analyze(self, text: str) -> dict:
        """Analyze the sentiment of the given text with enhanced metrics and insights."""
        try:
            if not text:
                return {
                    "label": "neutral",
                    "score": 0.5,
                    "intensity": "low",
                    "detailed_scores": [],
                    "key_metrics": {},
                    "recommendations": ["Please provide text for analysis"]
                }
            
            # Clean text for better analysis
            text = text.strip()
            
            # Get raw sentiment scores
            results = self.classifier(text)
            
            # Extract scores
            positive_score = 0
            negative_score = 0
            
            for result in results:
                if result['label'] == 'POSITIVE':
                    positive_score = result['score']
                else:
                    negative_score = result['score']
            
            # Calculate score difference for intensity
            score_diff = abs(positive_score - negative_score)
            if score_diff > 0.8:
                intensity = "high"
            elif score_diff > 0.4:
                intensity = "medium"
            else:
                intensity = "low"
            
            # Determine final sentiment
            if positive_score > negative_score:
                label = "positive"
                final_score = positive_score
            elif negative_score > positive_score:
                label = "negative"
                final_score = negative_score
            else:
                label = "neutral"
                final_score = 0.5
            
            # Calculate additional metrics
            sentiment_metrics = {
                "confidence": max(positive_score, negative_score),
                "sentiment_ratio": positive_score / (positive_score + negative_score) if (positive_score + negative_score) != 0 else 0,
                "polarization": score_diff,
                "clarity": 1.0 - (min(positive_score, negative_score) * 2)
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                label, final_score, intensity, sentiment_metrics
            )
            
            return {
                "label": label,
                "score": final_score,
                "intensity": intensity,
                "detailed_scores": results,
                "key_metrics": sentiment_metrics,
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "label": "neutral",
                "score": 0.5,
                "intensity": "low",
                "detailed_scores": [],
                "key_metrics": {},
                "recommendations": [f"Error analyzing sentiment: {str(e)}"]
            }

    def analyze_channel(self, text: str, channel: str = "general") -> dict:
        """Analyze sentiment with channel-specific context."""
        # Channel-specific sentiment modifiers
        channel_weights = {
            "twitter": 1.2,  # More emphasis on concise, immediate reactions
            "facebook": 1.0,  # Balanced sentiment analysis
            "reviews": 0.9,  # Slightly reduced weight for review platforms
            "email": 0.95,  # Slightly reduced weight for formal communications
            "general": 1.0   # Default weight
        }
        
        # Get base sentiment analysis
        sentiment = self.analyze(text)
        
        # Apply channel-specific adjustments
        weight = channel_weights.get(channel, 1.0)
        sentiment['score'] = min(1.0, sentiment['score'] * weight)
        sentiment['channel'] = channel
        
        # Add channel-specific metrics
        if channel == "twitter":
            sentiment['virality_potential'] = self._calculate_virality_score(text)
        elif channel == "reviews":
            sentiment['review_quality'] = self._assess_review_quality(text)
            
        return sentiment

    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze(text) for text in texts]

    def _calculate_virality_score(self, text: str) -> float:
        """Calculate potential virality of social media content."""
        viral_indicators = {
            'hashtag': text.count('#'),
            'mention': text.count('@'),
            'length': len(text),
            'engagement_words': len([w for w in text.lower().split() 
                                  if w in {'rt', 'share', 'like', 'follow'}])
        }
        
        # Simple scoring algorithm
        score = (
            min(viral_indicators['hashtag'], 3) * 0.2 +
            min(viral_indicators['mention'], 2) * 0.15 +
            (1.0 if 50 <= viral_indicators['length'] <= 280 else 0.7) * 0.4 +
            min(viral_indicators['engagement_words'], 2) * 0.25
        )
        
        return min(1.0, score)

    def _assess_review_quality(self, text: str) -> float:
        """Assess the quality and helpfulness of a review."""
        quality_metrics = {
            'length': len(text.split()),
            'detail_words': len([w for w in text.lower().split() 
                               if w in {'because', 'however', 'specifically', 'example'}]),
            'formatting': text.count('\n') + text.count('.') + text.count(',')
        }
        
        # Quality scoring algorithm
        score = (
            min(quality_metrics['length'] / 100.0, 1.0) * 0.4 +
            min(quality_metrics['detail_words'] * 0.25, 1.0) * 0.35 +
            min(quality_metrics['formatting'] / 5.0, 1.0) * 0.25
        )
        
        return min(1.0, score)

    def _generate_recommendations(self, label: str, score: float, intensity: str, metrics: dict) -> List[str]:
        """Generate actionable recommendations based on sentiment analysis."""
        recommendations = []
        
        # High-impact recommendations based on sentiment
        if label == "positive" and score > 0.8:
            recommendations.extend([
                "Amplify this messaging across other marketing channels",
                "Use similar tone and content structure in future campaigns",
                f"Leverage high clarity score ({metrics['clarity']:.2f}) in brand voice guidelines"
            ])
        elif label == "negative" and score > 0.6:
            recommendations.extend([
                "Address concerns immediately in communication strategy",
                "Develop crisis management response if sentiment persists",
                "Monitor related topics for sentiment changes"
            ])
        else:
            recommendations.extend([
                "Strengthen message to evoke stronger positive sentiment",
                "A/B test variations of content to find more impactful messaging",
                "Review competitor messaging for differentiation opportunities"
            ])
            
        # Engagement strategy based on metrics
        if metrics["confidence"] > 0.8:
            recommendations.append("High confidence score suggests clear audience resonance")
        if metrics["polarization"] > 0.7:
            recommendations.append("Content is generating strong reactions - consider audience segmentation")
            
        # Add intensity-based recommendations
        if intensity == "high":
            recommendations.append("Strong emotional response detected - consider expanding reach")
        elif intensity == "low":
            recommendations.append("Consider strengthening emotional appeal in messaging")
            
        return recommendations[:5]  # Return top 5 most relevant recommendations
