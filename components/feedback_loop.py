from typing import Dict, List, Optional
import os
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PredictionFeedback:
    prediction_id: str
    prediction_type: str  # 'sentiment', 'recommendation', 'brand_voice'
    prediction: Dict
    actual_outcome: Optional[Dict] = None
    feedback_score: Optional[float] = None
    timestamp: str = datetime.now().isoformat()

class MLFeedbackLoop:
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "ml_feedback.json"
        self._initialize_feedback_store()
    
    def _initialize_feedback_store(self):
        """Initialize the feedback storage file if it doesn't exist."""
        if not self.feedback_file.exists():
            self.feedback_file.write_text(json.dumps({"feedback": []}))
    
    def record_prediction(self, prediction_type: str, prediction: Dict) -> str:
        """Record a new prediction and return its ID for later feedback."""
        feedback_entry = PredictionFeedback(
            prediction_id=f"{prediction_type}_{datetime.now().timestamp()}",
            prediction_type=prediction_type,
            prediction=prediction
        )
        
        self._save_feedback(feedback_entry)
        return feedback_entry.prediction_id
    
    def add_feedback(self, prediction_id: str, actual_outcome: Dict, feedback_score: float) -> bool:
        """Add feedback for a specific prediction."""
        try:
            data = self._load_feedback()
            
            # Find and update the prediction entry
            for entry in data["feedback"]:
                if entry["prediction_id"] == prediction_id:
                    entry["actual_outcome"] = actual_outcome
                    entry["feedback_score"] = feedback_score
                    entry["feedback_timestamp"] = datetime.now().isoformat()
                    
                    self.feedback_file.write_text(json.dumps(data, indent=2))
                    return True
            
            return False
        except Exception as e:
            print(f"Error adding feedback: {str(e)}")
            return False
    
    def get_model_performance(self, prediction_type: str = None) -> Dict:
        """Calculate detailed model performance metrics based on feedback."""
        data = self._load_feedback()
        feedback_entries = data["feedback"]
        
        if prediction_type:
            feedback_entries = [f for f in feedback_entries if f["prediction_type"] == prediction_type]
        
        if not feedback_entries:
            return self._get_empty_performance_metrics()
        
        # Calculate basic metrics
        feedback_scores = [f["feedback_score"] for f in feedback_entries if "feedback_score" in f]
        if not feedback_scores:
            return self._get_empty_performance_metrics()
        
        # Calculate performance trend
        timestamps = [datetime.fromisoformat(f["timestamp"]) for f in feedback_entries if "feedback_score" in f]
        performance_trend = list(zip(timestamps, feedback_scores))
        performance_trend.sort(key=lambda x: x[0])
        
        # Calculate rolling averages
        window_size = 5
        rolling_scores = []
        if len(feedback_scores) >= window_size:
            for i in range(len(feedback_scores) - window_size + 1):
                window = feedback_scores[i:i+window_size]
                rolling_scores.append(sum(window) / window_size)
        
        # Calculate component-specific metrics
        component_performance = {}
        for entry in feedback_entries:
            if "prediction" in entry and "component" in entry["prediction"]:
                component = entry["prediction"]["component"]
                if component not in component_performance:
                    component_performance[component] = []
                if "feedback_score" in entry:
                    component_performance[component].append(entry["feedback_score"])
        
        component_metrics = {
            comp: {
                "average_score": np.mean(scores),
                "count": len(scores)
            }
            for comp, scores in component_performance.items()
        }
        
        # Get feedback quality metrics
        quality_metrics = self.analyze_feedback_quality(feedback_entries)
        
        return {
            "average_feedback_score": np.mean(feedback_scores),
            "feedback_count": len(feedback_entries),
            "performance_trend": [
                {"timestamp": ts.isoformat(), "score": score}
                for ts, score in performance_trend
            ],
            "rolling_average": rolling_scores[-1] if rolling_scores else None,
            "improvement_rate": (rolling_scores[-1] - rolling_scores[0]) if len(rolling_scores) > 1 else None,
            "component_performance": component_metrics,
            "recent_performance": np.mean(feedback_scores[-5:]) if len(feedback_scores) >= 5 else np.mean(feedback_scores),
            "feedback_quality": quality_metrics,
            "health_status": "good" if (
                quality_metrics["feedback_consistency"] > 0.7 and
                quality_metrics["feedback_with_comments"] > 0.3 and
                np.mean(feedback_scores) > 0.6
            ) else "needs_improvement"
        }
    
    def _get_empty_performance_metrics(self) -> Dict:
        """Return empty performance metrics structure."""
        return {
            "average_feedback_score": 0.0,
            "feedback_count": 0,
            "performance_trend": [],
            "rolling_average": None,
            "improvement_rate": None,
            "component_performance": {},
            "recent_performance": 0.0
        }
    
    def analyze_feedback_quality(self, feedback_entries: List[Dict]) -> Dict:
        """Analyze the quality of collected feedback."""
        quality_metrics = {
            "feedback_with_comments": 0,
            "average_response_time": 0,
            "interaction_depth": 0,
            "feedback_consistency": 0
        }
        
        if not feedback_entries:
            return quality_metrics
            
        comment_count = 0
        response_times = []
        interaction_counts = []
        previous_scores = []
        
        for entry in feedback_entries:
            # Check for comments
            if entry.get("actual_outcome", {}).get("feedback"):
                comment_count += 1
            
            # Collect response times
            if "interaction_time" in entry.get("actual_outcome", {}):
                response_times.append(entry["actual_outcome"]["interaction_time"])
            
            # Track interaction depth
            if "interaction_count" in entry.get("actual_outcome", {}):
                interaction_counts.append(entry["actual_outcome"]["interaction_count"])
            
            # Track score consistency
            if "feedback_score" in entry:
                previous_scores.append(entry["feedback_score"])
        
        # Calculate metrics
        quality_metrics["feedback_with_comments"] = comment_count / len(feedback_entries) if feedback_entries else 0
        quality_metrics["average_response_time"] = np.mean(response_times) if response_times else 0
        quality_metrics["interaction_depth"] = np.mean(interaction_counts) if interaction_counts else 0
        
        # Calculate score consistency (lower variance = more consistent)
        if len(previous_scores) > 1:
            quality_metrics["feedback_consistency"] = 1.0 - np.std(previous_scores)
        
        return quality_metrics

    def get_improvement_suggestions(self, prediction_type: str) -> List[str]:
        """Generate improvement suggestions based on feedback patterns and quality metrics."""
        performance = self.get_model_performance(prediction_type)
        suggestions = []
        
        if performance["feedback_count"] < 10:
            suggestions.append("Collect more feedback to improve model accuracy")
            return suggestions
        
        avg_score = performance["average_feedback_score"]
        recent_performance = performance.get("recent_performance", 0)
        rolling_average = performance.get("rolling_average", 0)
        
        # Performance threshold analysis
        if avg_score < 0.7:
            suggestions.append("Model performance needs improvement - consider retraining")
            suggestions.append("Review feedback patterns for potential biases")
        
        # Trend analysis
        trend = performance["performance_trend"]
        if len(trend) >= 2:
            recent_scores = [float(t["score"]) for t in trend[-5:]]
            recent_mean = np.mean(recent_scores)
            
            if recent_mean < avg_score:
                suggestions.append("Recent performance declining - investigate potential causes")
                if recent_mean < 0.5:
                    suggestions.append("Critical attention needed - performance significantly below target")
            
            # Volatility check
            score_std = np.std(recent_scores)
            if score_std > 0.2:
                suggestions.append("High performance volatility detected - stabilize model behavior")
        
        # Component-specific analysis
        component_metrics = performance.get("component_performance", {})
        for component, metrics in component_metrics.items():
            if metrics["average_score"] < 0.6:
                suggestions.append(f"Improve {component.replace('_', ' ')} component performance")
                
            if metrics["count"] < 5:
                suggestions.append(f"Collect more feedback for {component.replace('_', ' ')}")
        
        # Rolling average analysis
        if rolling_average is not None and avg_score > 0:
            trend_strength = (rolling_average - avg_score) / avg_score
            if trend_strength < -0.1:
                suggestions.append("Negative trend detected - review recent changes")
            elif trend_strength > 0.1:
                suggestions.append("Positive trend - consider expanding successful approaches")
        
        return suggestions[:5]  # Return top 5 most relevant suggestions
    
    def _save_feedback(self, feedback_entry: PredictionFeedback):
        """Save a new feedback entry to storage."""
        data = self._load_feedback()
        data["feedback"].append({
            "prediction_id": feedback_entry.prediction_id,
            "prediction_type": feedback_entry.prediction_type,
            "prediction": feedback_entry.prediction,
            "actual_outcome": feedback_entry.actual_outcome,
            "feedback_score": feedback_entry.feedback_score,
            "timestamp": feedback_entry.timestamp
        })
        
        self.feedback_file.write_text(json.dumps(data, indent=2))
    
    def _load_feedback(self) -> Dict:
        """Load feedback data from storage."""
        try:
            return json.loads(self.feedback_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {"feedback": []}

    def predict_performance(self, prediction_type: str = None, horizon: int = 5) -> Dict:
        """Predict future performance metrics based on historical patterns.
        
        Args:
            prediction_type: Type of prediction to analyze ('sentiment', 'recommendation', etc.)
            horizon: Number of time periods to predict into the future (default: 5 days)
            
        Returns:
            Dictionary containing predicted metrics and confidence scores
            
        Raises:
            ValueError: If horizon is not a positive integer
        """
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        data = self._load_feedback()
        feedback_entries = data["feedback"]
        
        if prediction_type:
            feedback_entries = [f for f in feedback_entries if f["prediction_type"] == prediction_type]
            
        if len(feedback_entries) < 3:  # Need minimum data for prediction
            return {
                "predictions": [],
                "confidence": 0.0,
                "message": "Insufficient historical data for prediction"
            }
            
        # Extract historical scores and timestamps
        historical_data = [
            (datetime.fromisoformat(entry["timestamp"]), entry.get("feedback_score", 0))
            for entry in feedback_entries
            if "feedback_score" in entry
        ]
        historical_data.sort(key=lambda x: x[0])
        
        if not historical_data:
            return {
                "predictions": [],
                "confidence": 0.0,
                "message": "No feedback scores available for prediction"
            }
            
        # Calculate basic trend
        scores = [score for _, score in historical_data]
        timestamps = [ts for ts, _ in historical_data]
        
        if len(scores) < 2:
            return {
                "predictions": [scores[0]] * horizon,
                "confidence": 0.3,
                "message": "Limited data, using simple projection"
            }
            
        # Calculate trend using simple moving average
        window = min(3, len(scores))
        moving_avg = []
        for i in range(len(scores) - window + 1):
            moving_avg.append(sum(scores[i:i+window]) / window)
        
        # Calculate trend direction and strength
        trend = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
        
        # Generate predictions
        last_score = scores[-1]
        predictions = []
        for i in range(horizon):
            predicted_score = min(1.0, max(0.0, last_score + trend * (i + 1)))
            predictions.append(predicted_score)
        
        # Calculate prediction confidence
        variance = np.var(scores)
        trend_stability = 1.0 / (1.0 + variance)  # Higher variance = lower stability
        data_confidence = min(1.0, len(scores) / 10.0)  # More data = higher confidence
        overall_confidence = trend_stability * data_confidence
        
        # Generate prediction insights
        trend_strength = abs(trend)
        trend_direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
        
        return {
            "predictions": predictions,
            "timestamps": [timestamps[-1] + timedelta(days=i+1) for i in range(horizon)],
            "confidence": overall_confidence,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "message": f"Performance trend is {trend_direction} with {overall_confidence:.1%} confidence"
        }