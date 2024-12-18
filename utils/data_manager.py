from typing import List, Dict
import json
import os
from datetime import datetime

class DataManager:
    def __init__(self):
        self.feedback_file = "feedback_data.json"
        self.ensure_feedback_file_exists()

    def ensure_feedback_file_exists(self):
        """Create feedback file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)

    def save_feedback(self, feedback: str, rating: int, category: str = "general"):
        """Save user feedback with enhanced metadata."""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except json.JSONDecodeError:
            feedback_data = []

        # Create enhanced feedback entry with metadata
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback,
            'rating': rating,
            'category': category,
            'word_count': len(feedback.split()),
            'sentiment_score': self._quick_sentiment_check(feedback),
            'metadata': {
                'day_of_week': datetime.now().strftime('%A'),
                'hour_of_day': datetime.now().hour,
                'feedback_length': 'short' if len(feedback) < 50 else 'medium' if len(feedback) < 200 else 'long'
            }
        }
        
        feedback_data.append(feedback_entry)
        
        # Save with error handling
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
            return False

    def get_feedback_data(self) -> List[Dict]:
        """Retrieve all feedback data."""
    def get_feedback_trends(self, days: int = 30) -> Dict:
        """Analyze feedback trends over time."""
        feedback_data = self.get_feedback_data()
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame(feedback_data)
        if df.empty:
            return {
                'average_rating_trend': [],
                'feedback_volume_trend': [],
                'sentiment_trend': [],
                'categories': {}
            }
            
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] > cutoff_date]
        
        # Group by date
        daily_stats = df.groupby(df['timestamp'].dt.date).agg({
            'rating': 'mean',
            'sentiment_score': 'mean',
            'feedback': 'count'
        }).reset_index()
        
        # Calculate category distribution
        category_stats = df['category'].value_counts().to_dict()
        
        return {
            'average_rating_trend': [
                {'date': str(date), 'value': float(rating)}
                for date, rating in zip(daily_stats['timestamp'], daily_stats['rating'])
            ],
            'feedback_volume_trend': [
                {'date': str(date), 'value': int(count)}
                for date, count in zip(daily_stats['timestamp'], daily_stats['feedback'])
            ],
            'sentiment_trend': [
                {'date': str(date), 'value': float(score)}
                for date, score in zip(daily_stats['timestamp'], daily_stats['sentiment_score'])
            ],
            'categories': category_stats
        }
    
    def export_feedback_data(self, format: str = 'json') -> str:
        """Export feedback data in various formats."""
        feedback_data = self.get_feedback_data()
        
        if format.lower() == 'csv':
            output = []
            if feedback_data:
                # Get all possible keys from the feedback entries
                keys = set()
                for entry in feedback_data:
                    keys.update(entry.keys())
                
                # Create CSV header
                header = ','.join(sorted(keys))
                output.append(header)
                
                # Add data rows
                for entry in feedback_data:
                    row = [str(entry.get(key, '')) for key in sorted(keys)]
                    output.append(','.join(row))
                    
                return '\n'.join(output)
        else:
            return json.dumps(feedback_data, indent=2)
    
    def _quick_sentiment_check(self, text: str) -> float:
        """Simple sentiment analysis for feedback categorization."""
        positive_words = {'great', 'good', 'excellent', 'amazing', 'helpful', 'useful', 'love', 'perfect'}
        negative_words = {'bad', 'poor', 'terrible', 'awful', 'useless', 'hate', 'worst', 'difficult'}
        
        words = set(text.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count == 0 and negative_count == 0:
            return 0.5
        
        total = positive_count + negative_count
        return positive_count / total if total > 0 else 0.5
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_average_rating(self) -> float:
        """Calculate average feedback rating."""
        feedback_data = self.get_feedback_data()
        if not feedback_data:
            return 0.0
        
        ratings = [entry['rating'] for entry in feedback_data]
        return sum(ratings) / len(ratings)
