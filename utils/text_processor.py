import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]

    def get_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key words from text."""
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Get frequency distribution
        freq_dist = nltk.FreqDist(tokens)
        
        # Return top N keywords
        return [word for word, _ in freq_dist.most_common(top_n)]
