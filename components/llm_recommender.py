from typing import Dict, List, Optional
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch
import json
from pathlib import Path

class LLMRecommender:
    def __init__(self):
        """Initialize the LLM recommender with RAG capabilities."""
        try:
            self.generator = pipeline('text-generation', model='gpt2')
            self.recent_recommendations = []
            self.feedback_scores = []
            
            # Initialize RAG components
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.knowledge_base = self._initialize_knowledge_base()
            self.vector_store = self._initialize_vector_store()
            
        except Exception as e:
            print(f"Error initializing LLM recommender: {str(e)}")
            raise
            
    def _initialize_knowledge_base(self) -> List[Dict]:
        """Initialize knowledge base with marketing examples and best practices."""
        try:
            kb_file = Path("data/knowledge_base.json")
            if kb_file.exists():
                return json.loads(kb_file.read_text())
            else:
                # Default knowledge base
                kb = [
                    {
                        "context": "social media engagement strategies",
                        "content": "Video content increases engagement by 48%. Live streams get 6x more interactions.",
                        "category": "social_media"
                    },
                    {
                        "context": "customer feed    back response",
                        "content": "Quick response within 1 hour increases customer satisfaction by 33%.",
                        "category": "customer_service"
                    },
                    {
                        "context": "brand awareness techniques",
                        "content": "Consistent brand messaging across channels increases recognition by 23%.",
                        "category": "branding"
                    }
                ]
                kb_file.parent.mkdir(exist_ok=True)
                kb_file.write_text(json.dumps(kb, indent=2))
                return kb
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            return []
            
    def _initialize_vector_store(self) -> faiss.IndexFlatL2:
        """Initialize FAISS vector store for similarity search."""
        try:
            # Create vector store
            vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(vector_dimension)
            
            # Add knowledge base embeddings
            if self.knowledge_base:
                texts = [f"{item['context']}: {item['content']}" for item in self.knowledge_base]
                embeddings = self.embedding_model.encode(texts)
                index.add(np.array(embeddings).astype('float32'))
            
            return index
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            return None

    def generate_recommendations(self, context: str, examples: List[Dict[str, str]] = None) -> List[str]:
        """Generate marketing recommendations using RAG and Chain of Thought reasoning."""
        try:
            if not context:
                return ["Please provide context for recommendations"]

            # Input validation
            if len(context) < 10:
                return ["Please provide more detailed context for better recommendations"]
                
            # Get performance metrics to adapt generation
            performance = self.get_recommendation_performance()
            
            # Retrieve relevant knowledge using RAG
            relevant_context = self._retrieve_relevant_knowledge(context)
            
            # Initialize generation parameters
            temperature = 0.7  # Default temperature
            top_p = 0.95      # Default top_p
            
            # Adapt parameters based on feedback performance
            if performance['total_feedback'] > 0:
                if performance['recent_trend'] == 'improving':
                    pass  # Keep current successful parameters
                elif performance['recent_trend'] == 'declining':
                    temperature -= 0.1
                    top_p -= 0.1
                
                if performance['average_score'] < 0.5:
                    temperature = max(0.3, temperature - 0.2)
                    top_p = max(0.7, top_p - 0.15)
                elif performance['average_score'] > 0.8:
                    temperature = min(0.9, temperature + 0.1)
                    top_p = min(0.98, top_p + 0.05)

            # Create Chain of Thought prompt
            prompt = self._create_chain_of_thought_prompt(context, relevant_context, examples)
            
            # Generate recommendations with Chain of Thought reasoning
            generated_text = self.generator(
                prompt,
                max_length=500,  # Increased for more detailed output
                num_return_sequences=1,
                temperature=0.8,  # Slightly increased for more creative outputs
                do_sample=True,
                top_p=0.95,
                no_repeat_ngram_size=2  # Reduced to allow more flexibility
            )
            
            # Process and extract recommendations from Chain of Thought
            raw_text = generated_text[0]['generated_text']
            recommendations = self._extract_recommendations_from_chain_of_thought(raw_text)
            
            if not recommendations:
                return [
                    "Unable to generate specific recommendations. Please try with different context."
                ]
            
            self.recent_recommendations = recommendations[:3]
            
            # Store recommendations for feedback tracking
            if not hasattr(self, 'recommendation_history'):
                self.recommendation_history = []
            self.recommendation_history.append({
                'context': context,
                'recommendations': self.recent_recommendations,
                'parameters': {
                    'temperature': temperature,
                    'top_p': top_p
                }
            })
            
            return self.recent_recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations. Please try again."]

    def generate_crisis_response(self, negative_context: str) -> Dict[str, str]:
        """Generate a crisis response plan."""
        try:
            if not negative_context:
                return {"error": "Please provide context for crisis response"}

            # Generate comprehensive response plan
            prompt = f"Crisis situation: {negative_context}\nGenerate crisis response plan:"
            response = self.generator(
                prompt,
                max_length=300,
                num_return_sequences=1,
                temperature=0.6,
                do_sample=True
            )

            # Parse and structure the response
            raw_text = response[0]['generated_text']
            response_parts = raw_text.split('\n')

            return {
                "immediate_action": self._extract_section(response_parts, "immediate"),
                "root_cause": self._extract_section(response_parts, "cause"),
                "communication_plan": self._extract_section(response_parts, "communication"),
                "corrective_actions": self._extract_section(response_parts, "action"),
                "prevention": self._extract_section(response_parts, "prevent")
            }
        except Exception as e:
            return {"error": str(e)}

    def _retrieve_relevant_knowledge(self, query: str) -> List[Dict]:
        """Retrieve relevant knowledge entries using RAG."""
        try:
            # Encode query to vector
            query_vector = self.embedding_model.encode([query])[0]
            
            # Search similar contexts in vector store
            k = 3  # Number of relevant entries to retrieve
            D, I = self.vector_store.search(np.array([query_vector]).astype('float32'), k)
            
            # Get corresponding knowledge entries
            relevant_entries = []
            for idx in I[0]:
                if idx < len(self.knowledge_base):
                    relevant_entries.append(self.knowledge_base[idx])
            
            return relevant_entries
        except Exception as e:
            print(f"Error retrieving relevant knowledge: {str(e)}")
            return []

    def _create_chain_of_thought_prompt(self, context: str, relevant_knowledge: List[Dict], examples: Optional[List[Dict[str, str]]] = None) -> str:
        """Create a structured prompt incorporating Chain of Thought reasoning."""
        prompt = f"""Given the marketing context: {context}

Step 1 - Analyze Available Knowledge:
{self._format_relevant_knowledge(relevant_knowledge)}

Step 2 - Consider Key Factors:
1. Target audience needs and behaviors
2. Current market trends and opportunities
3. Available resources and limitations
4. Expected outcomes and success metrics

Step 3 - Generate Recommendations:
Based on the above analysis, provide specific, actionable marketing recommendations.
Each recommendation should be clear and implementable.

Recommendations:
"""
        
        if examples:
            prompt += "\nSuccessful Examples for Reference:\n"
            for ex in examples:
                prompt += f"- Context: {ex['context']}\n  Result: {ex['outcome']}\n"
        
        return prompt

    def _format_relevant_knowledge(self, knowledge_entries: List[Dict]) -> str:
        """Format retrieved knowledge entries for the prompt."""
        if not knowledge_entries:
            return "No directly relevant past examples found."
            
        formatted = ""
        for entry in knowledge_entries:
            formatted += f"- {entry['content']} (Category: {entry['category']})\n"
        return formatted

    def _extract_recommendations_from_chain_of_thought(self, raw_text: str) -> List[str]:
        """Extract final recommendations from the Chain of Thought output."""
        try:
            # Get the text after the last occurrence of "Recommendations:"
            if "Recommendations:" in raw_text:
                recommendations_text = raw_text.split("Recommendations:")[-1]
            else:
                recommendations_text = raw_text

            # Split into lines and clean
            lines = recommendations_text.split('\n')
            recommendations = []
            
            current_recommendation = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Start of new recommendation
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                    if current_recommendation:
                        recommendations.append(' '.join(current_recommendation))
                    current_recommendation = [line.lstrip('123456789.- •')]
                else:
                    current_recommendation.append(line)
            
            # Add the last recommendation if exists
            if current_recommendation:
                recommendations.append(' '.join(current_recommendation))
            
            # Filter and clean recommendations
            cleaned_recommendations = []
            for rec in recommendations:
                rec = rec.strip()
                if rec and len(rec) > 20:  # Ensure meaningful content
                    cleaned_recommendations.append(rec)
            
            return cleaned_recommendations[:5] if cleaned_recommendations else ["Implement targeted social media campaign based on audience analysis"]
            
        except Exception as e:
            print(f"Error extracting recommendations: {str(e)}")
            return ["Focus on data-driven marketing strategies"]

    def _extract_section(self, response_parts: List[str], keyword: str) -> str:
        """Extract relevant section from response based on keyword."""
        for part in response_parts:
            if keyword.lower() in part.lower():
                return part.strip()
        return "Not specified in response"

    def analyze_brand_voice(self, content: str) -> Dict[str, float]:
        """Analyze brand voice characteristics with enhanced metrics."""
        if not content:
            return {
                "Professionalism": 0.0,
                "Friendliness": 0.0,
                "Innovation": 0.0,
                "Authority": 0.0,
                "Engagement": 0.0,
                "Consistency": 0.0,
                "Authenticity": 0.0
            }
        
        # Enhanced characteristics keywords with weights
        keywords = {
            "Professionalism": {
                "primary": ["professional", "expertise", "quality", "reliable", "excellence"],
                "secondary": ["efficient", "competent", "skilled", "precise"],
                "weight": 1.2
            },
            "Friendliness": {
                "primary": ["welcome", "help", "support", "care", "understand"],
                "secondary": ["friendly", "kind", "warm", "approachable"],
                "weight": 1.0
            },
            "Innovation": {
                "primary": ["innovative", "new", "advanced", "transform", "future"],
                "secondary": ["creative", "pioneering", "cutting-edge", "revolutionary"],
                "weight": 1.1
            },
            "Authority": {
                "primary": ["leader", "expert", "proven", "trusted", "established"],
                "secondary": ["authoritative", "respected", "recognized", "certified"],
                "weight": 1.15
            },
            "Engagement": {
                "primary": ["interact", "engage", "participate", "connect", "share"],
                "secondary": ["collaborate", "join", "discuss", "explore"],
                "weight": 1.0
            },
            "Consistency": {
                "primary": ["always", "consistently", "reliable", "steady", "stable"],
                "secondary": ["dependable", "constant", "regular", "uniform"],
                "weight": 0.9
            },
            "Authenticity": {
                "primary": ["genuine", "authentic", "real", "transparent", "honest"],
                "secondary": ["sincere", "truthful", "direct", "straightforward"],
                "weight": 1.05
            }
        }
        
        content = content.lower()
        words = content.split()
        total_words = len(words)
        scores = {}
        
        for trait, trait_data in keywords.items():
            # Calculate primary and secondary keyword matches
            primary_matches = sum(content.count(word) for word in trait_data["primary"])
            secondary_matches = sum(content.count(word) for word in trait_data["secondary"])
            
            # Weight the matches
            weighted_score = (
                (primary_matches * 1.0 + secondary_matches * 0.5) * 
                trait_data["weight"] / 
                (total_words / 10)
            )
            
            # Normalize score to [0, 1] range
            scores[trait] = min(weighted_score, 1.0)
            
        # Calculate additional metrics
        sentence_lengths = [len(s.split()) for s in content.split('.') if s.strip()]
        consistency_factor = 1.0 - (max(sentence_lengths) - min(sentence_lengths)) / max(sentence_lengths) if sentence_lengths else 0
        scores["Consistency"] = min(scores["Consistency"] * 0.7 + consistency_factor * 0.3, 1.0)
        
        return scores

    def get_voice_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on brand voice analysis."""
        recommendations = []
        avg_score = sum(scores.values()) / len(scores)
        
        # Overall recommendations
        if avg_score < 0.4:
            recommendations.append("Consider developing a more distinctive brand voice across all dimensions")
        elif avg_score > 0.8:
            recommendations.append("Maintain current strong brand voice while monitoring consistency")
            
        # Trait-specific recommendations
        for trait, score in scores.items():
            if score < 0.3:
                recommendations.append(f"Strengthen {trait.lower()} in communications by incorporating more relevant language and tone")
            elif score < 0.5:
                recommendations.append(f"Moderate improvement needed in {trait.lower()} expression")
            elif score > 0.8:
                recommendations.append(f"Strong {trait.lower()} presence - consider sharing best practices")
                
        # Balance check
        score_variance = np.var(list(scores.values()))
        if score_variance > 0.1:
            recommendations.append("Work on balancing voice characteristics for more consistent brand identity")
            
        return recommendations[:5]  # Return top 5 most relevant recommendations

    def get_example_cases(self) -> List[Dict[str, str]]:
        """Return pre-defined example cases for few-shot learning."""
        return [
            {
                "context": "Need to improve social media engagement",
                "outcome": "Implemented video content strategy resulting in 200% engagement increase"
            },
            {
                "context": "Customer feedback response time is slow",
                "outcome": "Deployed AI chatbot reducing response time by 80%"
            },
            {
                "context": "Brand awareness is low in new market",
                "outcome": "Launched influencer partnership program leading to 150% brand mention increase"
            }
        ]

    def get_recent_recommendations(self) -> List[str]:
        """Get the most recent recommendations generated."""
        return self.recent_recommendations if self.recent_recommendations else [
            "No recent recommendations available"
        ]

    def add_feedback(self, feedback_score: float):
        """Record feedback score for recommendations."""
        if not hasattr(self, 'feedback_scores'):
            self.feedback_scores = []
        self.feedback_scores.append(feedback_score)
        
        # Keep only recent feedback scores
        self.feedback_scores = self.feedback_scores[-10:]  # Keep last 10 scores
    
    def get_recommendation_performance(self) -> Dict:
        """Get performance metrics for recommendations."""
        if not hasattr(self, 'feedback_scores') or len(self.feedback_scores) == 0:
            return {
                'average_score': 0.0,
                'total_feedback': 0,
                'recent_trend': 'neutral'
            }
            
        avg_score = sum(self.feedback_scores) / len(self.feedback_scores)
        recent_scores = self.feedback_scores[-3:]  # Last 3 scores
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        
        return {
            'average_score': avg_score,
            'total_feedback': len(self.feedback_scores),
            'recent_trend': 'improving' if recent_avg > avg_score else 'declining'
        }