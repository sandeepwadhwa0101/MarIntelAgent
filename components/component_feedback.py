import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime

class FeedbackContainer:
    def __init__(self, title: str):
        """Initialize feedback container with title."""
        self.container = st.container()
        if title:
            self.container.subheader(title)
        self.feedback_component = ComponentFeedback(
            component_id=f"feedback_{title.lower().replace(' ', '_')}",
            component_type="visualization"
        )

    def __enter__(self):
        """Support context manager protocol."""
        return self.container

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle context manager exit and render feedback."""
        feedback_data = self.feedback_component.render()
        if feedback_data:
            self.container.success("Thank you for your feedback!")

class ComponentFeedback:
    def __init__(self, component_id: str, component_type: str):
        """Initialize feedback component.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (e.g., 'chart', 'insight', 'recommendation')
        """
        self.component_id = component_id
        self.component_type = component_type
        
        # Get ML feedback loop instance from session state
        if 'ml_feedback' not in st.session_state:
            st.session_state.ml_feedback = None
        
    def render(self) -> Optional[Dict[str, Any]]:
        """Render feedback buttons and comment box.
        
        Returns:
            Dictionary containing feedback data if submitted
        """
        cols = st.columns([1, 6, 1, 1])
        
        # Create unique keys for session state
        feedback_key = f"feedback_{self.component_id}"
        comment_key = f"comment_{self.component_id}"
        
        # Initialize session state if needed
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None
            st.session_state[comment_key] = ""
            st.session_state[f"interaction_count_{self.component_id}"] = 0
            st.session_state[f"start_time_{self.component_id}"] = datetime.now().timestamp()

        
        # Render feedback buttons
        with cols[2]:
            if st.button("👍", key=f"up_{self.component_id}"):
                st.session_state[feedback_key] = 1.0
                
        with cols[3]:
            if st.button("👎", key=f"down_{self.component_id}"):
                st.session_state[feedback_key] = 0.0
        
        # Show comment box if feedback given
        feedback_data = None
        if st.session_state[feedback_key] is not None:
            with cols[1]:
                comment = st.text_input(
                    "Add a comment (optional):",
                    key=comment_key,
                    placeholder="What made this helpful or not helpful?"
                )
                
                # Create enriched feedback data
                feedback_data = {
                    "component_id": self.component_id,
                    "component_type": self.component_type,
                    "score": st.session_state[feedback_key],
                    "comment": comment,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "response_time": datetime.now().timestamp() - st.session_state.get(f"start_time_{self.component_id}", datetime.now().timestamp()),
                        "interaction_count": st.session_state.get(f"interaction_count_{self.component_id}", 0) + 1
                    }
                }
                
                # Update interaction metrics in session state
                st.session_state[f"interaction_count_{self.component_id}"] = st.session_state.get(f"interaction_count_{self.component_id}", 0) + 1
                st.session_state[f"start_time_{self.component_id}"] = datetime.now().timestamp()
                
                # Record feedback in ML feedback loop if available
                if st.session_state.ml_feedback:
                    # Record enriched prediction data
                    prediction_id = st.session_state.ml_feedback.record_prediction(
                        prediction_type=self.component_type,
                        prediction={
                            "component": self.component_id,
                            "interaction_metrics": feedback_data["metrics"]
                        }
                    )
                    
                    # Add detailed feedback with metrics
                    st.session_state.ml_feedback.add_feedback(
                        prediction_id=prediction_id,
                        actual_outcome={
                            "feedback": comment if comment else "",
                            "interaction_time": feedback_data["metrics"]["response_time"],
                            "interaction_count": feedback_data["metrics"]["interaction_count"]
                        },
                        feedback_score=st.session_state[feedback_key]
                    )
        
        return feedback_data

def create_feedback_container(title: str) -> FeedbackContainer:
    """Create a container for a component with its feedback buttons."""
    return FeedbackContainer(title)