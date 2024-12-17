import streamlit as st
from components.sentiment_analyzer import SentimentAnalyzer
from components.llm_recommender import LLMRecommender
from components.visualization import DataVisualizer
from components.feedback_loop import MLFeedbackLoop
from utils.text_processor import TextProcessor
from utils.data_manager import DataManager


def init_session_state():
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 'Dashboard'
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    if 'ml_feedback' not in st.session_state:
        st.session_state.ml_feedback = None

def main():
    st.set_page_config(
        page_title="Marketing Intelligence Platform",
        page_icon="📊",
        layout="wide"
    )

    # Initialize components
    sentiment_analyzer = SentimentAnalyzer()
    llm_recommender = LLMRecommender()
    data_visualizer = DataVisualizer()
    data_manager = DataManager()
    
    # Initialize ML feedback loop if not already in session state
    if st.session_state.ml_feedback is None:
        st.session_state.ml_feedback = MLFeedbackLoop()

    # Load custom CSS
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Sentiment Analysis", "Recommendations", "Brand Voice", "Feedback"]
    )

    if page == "Dashboard":
        st.title("Marketing Intelligence Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Overview")
            data_visualizer.plot_performance_trends()
            
        with col2:
            st.subheader("Sentiment Distribution")
            data_visualizer.plot_sentiment_distribution()
        
        # Show ML Performance Metrics and Visualization if feedback data exists
        st.subheader("System Performance")
        if st.session_state.ml_feedback:
            # Plot ML performance trends
            sentiment_performance = st.session_state.ml_feedback.get_model_performance("sentiment")
            data_visualizer.plot_ml_performance(sentiment_performance)
            # Get performance metrics for different components
            sentiment_performance = st.session_state.ml_feedback.get_model_performance("sentiment")
            recommendations_performance = st.session_state.ml_feedback.get_model_performance("recommendation")
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Sentiment Analysis Accuracy",
                    f"{sentiment_performance['average_feedback_score']:.2%}",
                    delta=f"{sentiment_performance.get('improvement_rate', 0):.1%}" if sentiment_performance.get('improvement_rate') else None,
                    help="Based on user feedback"
                )
                if sentiment_performance.get('rolling_average'):
                    st.caption(f"Rolling Average: {sentiment_performance['rolling_average']:.2%}")
                
            with col2:
                st.metric(
                    "Recommendation Quality",
                    f"{recommendations_performance['average_feedback_score']:.2%}",
                    delta=f"{recommendations_performance.get('improvement_rate', 0):.1%}" if recommendations_performance.get('improvement_rate') else None,
                    help="Based on user feedback"
                )
                if recommendations_performance.get('rolling_average'):
                    st.caption(f"Rolling Average: {recommendations_performance['rolling_average']:.2%}")
            
            with col3:
                recent_perf = sentiment_performance.get('recent_performance', 0)
                st.metric(
                    "Recent Performance",
                    f"{recent_perf:.2%}",
                    delta="Improving" if recent_perf > sentiment_performance['average_feedback_score'] else "Needs Attention",
                    help="Performance over last 5 feedback entries"
                )
            
            # Show component-specific performance
            if sentiment_performance.get('component_performance'):
                st.subheader("Component Performance")
                for component, metrics in sentiment_performance['component_performance'].items():
                    st.metric(
                        f"{component.replace('_', ' ').title()}",
                        f"{metrics['average_score']:.2%}",
                        help=f"Based on {metrics['count']} feedback entries"
                    )
            
            # Show feedback quality metrics
            if sentiment_performance.get('feedback_quality'):
                st.subheader("Feedback Quality Metrics")
                quality = sentiment_performance['feedback_quality']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Comment Rate",
                        f"{quality['feedback_with_comments']:.1%}",
                        help="Percentage of feedback with comments"
                    )
                with col2:
                    st.metric(
                        "Response Time",
                        f"{quality['average_response_time']:.1f}s",
                        help="Average time taken to provide feedback"
                    )
                with col3:
                    st.metric(
                        "Interaction Depth",
                        f"{quality['interaction_depth']:.1f}",
                        help="Average number of interactions per feedback"
                    )
                with col4:
                    st.metric(
                        "Consistency Score",
                        f"{quality['feedback_consistency']:.1%}",
                        help="Consistency of feedback scores"
                    )
                
                # Show system health status
                status_color = "🟢" if sentiment_performance.get('health_status') == "good" else "🟡"
                st.info(f"{status_color} System Health: {sentiment_performance.get('health_status', 'unknown').replace('_', ' ').title()}")
            
            # Show performance predictions
            st.subheader("Performance Predictions")
            predictions = st.session_state.ml_feedback.predict_performance("sentiment")
            data_visualizer.plot_performance_prediction(predictions)
            
            # Show improvement suggestions
            suggestions = st.session_state.ml_feedback.get_improvement_suggestions("sentiment")
            if suggestions:
                st.subheader("Improvement Insights")
                for suggestion in suggestions:
                    st.info(suggestion)
        
        st.subheader("Recent Recommendations")
        recommendations = llm_recommender.get_recent_recommendations()
        for rec in recommendations:
            st.info(rec)

    elif page == "Sentiment Analysis":
        st.title("Multi-Channel Sentiment Analysis")
        
        # Channel selection
        channel = st.selectbox(
            "Select Channel",
            ["General", "Twitter", "Facebook", "Reviews", "Email"],
            help="Choose the channel for context-aware sentiment analysis"
        )
        
        text_input = st.text_area("Enter text for analysis:")
        if st.button("Analyze Sentiment"):
            # Get channel-specific analysis
            sentiment = sentiment_analyzer.analyze_channel(text_input, channel.lower())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Sentiment Score:", sentiment['score'])
                st.write("Sentiment Label:", sentiment['label'])
                st.write("Intensity:", sentiment['intensity'])
                data_visualizer.plot_sentiment_details(sentiment)
            
            with col2:
                st.subheader("Channel-Specific Insights")
                if channel.lower() == "twitter":
                    st.metric("Virality Potential", f"{sentiment.get('virality_potential', 0):.2f}")
                elif channel.lower() == "reviews":
                    st.metric("Review Quality", f"{sentiment.get('review_quality', 0):.2f}")
                
                if 'recommendations' in sentiment:
                    st.subheader("Recommendations")
                    for rec in sentiment['recommendations']:
                        st.info(rec)
                
                if 'key_metrics' in sentiment:
                    st.subheader("Key Metrics")
                    metrics = sentiment['key_metrics']
                    for metric, value in metrics.items():
                        st.metric(metric.title(), f"{value:.2f}")

    elif page == "Recommendations":
        st.title("LLM-Powered Recommendations")
        
        recommendation_type = st.radio(
            "Select Recommendation Type",
            ["General Marketing", "Crisis Response"]
        )
        
        if recommendation_type == "General Marketing":
            context = st.text_area("Enter marketing context:")
            use_examples = st.checkbox("Use previous successful examples", value=True)
            
            if st.button("Generate Recommendations"):
                examples = llm_recommender.get_example_cases() if use_examples else None
                recommendations = llm_recommender.generate_recommendations(context, examples)
                
                st.subheader("Generated Recommendations")
                for idx, rec in enumerate(recommendations, 1):
                    st.write(f"{idx}. {rec}")
                    
        else:  # Crisis Response
            negative_context = st.text_area("Describe the negative situation:")
            if st.button("Generate Crisis Response Plan"):
                response = llm_recommender.generate_crisis_response(negative_context)
                
                if "error" not in response:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("Immediate Action Required")
                        st.write(response["immediate_action"])
                        
                        st.warning("Root Cause")
                        st.write(response["root_cause"])
                        
                    with col2:
                        st.success("Communication Plan")
                        st.write(response["communication_plan"])
                        
                        st.info("Prevention Strategy")
                        st.write(response["prevention"])
                    
                    st.subheader("Corrective Actions")
                    st.write(response["corrective_actions"])
                else:
                    st.error(f"Error generating response: {response['error']}")

    elif page == "Brand Voice":
        st.title("Brand Voice Analysis")
        
        # Add tabs for different input methods
        input_method = st.radio(
            "Select Input Method",
            ["Upload Content", "Direct Input"]
        )
        
        content = None
        if input_method == "Upload Content":
            uploaded_file = st.file_uploader("Upload brand content:", type=['txt'])
            if uploaded_file:
                content = uploaded_file.read().decode()
        else:
            content = st.text_area(
                "Enter your brand content:",
                height=200,
                help="Paste your marketing content, social media posts, or other brand communications here."
            )
            
        if content:
            st.subheader("Analysis Results")
            
            # Analyze brand voice
            analysis = llm_recommender.analyze_brand_voice(content)
            
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Show radar chart
                data_visualizer.plot_brand_voice_analysis(analysis)
                
            with col2:
                # Show metrics with color-coded scores
                st.subheader("Voice Metrics")
                for char, score in analysis.items():
                    delta = None
                    if score > 0.7:
                        delta = "Strong"
                    elif score < 0.3:
                        delta = "Needs Focus"
                    st.metric(
                        char,
                        f"{score:.2f}",
                        delta=delta
                    )
            
            # Add insights section
            st.subheader("Key Insights")
            insights = []
            for char, score in analysis.items():
                if score > 0.7:
                    insights.append(f"💪 Strong {char.lower()} voice")
                elif score < 0.3:
                    insights.append(f"🎯 Opportunity to strengthen {char.lower()}")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("Your brand voice shows balanced characteristics across all dimensions.")

    elif page == "Feedback":
        st.title("User Feedback Collection")
        feedback = st.text_area("Share your feedback:")
        rating = st.slider("Rate your experience:", 1, 5, 3)
        if st.button("Submit Feedback"):
            # Save user feedback
            data_manager.save_feedback(feedback, rating)
            
            
            
            st.success("Thank you for your feedback! Your input helps improve our system.")
            
    

if __name__ == "__main__":
    init_session_state()
    main()
