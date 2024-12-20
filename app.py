import streamlit as st
from autogen_collaboration import improve_strategies, negative_sentiment, update_agent_visualization
import time
import json

# Page configuration
st.set_page_config(
    page_title="AI Marketing Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'response' not in st.session_state:
    st.session_state.response = None
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'agent_history' not in st.session_state:
    st.session_state.agent_history = []
if 'show_reasoning' not in st.session_state:
    st.session_state.show_reasoning = False
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None

def update_agent_status(agent_name, message):
    """Update the agent status in the session state"""
    st.session_state.current_agent = agent_name
    st.session_state.agent_history.append({
        'agent': agent_name,
        'message': message,
        'time': time.strftime('%H:%M:%S'),
        'status': 'active',
        'reasoning': [] if not isinstance(message, dict) else message.get('reasoning', [])
    })

def process_request():
    """Process the user's request and update the UI"""
    if not st.session_state.brand_name:
        st.error("Please enter a brand name")
        return

    st.session_state.processing = True
    st.session_state.agent_history = []  # Reset agent history for new request
    st.session_state.response = None  # Reset previous response

    try:
        status_container = st.empty()
        with status_container:
            st.info("🤖 Starting AI analysis...")

            # Call the appropriate function based on action type
            try:
                if st.session_state.action_type == "Improve Brand Performance":
                    # Update status to show Marketing Strategist is active
                    status_container.info("📊 Marketing Strategist is analyzing your brand...")
                    st.session_state.response = improve_strategies(st.session_state.brand_name)
                else:
                    # Update status to show Sentiment Analyzer is active
                    status_container.info("🎭 Sentiment Analyzer is processing social media data...")
                    st.session_state.response = negative_sentiment(st.session_state.brand_name)

                # Update status based on current agent
                if 'current_agent' in st.session_state:
                    agent_emoji = get_agent_emoji(st.session_state.current_agent)
                    status_container.info(f"{agent_emoji} {st.session_state.current_agent} is working on your request...")

                # Show success message and animation
                if st.session_state.response:
                    status_container.success("✨ Analysis complete! 🎉")
                    st.balloons()

            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                print(f"Error in function call: {str(e)}")  # Debug log
                return

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error in process_request: {str(e)}")  # Debug log
    finally:
        st.session_state.processing = False

# Main UI
st.title("🤖 AI Marketing Assistant")
st.markdown("---")

# Two-column layout for input
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Brand Information")
    brand_name = st.text_input(
        "Enter Brand Name",
        key="brand_name",
        placeholder="e.g., Nike, Apple, etc.",
        help="Enter the name of the brand you want to analyze"
    )

with col2:
    st.markdown("### Action Selection")
    action_type = st.radio(
        "Select Action",
        ["Improve Brand Performance", "Handle Negative Social Media Sentiment"],
        key="action_type",
        help="Choose the type of analysis you want to perform"
    )

# Action button
st.button(
    "Let AI Agents Do the Magic",
    on_click=process_request,
    disabled=st.session_state.processing
)

# Simple processing indicator
if st.session_state.processing:
    st.info(f"🤖 AI agents are analyzing your request... Currently active: {st.session_state.current_agent}")

# Response section
if st.session_state.response:
    st.markdown("### 📊 Analysis Results")
    with st.container():
        try:
            response_text = str(st.session_state.response).replace('**', '__').replace('\n\n', '\n') if st.session_state.response else "No response generated"
            st.markdown(
                f"""
                <div class="output-container">
                    <div class="response-area">
                        {response_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error displaying response: {str(e)}")

def get_agent_emoji(agent_name):
    """Return an appropriate emoji for each agent type"""
    emoji_map = {
        "Admin": "👨‍💼",
        "Marketing Strategist": "📊",
        "Sentiment Analyzer": "🎭",
        "Campaign Performance Analyzer": "📈",
        "Brand Voice Analyzer": "🎯",
        "Quality Assurance": "✨",
        "System": "🤖"
    }
    return emoji_map.get(agent_name, "🤖")

def get_health_status(agent, current_agent):
    """Determine the health status of an agent"""
    if agent == current_agent:
        return "active", "Working..."
    elif agent in [interaction['agent'] for interaction in st.session_state.agent_history[-3:]]:
        return "resting", "Resting"
    elif agent in [interaction['agent'] for interaction in st.session_state.agent_history]:
        return "completed", "Done"
    return "idle", "Ready"

# Collapsible Agent Interaction Flow at the bottom
if st.session_state.agent_history:
    st.markdown("---")
    with st.expander("🔍 View Agent Interaction Details", expanded=False):
        for idx, interaction in enumerate(st.session_state.agent_history):
            health_status, status_text = get_health_status(interaction['agent'], st.session_state.current_agent)
            emoji = get_agent_emoji(interaction['agent'])

            # Display agent interaction with health indicator
            st.markdown(f"""
                <div class="agent-interaction">
                    <div class="agent-status-wrapper">
                        <small style="color: #666;">{interaction['time']}</small>
                        <span class="agent-emoji">{emoji}</span>
                        <strong style="color: #0066cc;">{interaction['agent']}</strong>
                        <div class="agent-health">
                            <div class="health-indicator {health_status}" title="{status_text}"></div>
                        </div>
                    </div>
                </div>
                <div class="agent-message">
                    {interaction['message']}
                </div>
                """, unsafe_allow_html=True)

            # If there are reasoning steps, display them
            if interaction.get('reasoning'):
                st.markdown("**Reasoning Steps:**")
                for step in interaction['reasoning']:
                    st.markdown(f"• **{step['title']}**: {step['description']}")
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Powered by AI Agents | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
