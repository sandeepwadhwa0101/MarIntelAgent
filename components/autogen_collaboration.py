import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import time

API_KEY="sk-b6f5f86349bd44ed9d9df50a90d3729c"
URL=""
MODEL="llama3.1:8b"

config_list = [
  {
    "model": MODEL,
    "base_url": URL,
    "api_key": API_KEY,
  }
]

def update_agent_visualization(agent_name, message):
    """Update the Streamlit session state with current agent status"""
    if 'current_agent' in st.session_state:
        st.session_state.current_agent = agent_name
        if 'agent_history' in st.session_state:
            # Extract reasoning steps from message if available
            reasoning_steps = []
            if isinstance(message, str):
                # Try to identify reasoning steps in the message
                message_parts = message.split('\n')
                current_step = None
                for part in message_parts:
                    part = part.strip()
                    if part.startswith(('1.', '2.', '3.', '4.', '5.')) or part.startswith('**') or part.startswith('Step'):
                        if current_step:
                            reasoning_steps.append(current_step)
                        current_step = {
                            'title': part.strip('*').strip(),
                            'description': ''
                        }
                    elif current_step and part:
                        current_step['description'] += part + ' '
                if current_step:
                    reasoning_steps.append(current_step)

            # If no structured steps found but message exists, create a single reasoning step
            if not reasoning_steps and message:
                reasoning_steps = [{
                    'title': 'Analysis Step',
                    'description': message if isinstance(message, str) else str(message)
                }]

            st.session_state.agent_history.append({
                'agent': agent_name,
                'message': message if isinstance(message, str) else str(message),
                'time': time.strftime('%H:%M:%S'),
                'status': 'active',
                'reasoning': reasoning_steps
            })

def wrap_agent_methods(agent, agent_name):
    """Wrap agent methods to track interactions"""
    original_send = agent.send
    original_receive = agent.receive

    def send_wrapper(message, recipient, request_reply=None, silent=False):
        # Update visualization before sending message
        update_agent_visualization(agent_name, f"Sending message to {recipient.name}")
        return original_send(message, recipient, request_reply, silent)

    def receive_wrapper(message, sender, request_reply=None, silent=False):
        # Update visualization after receiving message
        update_agent_visualization(agent_name, f"Received message from {sender.name}")
        if hasattr(message, 'content'):
            update_agent_visualization(agent_name, str(message.content))
        return original_receive(message, sender, request_reply, silent)

    agent.send = send_wrapper
    agent.receive = receive_wrapper
    return agent

# Wrap each agent with visualization updates
user_proxy = wrap_agent_methods(UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",  
    system_message="1. A human admin. 2. Interact with the team. 3. Plan execution needs to be approved by this Admin.",
    code_execution_config=False,
    llm_config={"config_list": config_list},
    description="""Call this Agent if:   
        You need guidance.
        The program is not working as expected.
        You need api key                  
        DO NOT CALL THIS AGENT IF:  
        You need to execute the code.""",
), "Admin")

marketing_strategist = wrap_agent_methods(AssistantAgent(
    name="Marketing Strategist",
    llm_config={"config_list": config_list},
    system_message="""You are an accomplished Marketing Strategist, follow these guidelines: 
    1. Your strategy should include atleast 5 steps and should provide a detailed plan to solve the task.
    2. Post project review isn't needed. 
    3. Leverage data from sentinment_analyzer and campaign_performance_analyzer to identify the current situation of the brand and use it in your strategy.   
    4. The plan should account for brand's unique brand voice, consult brand_voice to get this information.
    5. Revise the plan based on feedback from admin and quality_assurance.
    6. Do not show appreciation in your responses, say only what is necessary.  
    7. The final message should include an accurate answer to the user request.
    8. Ensure that the impact from strategies is measurable, and provide relevant metrics that should be measured.
    9. Do not provide timelines or plan for request, solve it immediately.
    """,
), "Marketing Strategist")

sentiment_analyzer = wrap_agent_methods(AssistantAgent(
    name="Sentiment Analyzer",
    system_message="""You are an expert Sentiment Analyzer, follow these guidelines: 
    1. Take the social conversation data from a brand and analyze the sentiment based on these conversations.
    2. The sentiment can be either Postive, Negative or Neutral. 
    3. Do not provide timelines or plan for request, solve it immediately.
    """,
    llm_config={"config_list": config_list}
), "Sentiment Analyzer")

campaign_performance_analyzer = wrap_agent_methods(AssistantAgent(
    name="Campaign Performance Analyzer",
    system_message="""You are an Marketing Data Analyst, follow these guidelines: 
    1. Request historical campaign data from the admin. 
    2. The data should be weekly or daily, and must include metrics - impressions, clicks, and spend. It can include optional Engagement metric.
    3. As a data analyst, aggregate the data into monthly numbers, by summing up the Impressions, Clicks, Spend and Engagement.    
    4. On the aggregated data, calculate metrics such as CTR i.e. divide Clicks by Impressions, Engagement Rate i.e. divide engagement by impressions, CPC i.e. divide Spend by Clicks and CPM i.e. divide Impressions by Spend and multiplied by 1000.  
    5. Use widely available benchmarks to evaluate CTR, CPC, and Engagement Rate for this brand.
    6. Do not provide timelines or plan for request, solve it immediately.
    """,
    llm_config={"config_list": config_list}
), "Campaign Performance Analyzer")

brand_voice = wrap_agent_methods(AssistantAgent(
    name="Brand Voice Analyzer",
    system_message="""You are an accomplished Brand Voice Analyzer, follow these guidelines: 
    1. You fetch the brand voice of a brand by analyzing their past campaigns.
    2. You are expert at evaluating the brand voice and evaluating them in dimensions such as Professional, Casual, Funny and any other new dimension you may discover from the data.
    3. If you are unable to get this data from your knowledge base, request the data from Admin.
    4. Grade the brand voice on 5 dimensions, so that they can plotted in a radar chart.
    5. Do not provide timelines or plan for request, solve it immediately.
    """,
    llm_config={"config_list": config_list}
), "Brand Voice Analyzer")

quality_assurance = wrap_agent_methods(AssistantAgent(
    name="Quality Assurance",
    system_message="""You are a Quality Assurance. Follow these instructions:
      1. Double check the marketing plan, 
      2. if there's a error suggest a resolution
      3. If the task is not solved, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach.
      4. Do not provide timelines or plan for request, solve it immediately.
      5. Make sure the marketing strategies are actionable, alight with industry trends and as per the brand voice.
      6. Call out if any strategy is unconventional or out of box.
      """,
    llm_config={"config_list": config_list}
), "Quality Assurance")

allowed_transitions = {
    user_proxy: [marketing_strategist],
    marketing_strategist: [sentiment_analyzer, campaign_performance_analyzer, brand_voice, quality_assurance],
    sentiment_analyzer: [marketing_strategist],
    campaign_performance_analyzer: [marketing_strategist],
    brand_voice: [marketing_strategist],
    quality_assurance: [marketing_strategist, sentiment_analyzer, campaign_performance_analyzer, brand_voice],
}

system_message_manager = "You are the manager of a research group your role is to manage the team and make sure the project is completed successfully."
groupchat = GroupChat(
    agents=[user_proxy, marketing_strategist, sentiment_analyzer, campaign_performance_analyzer, brand_voice, quality_assurance],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
    messages=[],
    max_round=30,
    send_introductions=True
)
manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list}, system_message=system_message_manager)

def extract_message_content(message):
    """Helper function to safely extract message content"""
    try:
        if hasattr(message, 'content'):
            return message.content
        elif isinstance(message, dict):
            return message.get('content', str(message))
        return str(message)
    except Exception as e:
        print(f"Error extracting message content: {e}")
        return str(message)

def improve_strategies(brand):
    """Generate improvement strategies for a brand"""
    try:
        print(f"Starting strategy improvement analysis for {brand}")  # Debug log
        update_agent_visualization("System", f"Starting strategy improvement analysis for {brand}")
        task = f"How is my brand {brand} performing and what are some marketing strategies I can leverage to improve my brand awareness and engagements?"

        chat_result = user_proxy.initiate_chat(
            manager,
            message=task,
            clear_history=True  
        )

        # Get the last message and extract its content
        last_message = groupchat.messages[-1]
        return extract_message_content(last_message)
    except Exception as e:
        print(f"Error in improve_strategies: {e}")  # Debug log
        return f"Error analyzing brand strategy: {str(e)}"

def negative_sentiment(brand):
    """Handle negative sentiment analysis for a brand"""
    try:
        print(f"Starting negative sentiment analysis for {brand}")  # Debug log
        update_agent_visualization("System", f"Starting negative sentiment analysis for {brand}")
        task = f"My brand {brand} is experiencing negative sentiment on social media, what should I do to improve my brand image?"

        chat_result = user_proxy.initiate_chat(
            manager,
            message=task,
            clear_history=True  
        )

        # Get the last message and extract its content
        last_message = groupchat.messages[-1]
        return extract_message_content(last_message)
    except Exception as e:
        print(f"Error in negative_sentiment: {e}")  # Debug log
        return f"Error analyzing negative sentiment: {str(e)}"
