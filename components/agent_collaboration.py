import os
import time
import local_llm
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict
from dotenv import load_dotenv

from langchain.agents import Tool, AgentType, initialize_agent
#Initialize the language model
# from langchain.llms import Mistral

# # Initialize the Mistral language model
# llm = Mistral()


class Agent:

    def __init__(self, name: str, role: str, skills: List[str]):
        self.name = name
        self.role = role
        self.skills = skills
        self.llm = None
        try:
            self.llm = local_llm.get_local_llm()
        except local_llm.LLMConnectionError as e:
            print(f"Warning: Failed to initialize LLM for {name}: {str(e)}")
            # The LLM will be retried on first process() call

    def process(self, task: str, context: List[Dict] = None) -> str:
        # Ensure LLM is initialized
        if self.llm is None:
            try:
                self.llm = local_llm.get_local_llm()
            except local_llm.LLMConnectionError as e:
                return f"Error: Unable to process task due to LLM connection issue: {str(e)}"

        try:
            messages = [
                SystemMessage(
                    content=
                    f"You are {self.name}, a {self.role}. Your skills include: {', '.join(self.skills)}. Respond to the task based on your role and skills."
                )
            ]

            if context:
                for msg in context:
                    if msg['role'] == 'human':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'ai':
                        messages.append(AIMessage(content=msg['content']))

            messages.append(HumanMessage(content=task))
            
            # Add timeout handling
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.llm.invoke(messages)
                    return response.content
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        return f"Error: Failed to process task after {max_retries} attempts: {str(e)}"
                    time.sleep(2)  # Wait before retry
                    
        except Exception as e:
            return f"Error: Failed to process task: {str(e)}"


class SentimentAnalysisAgent(Agent):

    def __init__(self):
        super().__init__("Senti", "Sentiment Analysis Specialist", [
            "deep knowledge of marketing",
            "understanding of customer service and behavior",
            "identifying sentiment and tone of social media posts"
        ])


class CampaignPerformanceAgent(Agent):

    def __init__(self):
        super().__init__(
            "Campaign Performance", "Campaign Performance Analysis Expert", [
                "interpreting numerical data", "statistical analysis",
                "data visualization description",
                "marketing campaign measurement", "marketing campaign analysis"
            ])


class BrandVoice(Agent):

    def __init__(self):
        super().__init__("Brand Voice", "Brand Voice Expert",
                         ["interpreting brand voice", "brand building expert"])


#Research Historical Context
def research_historical_context(history_agent, task: str,
                                context: list) -> list:
    print("🏛️ History Agent: Researching historical context...")
    history_task = f"Provide relevant historical context and information for the following task: {task}"
    history_result = history_agent.process(history_task)
    context.append({
        "role": "ai",
        "content": f"History Agent: {history_result}"
    })
    print(f"📜 Historical context provided: {history_result[:100]}...\n")
    return context


#Identify Data Needs
def identify_data_needs(data_agent, task: str, context: list) -> list:
    print(
        "📊 Data Agent: Identifying data needs based on historical context...")
    historical_context = context[-1]["content"]
    data_need_task = f"Based on the historical context, what specific data or statistical information would be helpful to answer the original question? Historical context: {historical_context}"
    data_need_result = data_agent.process(data_need_task, context)
    context.append({
        "role": "ai",
        "content": f"Data Agent: {data_need_result}"
    })
    print(f"🔍 Data needs identified: {data_need_result[:100]}...\n")
    return context


#Provide Historical Data
def provide_historical_data(history_agent, task: str, context: list) -> list:
    print("🏛️ History Agent: Providing relevant historical data...")
    data_needs = context[-1]["content"]
    data_provision_task = f"Based on the data needs identified, provide relevant historical data or statistics. Data needs: {data_needs}"
    data_provision_result = history_agent.process(data_provision_task, context)
    context.append({
        "role": "ai",
        "content": f"History Agent: {data_provision_result}"
    })
    print(f"📊 Historical data provided: {data_provision_result[:100]}...\n")
    return context


#Analyze Data
def analyze_data(data_agent, task: str, context: list) -> list:
    print("📈 Data Agent: Analyzing historical data...")
    historical_data = context[-1]["content"]
    analysis_task = f"Analyze the historical data provided and describe any trends or insights relevant to the original task. Historical data: {historical_data}"
    analysis_result = data_agent.process(analysis_task, context)
    context.append({"role": "ai", "content": f"Data Agent: {analysis_result}"})
    print(f"💡 Data analysis results: {analysis_result[:100]}...\n")
    return context


#Synthesize Final Answer
def synthesize_final_answer(history_agent, task: str, context: list) -> str:
    print("🏛️ History Agent: Synthesizing final answer...")
    synthesis_task = "Based on all the historical context, data, and analysis, provide a comprehensive answer to the original task."
    final_result = history_agent.process(synthesis_task, context)
    return final_result


#HistoryDataCollaborationSystem Class
class HistoryDataCollaborationSystem:

    def __init__(self):
        self.history_agent = Agent("Clio", "History Research Specialist", [
            "deep knowledge of historical events",
            "understanding of historical contexts",
            "identifying historical trends"
        ])
        self.data_agent = Agent("Data", "Data Analysis Expert", [
            "interpreting numerical data", "statistical analysis",
            "data visualization description"
        ])

    def solve(self, task: str, timeout: int = 300) -> str:
        print(f"\n👥 Starting collaboration to solve: {task}\n")

        start_time = time.time()
        context = []
        error_count = 0
        max_errors = 3

        steps = [(research_historical_context, self.history_agent),
                 (identify_data_needs, self.data_agent),
                 (provide_historical_data, self.history_agent),
                 (analyze_data, self.data_agent),
                 (synthesize_final_answer, self.history_agent)]

        for step_func, agent in steps:
            if time.time() - start_time > timeout:
                return "Operation timed out. The process took too long to complete."
            
            # Check LLM availability before step
            if isinstance(agent, Agent) and agent.llm is None:
                try:
                    agent.llm = local_llm.get_local_llm()
                except local_llm.LLMConnectionError as e:
                    error_count += 1
                    if error_count >= max_errors:
                        return f"Critical error: Unable to establish LLM connection after multiple attempts: {str(e)}"
                    continue  # Skip this step and try next
            
            try:
                step_start_time = time.time()
                result = step_func(agent, task, context)
                
                # Handle potential error messages returned from process()
                if isinstance(result, str) and result.startswith("Error:"):
                    error_count += 1
                    print(f"Step error: {result}")
                    if error_count >= max_errors:
                        return f"Critical error: Too many step failures: {result}"
                    continue  # Skip this step and try next
                
                if isinstance(result, str):
                    return result  # This is the final answer
                context = result
                
                # Add execution time to context for monitoring
                step_time = time.time() - step_start_time
                if isinstance(context, list):
                    context.append({
                        "role": "system",
                        "content": f"Step execution time: {step_time:.2f}s"
                    })
                
            except Exception as e:
                error_count += 1
                error_msg = f"Error during collaboration step {step_func.__name__}: {str(e)}"
                print(error_msg)
                
                if error_count >= max_errors:
                    return f"Critical error: Collaboration failed after multiple attempts: {error_msg}"
                continue  # Try next step

        print("\n✅ Collaboration complete. Final answer synthesized.\n")
        return context[-1]["content"] if context else "Unable to generate response due to system issues"


#Example usage
# Create an instance of the collaboration system
collaboration_system = HistoryDataCollaborationSystem()

# Define a complex historical question that requires both historical knowledge and data analysis
question = "How did urbanization rates in Europe compare to those in North America during the Industrial Revolution, and what were the main factors influencing these trends?"

# Solve the question using the collaboration system
result = collaboration_system.solve(question)

# Print the result
print(result)
