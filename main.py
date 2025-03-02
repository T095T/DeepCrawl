import os
from typing import TypedDict, Annotated, List
import operator
from huggingface_hub import InferenceClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import streamlit as st

os.environ["HF_API_KEY"] = ""  #  Hugging Face API key
os.environ["TAVILY_API_KEY"] = ""  # Tavily API key

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=os.environ["HF_API_KEY"])

# Define the state
class AgentState(TypedDict):
    query: str
    research_data: List[str]
    draft: str
    messages: Annotated[List[str], operator.add]

# Wrapper for Hugging Face to mimic LangChain's invoke method
class HFWrapper:
    def __init__(self, client, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.client = client
        self.model = model

    def invoke(self, prompt):
        # Use text_generation for free-tier compatibility
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=500,  # Limit output length
            temperature=0.1      # Low for coherent responses
        )
        return type("Response", (), {"content": response})()

# Initialize LLM (Hugging Face) and tools
llm = HFWrapper(client)
tavily_tool = TavilySearchResults(max_results=5)
research_tool_node = ToolNode([tavily_tool])

# Research Agent
def research_node(state: AgentState):
    query = state["query"]
    results = tavily_tool.invoke(query)
    research_data = [result["content"] for result in results]
    return {
        "research_data": research_data,
        "messages": [f"Research Agent: Collected {len(research_data)} items for '{query}'"]
    }

# Drafting Agent
def drafting_node(state: AgentState):
    query = state["query"]
    research_data = state["research_data"]
    prompt = f"""
    You are an expert writer. Based on the following research data, draft a concise and accurate answer to the query: '{query}'.
    Research Data: {research_data}
    Provide a clear, structured response.
    """
    try:
        response = llm.invoke(prompt)
        draft = response.content
    except Exception as e:
        draft = f"Error: Failed to generate draft - {str(e)}"
        print(draft)
    return {
        "draft": draft,
        "messages": [f"Drafting Agent: Generated draft for '{query}'"]
    }

# Build the workflow
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("drafting", drafting_node)
workflow.add_edge("__start__", "research")
workflow.add_edge("research", "drafting")
workflow.add_edge("drafting", END)
app = workflow.compile()





# Streamlit interface
def main():
    st.title("Deep Research Web Crawler")
    st.write("Enter your input to generate a draft answer based on the latest research data.")

    # Input query
    query = st.text_area("Research Query", placeholder="e.g., What are the latest trends in the Industry?", height=100)

    if st.button("Submit"):
        if query:
            with st.spinner("Processing your query..."):
                initial_state = {
                    "query": query,
                    "research_data": [],
                    "draft": "",
                    "messages": []
                }
                result = app.invoke(initial_state)

                # Display results
                st.subheader("Query")
                st.write(query)

                st.subheader("Research Data")
                for i, data in enumerate(result["research_data"], 1):
                    preview = f"{data[:100]}..." if len(data) > 100 else data
                    with st.expander(f"{i}. {preview}"):
                        st.write(data)  # Full text inside expander

                st.subheader("Draft Answer")
                st.write(result["draft"])

                # st.subheader("Agent Messages")
                # for msg in result["messages"]:
                #     st.write(f"- {msg}")
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()