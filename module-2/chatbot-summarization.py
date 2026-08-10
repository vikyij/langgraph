from dotenv import load_dotenv
from typing import Literal
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class State(MessagesState):
    summary: str

#node to call our LLM that incorporates a summary, if it exists, into the prompt.
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}

# node to produce a summary
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal ["summarize_conversation",END]:
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Define a new graph
builder = StateGraph(State)
builder.add_node('conversation', call_model)
builder.add_node(summarize_conversation)

# Set the entrypoint as conversation
builder.add_edge(START, 'conversation')
builder.add_conditional_edges('conversation', should_continue)
builder.add_edge('summarize_conversation', END)

#compile
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("chatbot-summarization.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as chatbot-summarization.png")

config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Victoria")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like messi!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

print(graph.get_state(config).values.get("summary",""))

input_message = HumanMessage(content="is he one of or actually the GOAT")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

print(graph.get_state(config).values.get("summary",""))

input_message = HumanMessage(content="he is undisputedly the GOAT")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

print(graph.get_state(config).values.get("summary",""))