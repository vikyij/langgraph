from dotenv import load_dotenv
from typing import Literal
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio

load_dotenv()

# llm
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

#state
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

config = {"configurable": {"thread_id": "1"}}

# Start conversation
for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Victoria")]}, config, stream_mode="updates"):
    chunk['conversation']['messages'].pretty_print()

# Start conversation, again
config = {"configurable": {"thread_id": "2"}}

# stream_mode = 'values'
input_message = HumanMessage(content="hi! I'm Victoria")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
   
    print("---"*25)

#stream tokens
# config = {"configurable": {"thread_id": "3"}}
# input_message = HumanMessage(content="Tell me about Love")
# async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
#     print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")

async def main():
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about love")

    async for event in graph.astream_events(
        {"messages": [input_message]},
        config=config,
        version="v2",
    ):
        print(
            f"Node: {event['metadata'].get('langgraph_node', '')}. "
            f"Type: {event['event']}. "
            f"Name: {event['name']}"
        )


if __name__ == "__main__":
    asyncio.run(main())

# config = {"configurable": {"thread_id": "5"}}
# input_message = HumanMessage(content="Tell me about the 49ers NFL team")
# async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
#     # Get chat model tokens from a particular node 
#     if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
#         data = event["data"]
#         print(data["chunk"].content, end="|")

node_to_stream = 'conversation'  
async def main():
    config = {"configurable": {"thread_id": "5"}}
    input_message = HumanMessage(content="Tell me about love")

    async for event in graph.astream_events(
        {"messages": [input_message]},
        config=config,
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
            data = event["data"]
            print(data["chunk"].content, end="|")


if __name__ == "__main__":
    asyncio.run(main())