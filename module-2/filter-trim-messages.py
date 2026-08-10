from dotenv import load_dotenv
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, trim_messages
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# using RemoveMessage

# Nodes
def filter_messages(state: MessagesState):
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}

def chat_model_node(state: MessagesState):    
    return {"messages": [llm.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("filter", filter_messages)
builder.add_node("chat_model", chat_model_node)

builder.add_edge(START, "filter")
builder.add_edge("filter", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()

# Message list with a preamble
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Victoria", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Victoria", id="4"))

# Invoke
output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

# # filtering messages: If you don't need or want to modify the graph state, you can just filter the messages you pass to the chat model.

# Node
def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"][-1:])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)

builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me more about Dolphins!", name="Victoria"))

# The state has all of the mesages. But, when we look at the LangSmith trace we see that the model invocation only uses the last message

output = graph.invoke({'messages': messages})
for m in output['messages']:
    m.pretty_print()

# trim messages
# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatGroq(model="llama-3.3-70b-versatile"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

messages.append(output['messages'][-1])
messages.append(HumanMessage(f"Tell me where Orcas live!", name="Victoria"))

# Invoke, using message trimming in the chat_model_node 
# Let's look at the LangSmith trace to see the model invocation
messages_out_trim = graph.invoke({'messages': messages})
for m in messages_out_trim['messages']:
    m.pretty_print()