from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

# langchain messages
messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Victoria"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in Canada.", name="Victoria"))

for m in messages:
    m.pretty_print()

# chat models
llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=100)
result = llm.invoke(messages)
print(type(result))
print(result)
# print(result.response_metadata)

# tools
def multiply(a: int, b:int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

tool_call = llm_with_tools.invoke([HumanMessage(content="what is 5 multiplied by 5")])
print(tool_call)
# print(tool_call.tool_calls)

# using messages as graph state
class MessagesState(MessagesState):
    pass

# node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("chain_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as chain_graph.png")

graph_message=graph.invoke({"messages": HumanMessage(content="Multiply 5 and 20")})
for m in graph_message['messages']:
    m.pretty_print()