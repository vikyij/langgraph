from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage

load_dotenv()

def multiply(a: int, b: int): 
    """Multiply a and b

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens= 100)
llm_with_tools = llm.bind_tools([multiply])

# We use the built-in `ToolNode` and simply pass a list of our tools to initialize it. 
# We use the built-in `tools_condition` as our conditional edge.

#node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

# build graph
builder = StateGraph(MessagesState)
builder.add_node('tool_calling_llm', tool_calling_llm)
builder.add_node('tools', ToolNode([multiply]))
builder.add_edge(START, 'tool_calling_llm')
builder.add_conditional_edges(
    'tool_calling_llm',
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge('tools', END)

graph = builder.compile()

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("router_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as router_graph.png")

messages = [HumanMessage(content="Hi, what is your name?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()