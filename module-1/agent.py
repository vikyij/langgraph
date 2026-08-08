from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

def multiply(a: int, b: int):
    """Multiply a and b
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b:int):
    """Adds a and b

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b:int):
    """Divides a and b

    Args: 
        a: first int
        b: second int
    """
    return a/b

tools = [multiply, add, divide]
llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=100)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

#system message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

#node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}

#graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, 'assistant')
builder.add_conditional_edges('assistant',
                                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                               tools_condition)
builder.add_edge('tools', 'assistant')

graph = builder.compile()

# Generate and save the graph visualization
graph_image = graph.get_graph(xray=True).draw_mermaid_png()

with open("agent_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as agent_graph.png")

messages = HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")
messages = graph.invoke({'messages': [messages]})
for m in messages['messages']:
    m.pretty_print()
