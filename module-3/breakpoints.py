from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


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


memory = MemorySaver()
graph = builder.compile(interrupt_before=['tools'],checkpointer=memory)

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("breakpoints_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as breakpoints_graph.png")

#specify a thread
thread = {'configurable': {'thread_id': 1}}

# Specify an input
messages = {'messages': [HumanMessage(content='Add 3 and 4')]}
# Run the graph until the first interruption
for event in graph.stream(messages, thread, stream_mode='values'):
    event['messages'][-1].pretty_print()

# We can get the state and look at the next node to call. This is a nice way to see that the graph has been interrupted.
state = graph.get_state(thread)
print(state.next)

# When we invoke the graph with `None`, it will just continue from the last state checkpoint!
# Get user feedback
user_approval = input("Do you want to call the tool? (yes/no): ")

# Check approval
if user_approval.lower() == "yes":
    
    # If approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
        
else:
    print("Operation cancelled by user.")