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


messages = HumanMessage(content="Add 3 and 4.")
messages = graph.invoke({'messages': [messages]})
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Multiply that by 2.")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

# We don't retain memory of 7 from our initial chat! This is because [state is transient] to a single graph execution.

# Of course, this limits our ability to have multi-turn conversations with interruptions. We can use [persistence] to address this! 

# LangGraph can use a checkpointer to automatically save the graph state after each step.

# This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update. 

# One of the easiest checkpointers to use is the `MemorySaver`, an in-memory key-value store for Graph state.

# All we need to do is simply compile the graph with a checkpointer, and our graph has memory!
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

#specify a thread
config = {'configurable': {'thread_id': 1}}

# Specify an input
messages = [HumanMessage(content='Add 3 and 4')]
messages = graph.invoke({'messages': messages}, config)
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content='Multiply that by 2.')]
messages = graph.invoke({'messages': messages}, config)
for m in messages['messages']:
    m.pretty_print()