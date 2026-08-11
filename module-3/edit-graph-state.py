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
# def assistant(state: MessagesState):
#     return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}

# #graph
# builder = StateGraph(MessagesState)

# # Define nodes: these do the work
# builder.add_node('assistant', assistant)
# builder.add_node('tools', ToolNode(tools))

# # Define edges: these determine how the control flow moves
# builder.add_edge(START, 'assistant')
# builder.add_conditional_edges('assistant',
#                                 # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
#                                 # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#                                tools_condition)
# builder.add_edge('tools', 'assistant')

# graph = builder.compile()


# memory = MemorySaver()
# graph = builder.compile(interrupt_before=['assistant'],checkpointer=memory)

# # Generate and save the graph visualization
# graph_image = graph.get_graph().draw_mermaid_png()

# with open("edit_graph.png", "wb") as file:
#     file.write(graph_image)

# print("Graph saved as edit_graph.png")

# #specify a thread
# thread = {'configurable': {'thread_id': 1}}

# # Specify an input
# messages = {'messages': [HumanMessage(content='Add 3 and 4')]}
# # Run the graph until the first interruption
# for event in graph.stream(messages, thread, stream_mode='values'):
#     event['messages'][-1].pretty_print()

# # We can get the state and look at the next node to call. This is a nice way to see that the graph has been interrupted.
# state = graph.get_state(thread)
# print(state.next)

# #Now, we can directly apply a state update. Remember, updates to the `messages` key will use the `add_messages` reducer:
# # * If we want to over-write the existing message, we can supply the message `id`.
# # * If we simply want to append to our list of messages, then we can pass a message without an `id` specified, as shown below.
# graph.update_state(thread, {'messages': [HumanMessage(content='No, actually multiply 3 and 4')]})
# new_state = graph.get_state(thread).values
# for m in new_state['messages']:
#     m.pretty_print()
# for event in graph.stream(None, thread, stream_mode='values'):
#     event['messages'][-1].pretty_print()

# for event in graph.stream(None, thread, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# Awaiting user input
# We'll add a node that serves as a placeholder for human feedback within our agent. This `human_feedback` node allow the user to add feedback directly to state.
# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    pass

# Assistant node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

# Define edges: these determine the control flow
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "human_feedback")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()
with open("human_feedback_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as human_feedback_graph.png")

# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "2"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
    
# Get user input
user_input = input("Tell me how you want to update the state: ")

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()