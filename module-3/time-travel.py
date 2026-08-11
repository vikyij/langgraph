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
llm_with_tools = llm.bind_tools(tools)

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
graph = builder.compile(checkpointer=memory)

thread = {'configurable': {'thread_id': '1'}}
input={'messages': HumanMessage(content='Multiply 3 and 4')}
for event in graph.stream(input, thread, stream_mode='values'):
    event['messages'][-1].pretty_print()

# current state of our graph
state = graph.get_state(thread)
# print(state, 'here')

# state history of our agent
all_states = [s for s in graph.get_state_history(thread)]
# The first element is the current state, just as we got from `get_state`.
# print(all_states[-2])
# print(len(all_states))

# Replay: We can re-run our agent from any of the prior steps.
to_replay = all_states[-2]
# print(to_replay, 'replay')
# print(to_replay.values, 'values')
# print(to_replay.next, 'next')
# print(to_replay.config, 'config')

# To replay from here, we simply pass the config back to the agent! The graph knows that this checkpoint has aleady been executed. 
# It just re-plays from this checkpoint!
# for event in graph.stream(None, to_replay.config, stream_mode="values"):
#     event['messages'][-1].pretty_print()

# Forking:  if we want to run from that same step, but with a different input. 
to_fork = all_states[-2]
# print(to_fork.values['messages'])
# print(to_fork.config, 'config')

# We can just run `update_state` with the `checkpoint_id` supplied. 
# Because of how reducer on messages work, to overwrite the the message, we just supply the message ID, which we have `to_fork.values["messages"].id`.
fork_config = graph.update_state(to_fork.config, {'messages': [HumanMessage(content='Add 5 and 6', id=to_fork.values['messages'][0].id)]})
# print(fork_config)
all_states = [state for state in graph.get_state_history(thread) ]
# print(all_states[0].values["messages"])
# print(graph.get_state(thread))
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()

# print(graph.get_state(thread))