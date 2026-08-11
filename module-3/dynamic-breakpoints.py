from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
    
    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---Step 3---")
    return state

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Compile the graph with memory
graph = builder.compile(checkpointer=memory)

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("dynamic_breakpoints_graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as dynamic_breakpoints_graph.png")

input = {'input': 'Hello Grog'}
thread = {'configurable': {'thread_id': '1'}}

for event in graph.stream(input, thread, stream_mode='values'):
    print(event)

state = graph.get_state(thread)
print(state.next)
print(state.tasks)
graph.update_state(thread, {'input': 'Hello'})
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)