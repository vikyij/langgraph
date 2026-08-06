# define the state of the graph
import random
from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_state: str


# define nodes
def node_1(state):
    return {"graph_state": state['graph_state'] + " I am"}

def node_2(state):
    return {"graph_state": state['graph_state'] + " happy!"}

def node_3(state):
    return {"graph_state": state['graph_state'] + " sad!"}


# define edges
def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

# Generate and save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as file:
    file.write(graph_image)

print("Graph saved as graph.png")

result = graph.invoke({"graph_state": "Hi, I'm Victoria,"})
print(result)
