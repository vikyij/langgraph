from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# private state: First, let's cover the case of passing between nodes.
# This is useful for anything needed as part of the intermediate working logic of the graph, but not relevant for the overall graph input or output.
class PrivateState(TypedDict):
    foo: int

class OverallState(TypedDict):
    bar: int

def node_1(state: OverallState) -> PrivateState:
    return {"foo": state['bar'] + 1}

def node_2(state: PrivateState) -> OverallState:
    return{'bar': state['foo'] + 1}

# build graph
builder = StateGraph(OverallState)
builder.add_node('node_1', node_1)
builder.add_node('node_2', node_2)

builder.add_edge(START, 'node_1')
builder.add_edge('node_1', 'node_2')
builder.add_edge('node_2', END)

graph = builder.compile()

print(graph.invoke({'bar': 1}))

# input/output schemas
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState):
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)

graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# Notice that the output of invoke contains all keys in `OverallState`. 
print(graph.invoke({"question":"hi"}))

# Here, `input` / `output` schemas perform *filtering* on what keys are permitted on the input and output of the graph. 
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)

graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# We can see the `output` schema constrains the output to only the `answer` key.
print(graph.invoke({"question":"hi"}))