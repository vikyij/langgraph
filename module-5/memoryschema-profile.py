from dotenv import load_dotenv
from typing import TypedDict, List, Optional
import uuid
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from trustcall import create_extractor

load_dotenv()

# We'll extend our chatbot to save semantic memories to a single user profile
# class UserProfile(TypedDict):
#     """User profile schema with typed fields"""
#     user_name: str  # The user's preferred name
#     interests: List[str]  # A list of the user's interests

# TypedDict instance
# user_profile: UserProfile = {
#     "user_name": "Victoria",
#     "interests": ["singing", "technology", "traveling"]
# }
# print(user_profile)

# Initialize the in-memory store
# in_memory_store = InMemoryStore()

# Namespace for the memory to save
# user_id = "1"
# namespace_for_memory = (user_id, "memory")

# Save a memory to namespace as key and value
# key = "user_profile"
# value = user_profile
# in_memory_store.put(namespace_for_memory, key, value)

# Search 
# for m in in_memory_store.search(namespace_for_memory):
#     print(m.dict())

# Get the memory by namespace and key
# profile = in_memory_store.get(namespace_for_memory, "user_profile")
# print(profile)

# Initialize the model
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)

# Bind schema to model
# structured_llm = llm.with_structured_output(UserProfile)

# Invoke the model to produce structured output that matches the schema
# structured_output = structured_llm.invoke([HumanMessage("My name is victoria, I love singing")])
# print(structured_output)

# Chatbot instruction
# MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
# If you have memory for this user, use it to personalize your responses.
# Here is the memory (it may be empty): {memory}"""

# # Create new memory from the chat history and any existing memory
# CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history. 
# This will be saved for long-term memory. If there is an existing memory, simply update it. 
# Here is the existing memory (it may be empty): {memory}"""

# def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

#     """Load memory from the store and use it to personalize the chatbot's response."""
    
#     # Get the user ID from the config
#     user_id = config["configurable"]["user_id"]

#     # Retrieve memory from the store
#     namespace = ("memory", user_id)
#     existing_memory = store.get(namespace, "user_memory")

#     # Format the memories for the system prompt
#     if existing_memory and existing_memory.value:
#         memory_dict = existing_memory.value
#         formatted_memory = (
#             f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
#             f"Interests: {', '.join(memory_dict.get('interests', []))}"
#         )
#     else:
#         formatted_memory = None

#     # Format the memory in the system prompt
#     system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

#     # Respond using memory as well as the chat history
#     response = llm.invoke([SystemMessage(content=system_msg)]+state["messages"])

#     return {"messages": response}

# def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

#     """Reflect on the chat history and save a memory to the store."""
    
#     # Get the user ID from the config
#     user_id = config["configurable"]["user_id"]

#     # Retrieve existing memory from the store
#     namespace = ("memory", user_id)
#     existing_memory = store.get(namespace, "user_memory")

#     # Format the memories for the system prompt
#     if existing_memory and existing_memory.value:
#         memory_dict = existing_memory.value
#         formatted_memory = (
#             f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
#             f"Interests: {', '.join(memory_dict.get('interests', []))}"
#         )
#     else:
#         formatted_memory = None
        
#     # Format the existing memory in the instruction
#     system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)

#     # Invoke the model to produce structured output that matches the schema
#     new_memory = structured_llm.invoke([SystemMessage(content=system_msg)]+state['messages'])

#     # Overwrite the existing use profile memory
#     key = "user_memory"
#     store.put(namespace, key, new_memory)

    # Define the graph
# builder = StateGraph(MessagesState)
# builder.add_node("call_model", call_model)
# builder.add_node("write_memory", write_memory)
# builder.add_edge(START, "call_model")
# builder.add_edge("call_model", "write_memory")
# builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
# across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
# within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
# graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
# config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
# input_messages = [HumanMessage(content="Hi, my name is Victoria and I love to sing and listen to gospel songs.")]

# Run the graph
# for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
#     chunk["messages"][-1].pretty_print()

# Namespace for the memory to save
# user_id = "1"
# namespace = ("memory", user_id)
# existing_memory = across_thread_memory.get(namespace, "user_memory")
# print(existing_memory.value)

# `with_structured_output` is very useful, but what happens if we're working with a more complex schema? 
# This is a Pydantic model that describes a user's preferences for communication and trust fall.
class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str

class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None

class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None

class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None

class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None

class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore

class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences

class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences

# Now, let's try extraction of this schema using the `with_structured_output` method.
# Bind schema to model
model_with_structure = llm.with_structured_output(TelegramAndTrustFallPreferences)

# Conversation
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

# Invoke the model
# try:
#     model_with_structure.invoke(f"""Extract the preferences from the following conversation:
#     <convo>
#     {conversation}
#     </convo>""")
# except ValidationError as e:
#     print(e)

# Let's first show simple usage of extraction with TrustCall on this list of messages
# Conversation
conversation = [HumanMessage(content="Hi, I'm Victoria."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="I really like singing.")]

# Create the extractor
# trustcall_extractor = create_extractor(
#     llm,
#     tools=[UserProfile],
#     tool_choice="UserProfile"
# )

# Instruction
system_msg = "Extract the user profile from the following conversation"

# Invoke the extractor
# result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+conversation})
# for m in result["messages"]: 
#     m.pretty_print()

# schema = result["responses"]
# print(schema)

# print(schema[0].model_dump())

# print(result["response_metadata"])

# Update the conversation
updated_conversation = [HumanMessage(content="Hi, Victoria."), 
                        AIMessage(content="Nice to meet you, Victoria."), 
                        HumanMessage(content="I really like Singing."),
                        AIMessage(content="What is your music genre and favourite artist?"),
                        HumanMessage(content="Gospel music, I like kaestrings"),]

# Update the instruction
system_msg = f"""Update the memory (JSON doc) to incorporate new information from the following conversation"""

# Invoke the extractor with the updated instruction and existing profile with the corresponding tool name (UserProfile)
# result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+updated_conversation}, 
#                                     {"existing": {"UserProfile": schema[0].model_dump()}})  

# for m in result["messages"]: 
#     m.pretty_print()

# print(result["response_metadata"])

# updated_schema = result["responses"][0]
# print(updated_schema.model_dump())

# Now, let's bring Trustcall into our chatbot to create *and update* a memory profile.
# Schema 
class UserProfile(BaseModel):
    """ Profile of a user """
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")

# Create the extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[UserProfile],
    tool_choice="UserProfile", # Enforces use of the UserProfile tool
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"      
        )
    else:
        formatted_memory = None

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # Respond using memory as well as the chat history
    response = llm.invoke([SystemMessage(content=system_msg)]+state["messages"])
    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and save a memory to the store."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
        
    # Get the profile as the value from the list, and convert it to a JSON doc
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None
    
    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]+state["messages"], "existing": existing_profile})
    
    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()

    # Save the updated profile
    key = "user_memory"
    store.put(namespace, key, updated_profile)

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi, my name is Victoria")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# User input 
input_messages = [HumanMessage(content="I like to sing")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
print(existing_memory.dict())

# The user profile saved as a JSON object
print(existing_memory.value)

# User input 
input_messages = [HumanMessage(content="I also like to listen to music")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="What songs do you recommend for me?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


