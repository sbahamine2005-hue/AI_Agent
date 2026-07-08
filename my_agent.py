from langchain_neo4j.checkpoint import Neo4jSaver
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os 
import uuid
import wolframalpha
import requests
import ast
from neo4j import GraphDatabase

#the first think is loading the varible from the .env file
load_dotenv()

groq_api_key   = os.getenv("groq_api_key")
groq_api_key_2 = os.getenv("groq_api_key_2")
groq_api_key_3 = os.getenv("groq_api_key_3")
groq_api_key_4 = os.getenv("groq_api_key_4")
NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")
tavily_api_key = os.getenv("tavily_api_key")
weather_api_key = os.getenv("wheather_api_key")
APP_ID = os.getenv("appid")
wolfram_client = wolframalpha.Client(APP_ID) if APP_ID else None


#the first tool 
search_tool = TavilySearch(max_results=3)
#the wether toole
@tool
def weather_check( city: str) -> str:
    """get the current weather for a given city like London, Paris"""
    api = weather_api_key
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api}&units=metric")
    data = response.json()
    if "main" in data:
        return data
    else: 
        return f"Error: {response}" 


#math toole
@tool
def universal_math_solver(query: str) -> str:
    """
    Solves any mathematical query, equation, calculus problem, or scientific calculation.
    Input can be raw text, natural language, or a LaTeX string.
    Returns the direct computational result.
    """
    if not wolfram_client:
        return "Error: Wolfram|Alpha client is not configured. Missing AppID."
    
    try:
        # 1. Fire the query to the engine
        res = wolfram_client.query(query)
        
        # Handle cases where Wolfram didn't understand the input at all
        if getattr(res, "success", "false") == "false" or not hasattr(res, "pods"):
            return "Error: The mathematical query could not be resolved by the engine."
            
        output_lines = []
        
        # 2. Production Loop: Extract data dynamically without hardcoding math terms
        for pod in res.pods:
            # Always capture the primary answer pod if Wolfram explicitly flags it
            if pod.primary:
                primary_text = "\n".join([sub.plaintext for sub in pod.subpods if sub.plaintext])
                if primary_text:
                    return f"Result ({pod.title}): {primary_text}"
            
            # Fallback collection: gathering context just in case 'primary' isn't flagged
            for subpod in pod.subpods:
                if subpod.plaintext:
                    output_lines.append(f"{pod.title}: {subpod.plaintext}")
                    
        # 3. Smart Fallback: If no single 'primary' pod was flagged, return the top clear data points
        if output_lines:
            # Return the first 3 relevant mathematical breakdowns (e.g., Input, Result, Plot data)
            return "\n".join(output_lines[:3])
            
        return "Calculation completed, but no text-based result could be extracted."
        
    except Exception as e:
        return f"Production Engine Error: {str(e)}"


@tool
def query_conversation_graph(entity: str) -> str:
    """
    Search past conversations stored in memory by a keyword or topic.
    Use this when the user asks about previous questions, past searches, or conversation history.
    Examples: 'weather', 'Paris', 'math', 'my name', 'what did I ask about'
    """
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)).session(database=NEO4J_DATABASE) as session:
        result = session.run("""
            MATCH (t:Thread)-[:HAS_CHANNEL]->(cs:ChannelState)
            WHERE cs.channel = 'messages'
            AND toLower(cs.blob) CONTAINS toLower($entity)
            RETURN t.thread_id AS thread_id,
                   cs.channel AS channel,
                   cs.blob AS blob
            ORDER BY cs.version DESC
            LIMIT 5
        """, entity=entity)

        records = [r.data() for r in result]

        if not records:
            return f"No past conversations found mentioning '{entity}'."

        output = []
        for r in records:
            output.append(
                f"Thread: {r.get('thread_id')} | "
                f"Content: {r.get('blob', '')[:300]}"
            )

        return "\n\n".join(output)


my_tools = [search_tool, weather_check, universal_math_solver, query_conversation_graph]

#starting our LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key).bind_tools(my_tools)

print("the llm working fine")


#####################################################################################################################################
############################################ THE SELF CORECTION LOOP USING STANDER LOOP  ############################################




#####################################################################################################################################
############################################ THE SELF CORECTION LOOP USING THE LANGGRAPH ############################################ 
class AgentState(TypedDict):
    query : str
    current_query : str
    messages : str
    best_answer: str
    score: float
    best_score: float
    critique: str
    tool_used: str
    passed: bool
    retry_count: int

llm_synthesizer = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0, 
    groq_api_key=groq_api_key_2
)
#initialize the LLM (e.g., Llama-3.3 on Groq)
groq_model_1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key=groq_api_key_3
)
groq_model_2 = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key=groq_api_key_4
)
#the validation chain
validation_promt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict answer validator. 
        You will receive a user query, the tool used, and the agent's answer.
        Score the answer from 0 to 10 based on:
        - Relevance to the query
        - Groundedness (no hallucination)
        - Completeness
        - Correct tool usage
        Respond ONLY with valid JSON, no explanation, no markdown:
        {{"score": <float>, "passed": <bool>, "critique": "<what is wrong or missing>"}}
        passed is true if score >= 7.5, false otherwise."""),
    ("human", "Query: {user_query}\nTool Used: {tool}\nAnswer: {answer}")
])


#the reformulation chain
reformulation_promt = ChatPromptTemplate.from_messages([
    ("system", """You are a query reformulator.
    Given the original query, the previous failed answer, and the critique explaining why it failed,
    rewrite the query to be clearer and more specific so the agent can answer it correctly.
    Respond ONLY with the reformulated query as plain text."""),
    ("human", "Original Query: {query}\nPrevious Answer: {previous_answer}\nCritique: {critique}")
])

#the chain of processe 
reformulation_chain = reformulation_promt | groq_model_1 | StrOutputParser()
validation_chain = validation_promt | groq_model_2 | JsonOutputParser()

def llm_call(state: AgentState)-> dict:
    """Initial LLM attempt."""
    """
    - Fix #5: fallback to 'query' if 'current_query' not yet set
    - Fix #2: wrap in HumanMessage list for correct input type
    - Fix #4: execute tools if the LLM requests them
    - Fix #3: always store plain string in state["messages"]
    """
    # Fix #5 — safe query resolution
    query = state.get("current_query") or state["query"]

    # Fix #2 — correct input type
    response = llm.invoke([HumanMessage(content=query)])

    # Fix #4 — actually execute tools if requested
    if response.tool_calls:
        tool_node = ToolNode(tools=my_tools)
        tool_results = tool_node.invoke({"messages": [response]})
        tool_used = response.tool_calls[0]["name"]

        # build full conversation history for final synthesis
        full_messages = [
            HumanMessage(content=query),
            response,                        # AIMessage with tool_calls
            *tool_results["messages"]        # ToolMessage(s) with results
        ]

        # let the LLM synthesize a final answer from tool output
        final_response = llm_synthesizer.invoke(full_messages)

        # Fix #3 — store plain string
        return {"messages": final_response.content, "tool_used": tool_used}

    # Fix #3 — no tool call, still store plain string
    return {"messages": response.content, "tool_used": "model_only"}

def validation_node(state: AgentState)-> dict:
    """Validator evaluates the answer."""
    try:
        result = validation_chain.invoke({"user_query": state["query"], "tool": state["tool_used"], "answer":state["messages"]})
        score = result.get("score", 0.0)
        best_score = state.get("best_score", -1)
        best_answer = state.get("best_answer", "")
        critique = result.get("critique", "")
        passed = result.get("passed", False)

        if score > best_score:
            best_score = score
            best_answer = state["messages"]

        return{
            "score": score,
            "best_score": best_score,
            "best_answer": best_answer,
            "critique": critique,
            "passed": passed
        } 
    except Exception as e:
        return {"score": 0.0, "passed": False, "critique": f"Validator error: {str(e)}"}    

def reformulation_node(state: AgentState)-> dict:
    """Reformulates the query if validation fails"""

    result = reformulation_chain.invoke({"query":state.get("current_query") or state["query"],
                                        "previous_answer":state["messages"],
                                        "critique": state["critique"] })
    
    return {"current_query": result.strip(), "retry_count": state.get("retry_count", 0) + 1}

def route_after_validation(state: AgentState)-> dict:
    """Define the path"""
    try:
        score = state["score"]
        if state.get("passed") or state.get("score", 0) >= 7.5:
            return "to_end"
        if state.get("retry_count", 0)>=2:
            return "to_end"
        return "to_reformulation"

    except Exception as e:
        print(f"Error in router: {e}. Defaulting to safe path.")
        return "to_reformulation"      

    
#now the grapht
builder = StateGraph(AgentState)
#node section
builder.add_node("llm_call", llm_call)
builder.add_node("validation_node", validation_node)
builder.add_node("reformulation_node", reformulation_node)

#edges section
builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", "validation_node")
builder.add_conditional_edges(
    "validation_node",
    route_after_validation,
    {
        "to_end":END,
        "to_reformulation": "reformulation_node"
    }
)
builder.add_edge("reformulation_node", "llm_call")

# create the driver manually
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# create checkpointer directly from the driver
checkpointer = Neo4jSaver(driver, database=NEO4J_DATABASE)

# setup and compile
checkpointer.setup()

#and now compilate with the langgraph checkpointer
graph = builder.compile(checkpointer= checkpointer)

#now let's create a configurable so each conversation have a unique thread
my_configuration = {'configurable':{"thread_id": str(uuid.uuid4())}}





























                    





            



    

"""
Validator returns: { score: 0-10, passed: bool, critique: "Missing X, hallucinated Y" }
Next attempt gets: original query + previous answer + critique

Score on: relevance, groundedness, completeness, tool usage correctness
"""








"""
#nwo let's builde the mean core that will structer the user query
class SearchQuery(BaseModel):
    search_query : str=Field(None, description="Query that optimize web search") 
    jsutifucation : str = Field(None, json_schema_extra={"justification": "Why this query is relevant to the user's request."})

structer_llm = llm.with_structured_output(SearchQuery)
"""
"""#now let's test that 
respons = structer_llm.invoke("How does Calcium CT score relate to high cholesterol?")
tru_respons = llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(respons)
print("\n", tru_respons.content)
print("\n",respons.search_query)
print("\n",respons.jsutifucation)"""
"""graph = my_builder.compile()

respons = graph.invoke({"query":"How does Calcium CT score relate to high cholesterol?"})
print(respons['response'])"""


"""
user_input = {"messages": [HumanMessage(content="Can you check if this equation holds true: \(\int _{-\infty }^{\infty }e^{-x^{2}}\,dx=\sqrt{\pi }\)?")]}
    response = graph.invoke(user_input, config=my_configuration)
    ##########################################################""
    # 2. Get the last two messages
    all_messages = response["messages"]
    final_answer = all_messages[-1].content
    previous_message = all_messages[-2]

    # 3. Check if the previous message was a tool result
    # (LangChain wraps tool outputs in a specific 'ToolMessage' class)
    if previous_message.type == "tool":
        # Extract the actual name of the tool from the ToolMessage object
        tool_name = getattr(previous_message, "name", "Unknown Tool")
        print(f"Tool Used: The model successfully called '{tool_name}'!")
    else:
        print("Model Only: The model answered directly from its own knowledge.")

    print(f"\nAnswer:\n{final_answer}")
"""