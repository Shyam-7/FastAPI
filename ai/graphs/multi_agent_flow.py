from langgraph.graph import StateGraph, END
from ..agents.retrieval_agent import get_retrieval_agent
from ..chains.analysis_chain import get_analysis_chain
from typing import TypedDict, Annotated, Any
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    brand: str
    model: str
    age: str
    mileage: str
    history: Annotated[list[BaseMessage], operator.add]
    recommendations: str

def create_multi_agent_flow(llm, vector_store):
    retrieval_agent = get_retrieval_agent(llm, vector_store)
    analysis_chain = get_analysis_chain(llm)

    def retrieval_node(state):
        history = retrieval_agent.run(f"What is the service history for a {state['brand']} {state['model']}?")
        return {"history": history}

    def analysis_node(state):
        recommendations = analysis_chain.run(
            brand=state['brand'],
            model=state['model'],
            age=state['age'],
            mileage=state['mileage'],
            history=state['history']
        )
        return {"recommendations": recommendations}

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("analysis_node", analysis_node)

    workflow.set_entry_point("retrieval_node")

    workflow.add_edge("retrieval_node", "analysis_node")
    workflow.add_edge("analysis_node", END)

    app = workflow.compile()
    return app
