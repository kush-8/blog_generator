from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from typing import Literal

from src.state import BlogState, RevisionNotes
from src.agents import research_agent, editor_agent, writer_agent, get_llm

MAX_REVISIONS = 3

def get_feedback_notes(llm: ChatGroq, feedback: str) -> RevisionNotes:
    st_llm = llm.with_structured_output(RevisionNotes)
    prompt = f"""
    From given feedback, return whether the human approved the research content or not. 
    If not, provide notes on what the issues are and how to improve it. 
    In case of no feedback, return approved.
    Feedback: {feedback}
    """
    response = st_llm.invoke(prompt)
    return response.content

def researcher(state: BlogState):
    """
    It uses the research_agent to gather information based on the topic and audience provided in the BlogState.
    The gathered information is then stored in the research_content field of the BlogState.
    """
    llm = get_llm()
    research_content = research_agent(
        llm = llm, 
        topic=state.topic, 
        audience=state.audience, 
        human_feedback=state.research_feedback
    )
    state.research_content = research_content
    return state

def research_review(state: BlogState):
    """
    Pause and ask the human for feedback on the research content. If the human approves, continue to the next step. If not, raise an interrupt with the revision notes.
    """
    decision = interrupt({
        "stage": "research_review",
        "research_content": state.research_content,
        "instructions":(
            "Please provide feedback on the research content." ,
            "If you approve, please give a clear one word response 'approved'."
        )
    })
    feedback_notes = get_feedback_notes(get_llm(), decision)
    state.research_feedback = "" if feedback_notes.approved else feedback_notes.notes
    return state

def writer(state: BlogState):
    """
    Uses the research content to generate a draft of the blog post. The draft is stored in the draft field of the BlogState.
    """
    llm = get_llm()
    draft = writer_agent(
        llm = llm,
        research_content=state.research_content,
        topic=state.topic,
        audience=state.audience,
        revision_notes=state.draft_feedback
    )
    state.draft = draft
    return state

def draft_review(state: BlogState):
    """
    Pause and ask the human for feedback on the draft. If the human approves, continue to the next step. If not, raise an interrupt with the revision notes.
    """
    decision = interrupt({
        "stage": "draft_review",
        "draft": state.draft,
        "instructions":(
            "Please provide feedback on the draft." ,
            "If you approve, please give a clear one word response 'approved'."
        )
    })
    feedback_notes = get_feedback_notes(get_llm(), decision)
    state.draft_feedback = "" if feedback_notes.approved else feedback_notes.notes
    return state

def editor(state: BlogState):
    """
    Gives the final blog post.
    """
    llm = get_llm()
    final_blog_post = editor_agent(
        llm = llm,
        draft=state.draft,
        topic=state.topic,
        audience=state.audience
    )
    state.final_blog_post = final_blog_post
    return state

def research_review_router(state: BlogState)-> Literal["researcher", "writer"]:
    """
    Route the flow based on the human feedback on the research content.
    If the human approves, continue to the next step. If not, raise an interrupt with the revision notes.
    """
    if state.research_feedback:
        return "researcher"
    return "writer"

def draft_review_router(state: BlogState)-> Literal["editor", "writer"]:
    """
    Route the flow based on the human feedback on the draft.
    If the human approves, continue to the next step. If not, raise an interrupt with the revision notes.
    """
    if state.draft_feedback and state.revision_count < MAX_REVISIONS:
        return "writer"
    return "editor"

## Build and compile the graph
def build_graph():
    graph = StateGraph(BlogState)

    ## add nodes
    graph.add_node("researcher", researcher)
    graph.add_node("research_review", research_review)
    graph.add_node("writer", writer)
    graph.add_node("draft_review", draft_review)
    graph.add_node("editor", editor)

    ## add edges
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "research_review")
    graph.add_conditional_edges("research_review", research_review_router, 
                                {
        "researcher": "researcher",
        "writer": "writer"
    })
    graph.add_edge("writer", "draft_review")
    graph.add_conditional_edges("draft_review", draft_review_router, 
                                {
        "writer": "writer",
        "editor": "editor"
    })
    graph.add_edge("editor", END)
    return graph.compile(checkpointer=InMemorySaver())
