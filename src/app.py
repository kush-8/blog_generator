import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from langgraph.types import Command
import sys

# Ensure src directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from graph import build_graph

load_dotenv()

st.set_page_config(page_title="AI Blog Generator", page_icon="📝", layout="wide")

st.title("📝 AI Blog Generator")
st.markdown("Generate a comprehensive blog post using AI agents for researching, writing, and editing.")

# Initialize session state for thread_id and graph
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    
if "started" not in st.session_state:
    st.session_state.started = False

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "processing_msg" not in st.session_state:
    st.session_state.processing_msg = ""

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# If we are in the processing state, display spinner and execute graph
if st.session_state.is_processing:
    with st.status(st.session_state.processing_msg) as status:
        if "resume_payload" in st.session_state:
            stream_input = Command(resume=st.session_state.resume_payload)
            del st.session_state["resume_payload"]
        else:
            stream_input = {"topic": st.session_state.topic, "audience": st.session_state.audience}
            
        for event in st.session_state.graph.stream(stream_input, config=config):
            for key, value in event.items():
                if key != "__end__":
                    status.update(label=f"Completed stage: {key}...", state="running")
                    
        status.update(label="Process paused or completed!", state="complete")
        
    st.session_state.is_processing = False
    st.rerun()

# The UI rendering
if not st.session_state.started:
    st.subheader("Start a new Blog")
    
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "audience" not in st.session_state:
        st.session_state.audience = "general public"
        
    topic = st.text_input("Blog Topic", value=st.session_state.topic, placeholder="Enter your topic here...")
    audience = st.text_input("Target Audience", value=st.session_state.audience)
    
    if st.button("Generate Blog"):
        if topic:
            st.session_state.topic = topic
            st.session_state.audience = audience
            st.session_state.started = True
            st.session_state.is_processing = True
            st.session_state.processing_msg = "Initializing Agents and Researching..."
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        else:
            st.error("Please enter a topic first.")
else:
    # Get current state
    try:
        snap = st.session_state.graph.get_state(config)
    except Exception:
        snap = None

    if snap and snap.next:
        # We are interrupted and waiting for input
        node = snap.next[0]
        interrupt_payload = snap.interrupts[0].value
        
        st.info(f"Paused for Human Review: **{node}**")
        
        if node == "research_review":
            st.subheader("Research Content")
            st.markdown(interrupt_payload.get("research_content", ""))
            
            st.write("---")
            st.subheader("Review Research")
            instructions = interrupt_payload.get("instructions", [])
            if isinstance(instructions, tuple) or isinstance(instructions, list):
                for inst in instructions:
                    st.write(inst)
            else:
                st.write(instructions)
                
            feedback = st.text_area("Feedback (type 'approved' to proceed without changes):", key="research_feedback")
            
            if st.button("Submit Research Feedback"):
                st.session_state.resume_payload = feedback
                st.session_state.is_processing = True
                st.session_state.processing_msg = "Processing feedback and drafting..."
                st.rerun()
                    
        elif node == "draft_review":
            st.subheader("Draft Content")
            st.markdown(interrupt_payload.get("draft", ""))
            
            st.write("---")
            st.subheader("Review Draft")
            instructions = interrupt_payload.get("instructions", [])
            if isinstance(instructions, tuple) or isinstance(instructions, list):
                for inst in instructions:
                    st.write(inst)
            else:
                st.write(instructions)
                
            feedback = st.text_area("Feedback (type 'approved' to proceed without changes):", key="draft_feedback")
            
            if st.button("Submit Draft Feedback"):
                st.session_state.resume_payload = feedback
                st.session_state.is_processing = True
                st.session_state.processing_msg = "Processing feedback and editing..."
                st.rerun()

    elif snap and not snap.next and snap.values.get("final_blog_post"):
        st.success("Blog Generation Complete!")
        st.subheader("Final Blog Post")
        st.markdown(snap.values["final_blog_post"])
        
        st.write("---")
        if st.button("Generate Another Blog"):
            st.session_state.started = False
            st.session_state.topic = ""
            st.rerun()
