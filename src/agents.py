from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import wikipedia
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import create_agent


def get_llm(model_name: str = "openai/gpt-oss-20b", temperature: float = 0.5):
    llm = ChatGroq(model=model_name, temperature=temperature)
    return llm


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a given query and return a summary of the results.
    """
    wiki = WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=3)
    result = wiki.run(query)
    return result

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search DuckDuckGo for a given query and return the top results.
    """
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result

@tool
def arxiv_search(query: str) -> str:
    """
    Search arXiv for a given query and return the top results.
    """
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    result = arxiv.run(query)
    return result

research_tools = [wikipedia_search, duckduckgo_search, arxiv_search]


RESEARCHER_PROMPT = """
    You are a professional research assistant.

    Given a topic, target audience, and additional instructions, your job is to gather factual, relevant, and reliable information about the topic. 
    You should use the following tools to gather information:
    - Wikipedia search tool for background knowledge.
    - DuckDuckGo search tool for recent developments.
    - Arxiv search tool for academic papers.
    Guidelines:
    - Never write the final article.
    - Return concise research notes.
    - Include sources whenever possible.
"""

def research_agent(llm: ChatGroq, topic: str, audience: str, human_feedback: str = "") -> str:
    """
    Research agent that generates a research report based on the given topic and target audience.
    """
    agent = create_agent(
        model=llm,
        tools=research_tools,
        system_prompt=RESEARCHER_PROMPT,
    )
    if human_feedback:
        human_feedback = f"HThese are the instructions based on the human feedback on previous research:{human_feedback}"
    else:
        human_feedback = "No additional instructions provided."
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nTarget Audience: {audience}\n {human_feedback}"
                }
            ]
        }
    )
    return response["messages"][-1].content

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
            You are a professional writer. 
            You will be given research notes and your job is to write a well-structured, engaging, and informative article based on the research notes.
            Guidelines:
            -Length: The article should be between 800 and 1200 words.
            -Structure: The article should have a clear introduction, body, and conclusion and overall 3-5 sections at least. Use headings and subheadings where appropriate.
            -Style: The article should be written in a style that is clear, friendly and appropriate for the target audience
            -Formatting: Use markdown formatting
            -do NOT add a 'word Count' line at the end of the article.
        """
    },
    {
        "role": "user",
        "content": """
            Topic: {topic}
            Target Audience: {audience}
            Research Notes: {research}

            {revision_notes}

            write the full blog post.
        """
    }
])

def writer_agent(llm: ChatGroq, topic: str, audience: str, research_content: str, revision_notes: str = "") -> str:
    """
    Writer agent that generates a blog post based on the given research content and target audience.
    """
    if revision_notes:
        revision_notes = f"These are the instructions based on the human feedback on previous draft: {revision_notes}. Make sure to follow these properly."
    else:
        revision_notes = "No additional instructions provided."
    chain = WRITER_PROMPT | llm
    response = chain.invoke(
        {
            "topic": topic,
            "audience": audience,
            "research": research_content,
            "revision_notes": revision_notes
        }
    )
    return response.content


EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
            You are an editor agent -- the final quality control for the blog post. 
            Take the draft and produce the FINAL polished version.
            Specifically, you should:
            - Fix grammar, spelling, awkward phrasing, and punctuation errors.
            - Improve flow and transition between sections.
            - Ensure the article is clear, concise, and engaging.
            - make the title and intro more compelling and attention-grabbing if needed.
            - keep the same structure and markdown formatting as the original draft.
            - Blog wordings should look human-written, not AI-generated. Don't use any special chars and complex/uncommon words that are not used in daily life.
            Output only the final polished version of the blog post, do not include any additional commentary or notes.
        """
    },
    {
        "role": "user",
        "content": """
            Topic: {topic}
            Audience: {audience}
            Draft: {draft}
        """
    }
])

def editor_agent(llm: ChatGroq, topic: str, audience: str, draft: str) -> str:
    """
    Editor agent that polishes the blog post for final quality control.
    """
    chain = EDITOR_PROMPT | llm
    response = chain.invoke(
        {
            "topic": topic,
            "audience": audience,
            "draft": draft,
        }
    )
    return response.content









