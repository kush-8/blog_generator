from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import requests
import wikipediaapi
import arxiv
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import create_agent
from datetime import datetime

year = datetime.now().year


def get_llm(model_name: str = "openai/gpt-oss-20b", temperature: float = 0.5):
    llm = ChatGroq(model=model_name, temperature=temperature)
    return llm


@tool
def wikipedia_search(query: str) -> list[dict]:
    """
    Search Wikipedia and return the top 3 matching pages.
    """
    print(f"Searching Wikipedia for: {query}")

    USER_AGENT = "BlogGenerator/1.0 (kushagarsharma731@gmail.com)"

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent=USER_AGENT,
    )

    HEADERS = {
        "User-Agent": USER_AGENT,
    }

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        headers=HEADERS,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3,
        },
        timeout=10,
    )

    response.raise_for_status()

    search_results = response.json()["query"]["search"]

    pages = []

    for result in search_results:
        page = wiki.page(result["title"])

        if page.exists():
            pages.append(
                {
                    "title": page.title,
                    "summary": page.summary,
                    "url": page.fullurl,
                }
            )

    return pages

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search DuckDuckGo for a given query and return the top results.
    """
    print(f"Searching DuckDuckGo for: {query}")
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result

client = arxiv.Client()

@tool
def arxiv_search(query: str) -> list[dict]:
    """
    Search arXiv and return the top 3 papers.
    """
    print(f"Searching arXiv for: {query}")
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for paper in client.results(search):
        papers.append(
            {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published.date()),
                "summary": paper.summary[:700],
                "url": paper.entry_id,
            }
        )

    return papers

research_tools = [wikipedia_search, duckduckgo_search, arxiv_search]


RESEARCHER_PROMPT = """
You are an expert Research Analyst for a professional publishing company.

Your ONLY responsibility is to collect factual information and produce a structured research report.

You are NOT writing the final article.

=========================
AVAILABLE TOOLS
=========================

Wikipedia
Use for:
- definitions
- background
- history
- terminology

DuckDuckGo
Use for:
- recent developments
- statistics
- surveys
- case studies
- industry examples
- government reports

arXiv
Use for:
- academic papers
- scientific findings
- technical explanations

=========================
TOOL USAGE RULES
=========================

Only call a tool when additional information is needed.

You may call the same tool multiple times.

Maximum total tool calls: 6.

Never call tools unnecessarily.

If sufficient information has already been collected,
produce the final report.

=========================
IMPORTANT
=========================

DO NOT output your reasoning.

DO NOT explain your research process.

DO NOT write things like:

- "I'll search..."
- "Let's look up..."
- "I need more information..."
- "I'll now use Wikipedia..."
- "Next I'll search..."

Your response must ALWAYS be one of the following:

1. A valid tool call

OR

2. The final research report

Never output planning text.

=========================
RESEARCH OBJECTIVES
=========================

Gather enough information for a professional blog article.

Collect:

- definitions
- background
- important concepts
- latest developments
- statistics
- academic evidence
- practical examples
- benefits
- challenges
- ethical concerns
- future outlook
- actionable recommendations

Prefer evidence over opinions.

When possible include:

- publication year
- organization
- survey numbers
- percentages
- paper titles

=========================
OUTPUT FORMAT
=========================

# Executive Summary

# Background

# Key Concepts

# Current Trends

# Statistics & Data

# Academic Research

# Industry Examples

# Benefits

# Challenges

# Ethical Considerations

# Future Outlook

# Practical Recommendations

# References

=========================
QUALITY
=========================

The report should be:

- factual
- well-organized
- comprehensive
- concise
- evidence-based
- neutral

Never invent facts.

If sources disagree, mention both viewpoints.

Cite every important claim with its source.
"""

def research_agent(llm: ChatGroq, topic: str, audience: str, human_feedback: str = "", year: int = year) -> str:
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
                    "content": f"""Topic:
                                    {topic}

                                    Audience:
                                    {audience}

                                    Your task:

                                    1. Collect enough factual information for an 800–1200 word article.
                                    2. Use tools only when needed.
                                    3. Use at most 6 tool calls.
                                    4. Do not reveal your reasoning.
                                    5. Return only the final research report.
                                    Prioritize information from {year} whenever possible."""
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









