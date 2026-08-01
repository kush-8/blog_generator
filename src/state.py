from pydantic import BaseModel, Field

class BlogState(BaseModel):
    # User input fields
    topic: str = Field(description="The topic of the blog post")
    audience: str = Field(description="The target audience of the blog post")

    # research fields
    research_content: str = Field(default="", description="The research content for the blog post")
    research_feedback: str = Field(default="", description="Feedback on the research content")

    # draft fields
    draft: str = Field(default="", description="A draft of the blog post")
    draft_feedback: str = Field(default="", description="Feedback on the draft of the blog post")

    # Editor Output fields
    final_blog_post: str = Field(default="", description="The final version of the blog post after editing")

    # Metadata fields
    revision_count: int = Field(default=0, description="The number of revisions made to the blog post")