from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Update(BaseModel):
    """
    Represents an AI news update, including the title and verbatim content.
    """
    title: str = Field(description="title of the AI news update")
    content: str = Field(description="content of the update quoted from the original source verbatim")
    link: str = Field(description="URL of the hyperlink")
    # List of hyperlinks in the content


# Create a type for a list of updates
class UpdateList(BaseModel):
    """
    A list containing multiple AI news updates.
    """
    updates: List[Update] = Field(description="list of AI news updates")


class Newsletter(BaseModel):
    """
    Represents an AI news update newsletter.
    """
    uuid: str
    from_email: str
    subject: str
    # Make it UTC
    date: datetime
    email_body: str
    message_html: str


class NewsletterUpdate(BaseModel):
    """
    Represents the entire data of an AI news update.
    """
    uuid: str = Field(description="unique key of the AI news update newsletter")
    title: str = Field(description="title of the AI news update")
    content: str = Field(description="content of the update quoted from the original source verbatim")
    date: datetime = Field(description="date of the email message containing the AI news update")
    src: str = Field(description="source of the AI update")
    src_uuid: str = Field(description="unique key of the AI news update newsletter")
    # Optional Link
    link: Optional[str] = Field(description="URL of the hyperlink")


class NewsletterUpdateEmbedded(NewsletterUpdate):
    """
    Represents the entire data of an AI news update after processing by the Embedder Class.
    """
    lemmatized: str = Field(description="lemmitized content of the update")
    resolved_link: Optional[str] = Field(description="resolved link of the hyperlink")
    embedding: List[float] = Field(description="embedding of the update")
