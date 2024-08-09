from typing import List, Optional
from pydantic import BaseModel, Field

class NewsPost(BaseModel):
    id: str = Field(..., alias='_id')
    title: str
    link: str
    source: str
    summary: str
    content: Optional[str] = None
    published: str
    extracted_time: str
    embedded: bool = False


class ContentUpdate(BaseModel):
    title: str
    link: str
    src: str
    content: Optional[str] = None
    date: str
    src_uuid: str
    embedding: Optional[List[float]] = None


class Content(BaseModel):
    type: str
    language: str = None
    base: str
    value: str

class RedditPost(BaseModel):
    id: str = Field(..., alias='_id')
    title: str
    link: str
    subreddit: str
    summary: str
    content: List[Content]
    published: str
    extracted_time: str
    embedded: bool = False


class RedditFeed(BaseModel):
    type: str
    posts: List[RedditPost]

