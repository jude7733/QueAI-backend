from typing import List, TypedDict
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.retrievers import ArxivRetriever
from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import getpass
import os
import dotenv
from pydantic import BaseModel, Field

dotenv.load_dotenv()


class SearchInput(BaseModel):
    """Basic input validation for web search"""

    query: str = Field(..., description="Search query string")


class CustomState(TypedDict):
    sources: List[str]


@tool(
    description="Use this tool to fetch live information from the web",
    args_schema=SearchInput,
)
def search_web(query: str) -> str:
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

    web_tool = TavilySearch(
        max_results=3,
        # topic="general",
        # include_answer=True,
        # include_raw_content=False,
        include_images=False,
        include_image_descriptions=True,
        search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )

    try:
        writer = get_stream_writer()
        search_results = web_tool.invoke(query)
        urls = [item["url"] for item in search_results["results"]]
        writer(
            {
                "data": {"sources": urls, "images": search_results["images"]},
                "type": "tool",
            }
        )

        return search_results["results"]

    except Exception as e:
        error_message = f"An error occurred while searching the web: {str(e)}"
        print(f"-> Error: {error_message}")
        return error_message


@tool(
    description="Searches Wikipedia and returns top 2 results.",
    args_schema=SearchInput,
)
def search_wiki(query: str) -> str:
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
    )
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = wikipedia_tool.run(query)
    return result or "No relevant information found on Wikipedia."


def format_arxiv_docs(docs: List[Document], max_chars: int = 1000) -> str:
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = meta.get("Title") or "Untitled"
        summary = meta.get("Summary") or meta.get("summary") or ""
        # NOTE: limiting content to first 1000 char result in only only showing intro metadata
        content = (doc.page_content or "")[:max_chars]
        blocks.append(f"### [{i}] {title}\nSummary: {summary}\n\ncontent:\n{content}")
    return "\n\n---\n\n".join(blocks)


@tool(
    description="Searches Arxiv and return maximum 2 results.",
    args_schema=SearchInput,
)
def search_arxiv(query: str) -> str:
    retriever = ArxivRetriever(
        load_max_docs=2,
        get_full_documents=True,
    )

    result_docs = retriever.invoke(query)
    return format_arxiv_docs(result_docs)


if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = search_arxiv.invoke({"query": query})
    print(results)
