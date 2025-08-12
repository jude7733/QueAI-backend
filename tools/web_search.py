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


@tool(
    description="Generic web search tool. Use this tool to fetch live information from the web",
    args_schema=SearchInput,
)
def web_search_tool(query: str) -> str:
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

    web_tool = TavilySearch(
        max_results=3,
        topic="general",
        include_answer=True,
        # include_raw_content=False,
        # include_images=False,
        # include_image_descriptions=False,
        search_depth="advanced",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )

    try:
        search_results = web_tool.invoke(query)
        urls = [item["url"] for item in search_results["results"]]
        print("-> Search URLs:", urls)

        return search_results["answer"]

    except Exception as e:
        error_message = f"An error occurred while searching the web: {str(e)}"
        print(f"-> Error: {error_message}")
        return error_message


if __name__ == "__main__":
    results = web_search_tool(query="Current temperature of kochi")
    print("Search Results:\n", results)
