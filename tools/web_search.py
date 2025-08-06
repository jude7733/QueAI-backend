from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import getpass
import os
import dotenv

dotenv.load_dotenv()


# TODO: Add extractor
# TODO: Add types Basemodel
@tool
def web_search_tool(query: str) -> str:
    """
    Generic web search tool. Use this tool to fetch live information from the web.

    Args:
        query: The search query string.

    Returns:
        Final answer string from the search results.
    """
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
        # Catch any other exceptions during the process
        error_message = f"An error occurred while searching the web: {str(e)}"
        print(f"-> Error: {error_message}")
        return error_message


if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = web_search_tool(query)
    print("Search Results:\n", results)
