from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import dotenv

dotenv.load_dotenv()

web_tool = TavilySearch(
    max_results=2,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)


@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web using Tavily and returns the formatted content from the results.
    This tool does NOT summarize; it extracts and combines the raw content found.

    Args:
        query: The search query string.

    Returns:
        A single string containing the content of all search results,
        separated by newlines, or an error message.
    """
    try:
        search_results = web_tool.invoke(query)

        # Check if the search returned any results
        if not search_results:
            print("-> No results found.")
            return "No results found."

        content_values = [
            item["content"]
            for item in search_results.get("results", [])
            if "content" in item
        ]
        return "\n\n".join(content_values)

    except Exception as e:
        # Catch any other exceptions during the process
        error_message = f"An error occurred while searching the web: {str(e)}"
        print(f"-> Error: {error_message}")
        return error_message
