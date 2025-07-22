from langchain_tavily import TavilySearch
from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """
    Perform a live web search for up-to-date information and return raw content from top results.

    The output may contain multiple paragraphs of raw data.
    It is your job to:
    - Clean and synthesize the useful information.
    - Extract relevant points that answer the user's question.
    - Discard repeated, irrelevant, or low quality content.
    - Translate, explain, or summarize information as needed, depending on user prompt.

    Always assume the output may be noisy and requires processing before showing to the user.

    Args:
        query: The search query string.

    Returns:
    Raw combined content from top search results. You must post-process this information.
    """
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
