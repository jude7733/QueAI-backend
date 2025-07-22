from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia and returns a 2-sentence summary of the most relevant result.
    Catches common issues such as redirects and disambiguation.
    """

    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000,
    )
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = wikipedia_tool.run(query)
    return result or "No relevant information found on Wikipedia."
