from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia and returns general answers.
    """

    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
    )
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = wikipedia_tool.run(query)
    return result or "No relevant information found on Wikipedia."


if __name__ == "__main__":
    query = "Python programming language"
    summary = search_wikipedia.invoke(query)
    print(summary)
