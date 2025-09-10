from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two integers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Web search is not available - TAVILY_API_KEY not configured"

    try:
        from langchain_tavily import TavilySearch

        os.environ["TAVILY_API_KEY"] = tavily_api_key

        tavily_tool = TavilySearch(max_results=3)

        results = tavily_tool.invoke({"query": query})

        if isinstance(results, dict) and 'results' in results:
            search_results = results['results']
            return "\n\n".join([
                f"{result.get('title', 'No title')}\n{result.get('content', 'No content')}"
                for result in search_results[:3]
            ])
        return str(results)

    except Exception as e:
        return f"Web search failed: {str(e)}"

def get_tools():
    tools = [add, subtract, multiply, divide]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(web_search)
    return tools