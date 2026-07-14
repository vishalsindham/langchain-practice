from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args :
        query : The query to search for
    Returns :
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from Langchain agent.")
    result = agent.invoke(
        {"messages": HumanMessage(content="How is the weather in Tokyo.")}
    )
    print(result)


if __name__ == "__main__":
    main()
