from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from Langchain agent.")
    result = agent.invoke(
        {"messages": HumanMessage(content="How is the weather in Tokyo.")}
    )
    print(result)


if __name__ == "__main__":
    main()
