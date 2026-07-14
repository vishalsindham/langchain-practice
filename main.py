from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)



def main():

    print("Hello from Langchain agent.")

    dynamic_query = """ This is a response from a LLM model. parse this and format this properly which can be read by human.
        Below is the response. Just return the answer part
        {context_data} 
    """
    final_prompt = PromptTemplate(
        template=dynamic_query
    )
    localmodel = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.6)

    chain = final_prompt | localmodel

    result = agent.invoke(
        {"messages": HumanMessage(content="How is the weather in Tokyo.")}
    )
    context_data = f"""{result}"""

    response = chain.invoke(input={"context_data": context_data})

    print(response)


if __name__ == "__main__":
    main()
