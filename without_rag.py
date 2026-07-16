from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "gemini-3.1-flash-lite"

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite") | StrOutputParser()



response = llm.invoke([HumanMessage(content="What is PineCone..?")]) 

print(response)