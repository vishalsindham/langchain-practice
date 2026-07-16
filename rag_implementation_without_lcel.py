import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite") | StrOutputParser()

vectorstore = PineconeVectorStore(
    embedding=embeddings,index_name=os.environ["INDEX_NAME"]
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:
"""
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================


def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # step 1 Retrieve relevant documents
    docs = retriever.invoke(query)

    # step 2 : Format documents into context string
    context = format_docs(docs)

    # step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages) 

    # step 5: return the content
    return response

if __name__ == "__main__":
    print("Retrieving")

    query = "What is Pinecone in machine Learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result)

