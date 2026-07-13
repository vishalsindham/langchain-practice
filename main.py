from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    context_data = """
        ERROR 2026-07-13 14:32:01 org.apache.spark.deploy.yarn.ApplicationMaster: 
        User class threw exception: org.apache.spark.sql.AnalysisException: 
        Cannot write incompatible data to table 'production.user_conversions':
        - Cannot safely cast 'signup_timestamp' from StringType to TimestampType.
        - Missing target column 'referral_code' in source dataframe.
        Command executed by user_id: 9942. Cluster allocation: 16 nodes, 64GB RAM.

"""
    user_query = """
             Analyze this log snippet. {context_data}  
            List the exact schema mismatches found, and tell me if this is a cluster sizing issue or a data type compatibility issue.
"""

    prompt_template = PromptTemplate(
        template=user_query,
        input_variables=["context_data"]
    )

    llm = ChatOllama(model="qwen2.5-coder:1.5b",temperature=0.4)

    chain = prompt_template | llm

    response = chain.invoke(input={"context_data": context_data})

    print(response.content)
    
if __name__ == "__main__":
    main()