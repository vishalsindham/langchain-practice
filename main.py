from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain..!")
    information = """
The river flows gently through the valley, winding its way past ancient trees. The water is cool and clear, reflecting the bright blue sky above. As the sun sets, the river takes on a golden hue, offering a peaceful end to the day.
"""

    summary_template = """
    given the paragraph {paragraph} is a random data:
    1. count how many times river occures 
    2. share the characters in the word counted
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["paragraph"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.1-flash-lite")

    # Using Langchain expression language
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"paragraph": information})
    print(response.content)


if __name__ == "__main__":
    main()
