from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool


MAX_ITERATIONS=10
MODEL="qwen3:1.7b"


@tool
def write_file(text: str) -> str:
    """
    Take the text and write it to a file.
    args :
        text : str
    output : 
        str
    """
    with open("agent.txt", "w") as file:
        file.write(text)
    file.close()
    return "Content written"

tools_list = [write_file]
tools_name = {tool.name : tool for tool in tools_list}
llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0.4)

llm_with_tools = llm.bind_tools(tools_list)



messages = [
    SystemMessage(content= 
                  "You are helpful assistant"
                  "Perform the tasks given by user"
                  "STRICT RULES — you must follow these exactly:\n"
                  "1.When a user tells to write to a file use write_file tool"
                  "2. Pass the content to be written as argument"
                  "3. Check if your passing tool_calls"
                  "4. Check if all steps are followed"
                  ),
    HumanMessage(content=f"Hi")
]



chat_history = llm_with_tools.invoke(messages)
print(chat_history.content)
user_query = ""

while user_query != "bye":

    user_query = input("User :   ")
    messages.append(HumanMessage(content=f"{user_query}"))
    response = llm_with_tools.invoke(messages)
    print(f"LLM : \n {response.content}")
    messages.append(response)

    for tool_call in response.tool_calls:
        tool_result = tools_name[tool_call["name"]].invoke(tool_call)
        messages.append(tool_result)
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    print(response.content)
    print("--" * 100)
