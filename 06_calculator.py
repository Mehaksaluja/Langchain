from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature = 0
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers"""
    return a*b

model_with_tools = model.bind_tools([multiply])

response = model_with_tools.invoke(
    "What is the multiplication of 10 and 78"
)

print("Content:", response.content)
print("Tool calls:", response.tool_calls)

for tool_call in response.tool_calls:
    result = multiply.invoke(tool_call)
    print("Tool result:", result)