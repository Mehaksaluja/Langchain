from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

model = init_chat_model(
    "openai:gpt-4o-mini",
    timeout = 1000,
    max_retries = 2,
    temperature = 0,
    max_tokens = 200
)

messages = [
    SystemMessage("You are a helpful AI teacher. Explain technical concepts simply.")
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower == "exit":
        break

    messages.append(HumanMessage(user_input))

    full_response = None
    print("\nAI: ", end="")

    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
        full_response = (
            chunk if full_response is None
            else full_response + chunk
        )
    print()

    messages.append(full_response)
