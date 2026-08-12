from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import os
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = "gpt-4o-mini"

def get_model():
    return init_chat_model(MODEL_NAME)

# SECTION 1: Three ways to send input
def section_1():
    model = get_model()

    print("Way 1: Send a single string as input")
    response = model.invoke("Hello, How are you?")
    print(response.text)

    print("\nWay 2: Send a list of messages as input")
    messages = [
        SystemMessage("You are a poetry expert. Keep answers to one line."),
        HumanMessage("Write a haiku about the ocean."),
    ]
    response = model.invoke(messages)
    print(response.text)

    print("\nWay 3: list of plain dictionaries")
    messages = [
        {"role": "system", "content": "You are a poetry expert. Keep answers to one line."},
        {"role": "user", "content": "Write a haiku about the ocean."},
    ]
    response = model.invoke(messages)
    print(response.text)


# SECTION 2: SystemMessage - steering behavior
def section_2():
    model = get_model()
    sys_message = SystemMessage("""You are a senior Python developer with expertise in web frameworks.
    Always provide a tiny code example. Be concise.""")
    response = model.invoke([sys_message, HumanMessage("How do I create a simple web server in Python?")])
    print(response.text)


# SECTION 3: HumanMessage with metadata
def section_3():
    human_message = HumanMessage(
        content="Write a haiku about the ocean.",
        name="Mehak Saluja",
        id="msg_456",
    )
    print(human_message.content)
    print(human_message.name)
    print(human_message.id)


# SECTION 4: AIMessage - attributes, usage, and injecting a fake turn
def section_4a():
    model = get_model()
    response = model.invoke("Write a haiku about the ocean.")
    print("type:", type(response))
    print("content:", response.text)  
    print("usage_metadata:", response.usage_metadata)
    print("id:", response.id)

def section_4b():
    model = get_model()
    ai_msg = AIMessage("I'd be happy to help you with that question!")
    messages = [
        SystemMessage("You are a helpful assistant"),
        HumanMessage("Can you help me?"),
        ai_msg,
        HumanMessage("Great! What's 2+2?")
    ]
    response = model.invoke(messages)
    print(response.text)


# SECTION 5: Tool calls inside AIMessage + full ToolMessage round trip
def get_weather(location: str) -> str:
    # This is a mock implementation of a weather tool.
    # In a real scenario, this function would call an external API to get the weather.
    return f"The current weather in {location} is sunny with a temperature of 25°C."

def section_5():
    model = get_model()
    model_with_tools = model.bind_tools([get_weather])
    ai_message = model_with_tools.invoke("What is the weather in New York?")
    print("Tool calls requested by model:", ai_message.tool_calls)

    if not ai_message.tool_calls:
        print("No tool calls were made by the model.")
        print(ai_message.text)
        return

    call = ai_message.tool_calls[0]
    result = get_weather(**call["args"])
    print("tool ran locally, result:", result)

    tool_message = ToolMessage(
        content=result,
        tool_call_id=call["id"]
    )
    final = model_with_tools.invoke([
        HumanMessage("What is the weather in New York?"),
        ai_message,
        tool_message,
    ])
    print("final answer:", final.text)


# SECTION 6: content vs content_blocks (works fully offline, no API call)
def section_6():
    anthropic_style = AIMessage(
        content=[
            {"type": "thinking", "thinking": "The user wants a joke...", "signature": "abc123"},
            {"type": "text", "text": "Why did the chicken cross the road?"},
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    print("Anthropic raw content:", anthropic_style.content)
    print("Anthropic normalized content_blocks:", anthropic_style.content_blocks)

    openai_style = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking..."}]},
            {"type": "text", "text": "Because it wanted to.", "id": "msg_1"},
        ],
        response_metadata={"model_provider": "openai"},
    )
    print("\nOpenAI raw content:", openai_style.content)
    print("OpenAI normalized content_blocks:", openai_style.content_blocks)


# SECTION 7: Multimodal - sending an image
def section_7():
    model = get_model()

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {
                "type": "image",
                "url": "https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg",
            },
        ],
    }
    response = model.invoke([message])
    print(response.text)


# SECTION 8: Streaming - watching tokens arrive + merging chunks
def section_8():
    model = get_model()

    full_message = None
    print("Streaming response:\n")
    for chunk in model.stream("Explain recursion in two short sentences."):
        print(chunk.text, end="", flush=True)
        full_message = chunk if full_message is None else full_message + chunk

    print("\n\n--- reassembled full message ---")
    print(full_message.text)


# BONUS: ToolMessage with extra data the model never sees
def section_9():
    message_content = "It was the best of times, it was the worst of times."
    artifact = {"document_id": "doc_123", "page": 0}

    tool_message = ToolMessage(
        content=message_content,
        tool_call_id="call_123",
        name="search_books",
        artifact=artifact,
    )
    print("content sent to model:", tool_message.content)
    print("artifact kept for your app only:", tool_message.artifact)


if __name__ == "__main__":
    # section_1()
    # section_2()
    # section_3()
    # section_4a()
    # section_4b()
    # section_5()
    # section_6()
    # section_7()
    # section_8()
    section_9()