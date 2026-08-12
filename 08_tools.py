import json
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

# SECTION 1 — Basic @tool
# Concept: @tool reads your function name, docstring, type hints

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


def section_1_basic_tool():
    print("name  :", search_database.name)
    print("desc  :", search_database.description)
    print("schema:", json.dumps(search_database.args, indent=2))
    result = search_database.invoke({"query": "john doe", "limit": 3})
    print("result:", result)

    result2 = search_database.invoke({"query": "alice"})
    print("result with default limit:", result2)


# SECTION 2 — Custom name and description
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return f"Web results for: {query}"


@tool("calculator", description="Performs arithmetic calculations. Use this for ANY math problem.")
def calc(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def section_2_custom_name_description():
    print("Tool name model sees:", search.name)
    print("invoke:", search.invoke({"query": "what is langchain"}))
    print("description:", calc.description)
    print("invoke 22*4+10 =", calc.invoke({"expression": "22 * 4 + 10"}))
    print("invoke 100/7  =", calc.invoke({"expression": "round(100/7, 2)"}))


# SECTION 3 — Pydantic schema for strict input validation
class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Whether to include a 5-day forecast"
    )


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional 5-day forecast for a city."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units}"
    if include_forecast:
        result += " | Forecast: Sunny all week"
    return result


def section_3_pydantic_schema():
    print(json.dumps(get_weather.args, indent=2))
    r1 = get_weather.invoke({"location": "Paris"})
    print("Only location:", r1)

    r2 = get_weather.invoke({
        "location": "New York",
        "units": "fahrenheit",
        "include_forecast": True
    })
    print("All fields:", r2)
    try:
        get_weather.invoke({"location": "Tokyo", "units": "kelvin"})
    except Exception as e:
        print(f"Caught error: {type(e).__name__}")
        print("Pydantic protected you — 'kelvin' is not allowed")


# SECTION 4 — What the model actually sees
def section_4_what_model_sees():
    print(json.dumps(convert_to_openai_tool(search_database), indent=2))
    print(json.dumps(convert_to_openai_tool(get_weather), indent=2))


# SECTION 5 — Tool return types
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price as a human-readable sentence."""
    prices = {"AAPL": 182.5, "GOOGL": 140.2, "MSFT": 378.9}
    price = prices.get(ticker.upper(), "unknown")
    return f"{ticker.upper()} is currently trading at ${price} per share."


@tool
def get_stock_data(ticker: str) -> dict:
    """Get structured stock data for analysis."""
    return {
        "ticker": ticker.upper(),
        "price": 182.5,
        "change_percent": 1.2,
        "volume": 54000000,
        "market_cap": "2.8T"
    }


@tool
def get_chart(ticker: str) -> list:
    """Get a stock chart image with description."""
    return [
        {"type": "text", "text": f"Here is the 30-day chart for {ticker.upper()}:"},
        {"type": "image", "url": "https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg"},
    ]


@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the exact shipping status of a customer order."""
    statuses = {
        "12345": "Shipped — arriving tomorrow by 8pm",
        "99999": "Processing — not yet dispatched",
    }
    return statuses.get(order_id, f"Order {order_id} not found.")


def section_5_return_types():
    r = get_stock_price.invoke({"ticker": "AAPL"})
    print("result:", r)
    print("use when: result is naturally readable text")

    r = get_stock_data.invoke({"ticker": "AAPL"})
    print("result:", r)
    print("use when: model needs to reason over specific fields")

    r = get_chart.invoke({"ticker": "AAPL"})
    print("result:", r)
    print("use when: tool result should include an image")

    print("return_direct flag:", fetch_order_status.return_direct)
    r = fetch_order_status.invoke({"order_id": "12345"})
    print("result:", r)


# SECTION 6 — Inspect tool properties
def section_6_inspect_tool_properties():
    all_tools = [search_database, search, calc, get_weather,
                 get_stock_price, get_stock_data, fetch_order_status]

    print("\n--- All tools at a glance ---")
    print(f"{'NAME':<25} {'RETURN_DIRECT':<15} {'ARGS'}")
    print("-" * 65)
    for t in all_tools:
        args = list(t.args.keys())
        print(f"{t.name:<25} {str(t.return_direct):<15} {args}")

    print("\n--- Dive into one tool ---")
    t = get_weather
    print("name         :", t.name)
    print("description  :", t.description)
    print("args         :", json.dumps(t.args, indent=2))
    print("return_direct:", t.return_direct)


# SECTION 7 — Bind tools to a model, watch it pick one
def section_7_bind_tools_to_model():
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o-mini")
    model_with_tools = model.bind_tools([search_database, calc, get_weather])

    print("\n--- Ask something that needs calc ---")
    response = model_with_tools.invoke("What is 1234 * 5678?")
    print("text      :", response.text)
    print("tool_calls:", response.tool_calls)

    print("\n--- Ask something that needs weather ---")
    response = model_with_tools.invoke("What's the weather in London?")
    print("text      :", response.text)
    print("tool_calls:", response.tool_calls)

    print("\n--- Ask something needing NO tool ---")
    response = model_with_tools.invoke("What is the capital of France?")
    print("text      :", response.text)
    print("tool_calls:", response.tool_calls)

# Concept: model requests -> you run -> send result back -> model answers
def section_8_manual_tool_call_loop():
    from langchain.chat_models import init_chat_model
    model = init_chat_model("gpt-4o-mini")
    model_with_tools = model.bind_tools([calc, get_weather])

    question = "What is 42 * 18?"
    messages = [HumanMessage(question)]
    print(f"Question: {question}")

    print("\n--- Step 1: Model decides what tool to call ---")
    ai_message = model_with_tools.invoke(messages)
    print("model text    :", ai_message.text)
    print("model wants   :", ai_message.tool_calls)

    if not ai_message.tool_calls:
        print("Model answered directly:", ai_message.text)
        return

    print("\n--- Step 2: Your code runs the tool ---")
    tool_call = ai_message.tool_calls[0]
    print("tool name :", tool_call["name"])
    print("tool args :", tool_call["args"])
    print("tool id   :", tool_call["id"])

    tool_map = {"calculator": calc, "get_weather": get_weather}
    chosen_tool = tool_map[tool_call["name"]]
    result = chosen_tool.invoke(tool_call["args"])
    print("tool result:", result)

    print("\n--- Step 3: Wrap result in ToolMessage ---")
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
    )
    print("content:", tool_message.content)
    print("id     :", tool_message.tool_call_id)

    print("\n--- Step 4: Model writes final answer ---")
    messages = [HumanMessage(question), ai_message, tool_message]
    final = model_with_tools.invoke(messages)
    print("FINAL ANSWER:", final.text)


# SECTION 9 — return_direct with a real model
def section_9_return_direct_in_action():
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_agent
    model = init_chat_model("gpt-4o-mini")

    @tool
    def get_price_normal(item: str) -> str:
        """Get the price of an item."""
        return f"{item}: $29.99"

    @tool(return_direct=True)
    def get_price_direct(item: str) -> str:
        """Get the exact listed price of an item."""
        return f"{item}: $29.99"

    agent_normal = create_agent(model, tools=[get_price_normal])
    agent_direct = create_agent(model, tools=[get_price_direct])

    q = {"messages": [{"role": "user", "content": "What is the price of a coffee mug?"}]}

    print("\n--- Normal (model rewrites the output) ---")
    result = agent_normal.invoke(q)
    print("Answer:", result["messages"][-1].content)

    print("\n--- return_direct (raw output returned immediately) ---")
    result = agent_direct.invoke(q)
    print("Answer:", result["messages"][-1].content)


# SECTION 10 — Multiple tools, model picks the right one
def section_10_multiple_tools_model_picks():
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_agent

    @tool
    def get_weather_info(city: str) -> str:
        """Get current weather conditions for a city. Use for weather questions."""
        return f"It is sunny and 22 degrees in {city}."

    @tool
    def do_math(expression: str) -> str:
        """Perform mathematical calculations. Use for any arithmetic or math problems."""
        try:
            return f"Result: {eval(expression)}"
        except:
            return "Invalid expression"

    @tool
    def search_web(query: str) -> str:
        """Search the web for general information and facts."""
        return f"Top result for '{query}': Simulated search result."

    @tool
    def get_time(timezone: str = "UTC") -> str:
        """Get the current time in a given timezone. Use when asked about time."""
        return f"Current time in {timezone}: 14:32:05"

    model = init_chat_model("gpt-4o-mini")
    agent = create_agent(model, tools=[get_weather_info, do_math, search_web, get_time])

    questions = [
        "What is 99 * 77?",
        "What's the weather in Tokyo?",
        "What time is it in India?",
        "Who invented the telephone?",
    ]

    print("\n--- Watch the model pick the right tool for each question ---\n")
    for q in questions:
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        answer = result["messages"][-1].content
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    # section_1_basic_tool()
    # section_2_custom_name_description()
    # section_3_pydantic_schema()
    # section_4_what_model_sees()
    # section_5_return_types()
    # section_6_inspect_tool_properties()
    # section_7_bind_tools_to_model()
    section_8_manual_tool_call_loop()
    section_9_return_direct_in_action()
    section_10_multiple_tools_model_picks()