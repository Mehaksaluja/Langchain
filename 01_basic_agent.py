import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import uuid
import warnings
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"

load_dotenv()

@tool(description="Calculates final price after applying discount percentage.")
def calculate_discount(price: float, discount_percentage: float) -> float:
    final_price = price - (price * (discount_percentage / 100))
    return f"Final price after {discount_percentage}% discount is ${final_price:.2f}"

class SummaryResponse(BaseModel):
    summary: str = Field(..., description="A short summary of the user query request."),
    confidence_score: float = Field(..., description="A confidence score between 0 and 1 indicating the reliability of the summary.")

memory = InMemorySaver()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[calculate_discount],
    system_prompt="You are a helpful retail assistant. Be polite and accurate.",
    response_format=SummaryResponse,
    checkpointer=memory,
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("Running Turn 1...")

res1 = agent.invoke({"messages": [{"role": "user", "content": "I want to buy a $200 jacket with 20% discount."}]}, config=config)

print("\n--- Response 1 ---")
print(res1["structured_response"])

print("\nRunning Turn 2 (Testing Memory)...")

res2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Can you summarize what item I was buying?"}]},
    config=config
)

print("\n--- Response 2 (Memory Test) ---")
print(res2["structured_response"])