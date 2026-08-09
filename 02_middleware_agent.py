import os
import warnings
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware, 
    ToolRetryMiddleware, 
    PIIMiddleware,
    HumanInTheLoopMiddleware,
)

load_dotenv()
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"

@tool(description="Deletes a user account from the system database using their email.")
def delete_user_account(user_email: str) -> str:
    return f"SUCCESS: Account for {user_email} has been permanently deleted."

agent = create_agent(
    model="gpt-4o-mini",
    tools=[delete_user_account],
    system_prompt="You are an admin assistant. You can manage user accounts. If asked to delete an account, use your tool.",
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
        HumanInTheLoopMiddleware(interrupt_on={"delete_user_account": True}),
        PIIMiddleware("email"),
    ],
)

print("User: Please delete the account for test.user@gmail.com")
print("-" * 50)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Please delete the account for test.user@gmail.com"}]
})

print("\n--- Final Output ---")
print(response)