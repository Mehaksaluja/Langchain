import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import TodoListMiddleware
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

@tool(description="Search for a query and return a short summary.")
def web_search(query: str) -> str:
    return f"Mock search results found for topic: {query}"

backend = StateBackend()

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    system_prompt="You are a Lead Tech Manager. Break down complex user tasks into a todo list, and delegate research work to your researcher subagent.",
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Expert in deep web research and structuring summaries.",
                    "system_prompt": "Search thoroughly using the web search tool and return clean bullet points.",
                    "tools": [web_search],
                    "model": "openai:gpt-4o-mini",
                    "middleware": [],
                }
            ],
        ),
    ],
)

print("Initializing Deep Agent execution...")
print("-" * 50)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Research the latest trends in Agentic AI workflows and create a summary."}]
})

print("\n--- Final Deep Agent Output ---")
print(response)