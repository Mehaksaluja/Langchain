from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model, after_model, SummarizationMiddleware
from langgraph.runtime import Runtime
from typing import Any


# =============================================================================
# SECTION 1 — checkpointer + thread_id
# checkpointer = where history is stored | thread_id = which conversation to load
# Same thread_id across calls = agent remembers. New thread_id = fresh start.
# =============================================================================

def section_1_checkpointer_and_thread():
    # InMemorySaver stores history in RAM — fine for dev, lost on restart
    checkpointer = InMemorySaver()

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "session_1"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Alice."}]},
        config,
    )
    print(result["messages"][-1].content)

    # same thread_id → agent loads Alice's history → remembers
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config,
    )
    print(result["messages"][-1].content)

    # different thread_id = zero memory of Alice
    new_config = {"configurable": {"thread_id": "session_2"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        new_config,
    )
    print(result["messages"][-1].content)


# =============================================================================
# SECTION 2 — Custom AgentState
# Extend AgentState to store extra fields alongside message history.
# Checkpointer saves these fields too, so they persist across turns.
# =============================================================================

def section_2_custom_state():
    class CustomState(AgentState):
        user_name: str
        visit_count: int

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        state_schema=CustomState,
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "custom_1"}}

    # pass custom fields alongside messages — saved in checkpointer
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Hello!"}],
            "user_name": "Bob",
            "visit_count": 1,
        },
        config,
    )
    print(result["messages"][-1].content)
    print("user_name:", result.get("user_name"))
    print("visit_count:", result.get("visit_count"))


# =============================================================================
# SECTION 3 — Trim messages
# Long conversations overflow the context window.
# @before_model runs before every model call — trim old messages here.
# =============================================================================

def section_3_trim_messages():
    @before_model
    def trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]

        if len(messages) <= 4:
            return None  # short enough, nothing to do

        # keep first message + last 4, discard everything in between
        kept = [messages[0]] + messages[-4:]
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),  # wipe current history
                *kept,                                   # restore what we want
            ]
        }

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[trim],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "trim_1"}}
    turns = [
        "My name is Bob.",
        "I love Python.",
        "My hobby is hiking.",
        "Write a haiku about mountains.",
        "Write a haiku about coding.",
        "What is my name?",  # tests whether first message survived trimming
    ]

    for msg in turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config,
        )
        print(f"User : {msg}")
        print(f"Agent: {result['messages'][-1].content[:100]}\n")


# =============================================================================
# SECTION 4 — Delete specific messages
# RemoveMessage(id=...) deletes one message by its ID.
# @after_model runs after every model call — use it to filter responses.
# =============================================================================

def section_4_delete_messages():
    FORBIDDEN = ["password", "secret", "confidential"]

    @after_model
    def remove_sensitive(state: AgentState, runtime: Runtime) -> dict | None:
        last = state["messages"][-1]
        # if response contains a forbidden word, delete it from history
        if any(word in last.content.lower() for word in FORBIDDEN):
            return {"messages": [RemoveMessage(id=last.id)]}
        return None

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[remove_sensitive],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "delete_1"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2 + 2?"}]},
        config,
    )
    print("Normal:", result["messages"][-1].content[:80])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me a secret about cats."}]},
        config,
    )
    last = result["messages"][-1]
    print("Filtered:", last.content[:80] if last.content else "[deleted]")


# =============================================================================
# SECTION 5 — Summarization
# Trim loses information. Summarization compresses old messages into a summary.
# SummarizationMiddleware is prebuilt — no code needed, just configure it.
# =============================================================================

def section_5_summarization():
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",      # model used to write the summary
                trigger=("tokens", 500),  # summarize when history > 500 tokens
                keep=("messages", 4),     # keep last 4 messages after summarizing
            )
        ],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "summary_1"}}
    turns = [
        "My name is Bob.",
        "I am a Python developer.",
        "My hobby is hiking.",
        "Write a haiku about mountains.",
        "Write a haiku about coding.",
        "What do you remember about me?",  # tests whether summary preserved key facts
    ]

    for msg in turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config,
        )
        print(f"User : {msg}")
        print(f"Agent: {result['messages'][-1].content[:120]}\n")


# =============================================================================
# MAIN — uncomment sections one at a time
# =============================================================================

if __name__ == "__main__":
    section_1_checkpointer_and_thread()
    # section_2_custom_state()
    # section_3_trim_messages()
    # section_4_delete_messages()
    # section_5_summarization()