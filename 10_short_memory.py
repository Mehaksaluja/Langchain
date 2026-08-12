from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import AgentState
from langchain.messages import HumanMessage, AIMessage, RemoveMessage


# =============================================================================
# SECTION 1 — The core problem: why memory is needed
# =============================================================================

def section_1_the_problem():
    print("\n" + "="*60)
    print("SECTION 1 — Why memory is needed")
    print("="*60)

    print("""
WITHOUT memory — every call is isolated:

  Call 1:  User: "Hi, my name is Alice"
           Model: "Hello Alice!"

  Call 2:  User: "What's my name?"
           Model: "I don't know your name."  ← forgets everything

The model has no idea what happened in Call 1.
Each invoke() is a fresh start with zero context.
""")

    print("""
WITH short-term memory (checkpointer + thread_id):

  Call 1:  User: "Hi, my name is Alice"
           Model: "Hello Alice!"
           → full conversation saved to checkpointer

  Call 2:  User: "What's my name?"
           → checkpointer loads the saved history for this thread
           Model: "Your name is Alice!"  ← remembers!
""")

    print("KEY INSIGHT:")
    print("  No checkpointer   -> model forgets after every invoke()")
    print("  With checkpointer -> history is saved and loaded for each thread")
    print("  thread_id         -> which conversation's history to load")


# =============================================================================
# SECTION 2 — Key concepts: checkpointer, thread_id, InMemorySaver
# =============================================================================

def section_2_key_concepts():
    print("\n" + "="*60)
    print("SECTION 2 — Key concepts (offline)")
    print("="*60)

    print("\n--- checkpointer ---")
    print("A checkpointer is a storage system for conversation history.")
    print("It saves state after every step, loads it before the next one.")
    print()
    saver = InMemorySaver()
    print("InMemorySaver:", type(saver).__name__)
    print("Stores everything in RAM.")
    print("Lost when your program restarts.")
    print("Use for: development, testing, demos")
    print("NOT for: production (use PostgresSaver, RedisSaver, etc.)")

    print("\n--- thread_id ---")
    print("A thread is one conversation. thread_id is its unique label.")
    print("Different thread_id = completely separate memory.")
    print()
    alice_thread = {"configurable": {"thread_id": "alice_session_1"}}
    bob_thread   = {"configurable": {"thread_id": "bob_session_1"}}
    print("Alice's thread:", alice_thread)
    print("Bob's thread  :", bob_thread)
    print()
    print("agent.invoke(input, alice_thread) -> loads Alice's history only")
    print("agent.invoke(input, bob_thread)   -> loads Bob's history only")
    print("They never mix, even in the same agent.")

    print("\n--- How it all connects ---")
    print("""
  create_agent(checkpointer=InMemorySaver())
       ↓
  agent.invoke(input, {"configurable": {"thread_id": "1"}})
       ↓
  checkpointer saves state for thread "1"
       ↓
  next invoke() with same thread_id loads that saved state
       ↓
  model sees full conversation history automatically
""")
    print("KEY INSIGHT:")
    print("  thread_id    = which conversation (who/what session)")
    print("  checkpointer = where history is stored (RAM, Postgres, Redis...)")
    print("  You always need BOTH for memory to work")


# =============================================================================
# SECTION 3 — AgentState: what gets saved
# =============================================================================

def section_3_agent_state():
    print("\n" + "="*60)
    print("SECTION 3 — AgentState: what gets saved (offline)")
    print("="*60)

    print("\nAgentState is the object that gets checkpointed.")
    print("It holds everything about the current conversation.\n")

    print("Default AgentState has:")
    print("  messages           -> the full conversation history")
    print("  structured_response -> structured output if response_format is set")
    print()

    print("You can extend it with your own fields:\n")

    class CustomAgentState(AgentState):
        user_name: str
        visit_count: int
        preferences: dict

    print("class CustomAgentState(AgentState):")
    print("    user_name: str")
    print("    visit_count: int")
    print("    preferences: dict")
    print()
    print("Now the checkpointer also saves user_name, visit_count, preferences")
    print("alongside message history for each thread.")

    print("\n--- Why custom state matters ---")
    print("""
Scenario: a customer support agent

  After turn 1: user says "I'm a premium member"
  → tool sets state["membership"] = "premium"

  After turn 2: user asks for a discount
  → tool reads state["membership"] == "premium"
  → gives the right discount automatically

  Stored in the checkpointer — persists across all turns in the session.
""")

    print("--- What a message list looks like over time ---")
    msgs = [
        HumanMessage(content="Hi, my name is Alice", id="m1"),
        AIMessage(content="Hello Alice!", id="m2"),
        HumanMessage(content="What's my name?", id="m3"),
        AIMessage(content="Your name is Alice!", id="m4"),
    ]
    for m in msgs:
        role = "Human" if isinstance(m, HumanMessage) else "AI"
        print(f"  [{role}] {m.content}")
    print()
    print("This full list is saved by the checkpointer.")
    print("On next invoke(), model receives all 4 messages + new input.")

    print("\nKEY INSIGHT:")
    print("  AgentState.messages  -> the growing conversation history")
    print("  Custom fields        -> anything else your app needs to remember")
    print("  Checkpointer saves ALL of this between invoke() calls")


# =============================================================================
# SECTION 4 — Basic memory: agent that remembers your name
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_4_basic_memory():
    from langchain.agents import create_agent

    print("\n" + "="*60)
    print("SECTION 4 — Basic memory: agent that remembers (needs API key)")
    print("="*60)

    checkpointer = InMemorySaver()

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        checkpointer=checkpointer,  # add this → agent now has memory
    )

    config = {"configurable": {"thread_id": "session_1"}}

    print("\n--- Turn 1: tell agent your name ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hi! My name is Alice."}]},
        config,
    )
    print("Agent:", result["messages"][-1].content)

    print("\n--- Turn 2: agent should remember ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config,   # same thread_id → loads Alice's history
    )
    print("Agent:", result["messages"][-1].content)

    print("\n--- Turn 3: continue conversation ---")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What did I say in my first message?"}]},
        config,
    )
    print("Agent:", result["messages"][-1].content)

    print("\n--- Different thread = fresh start ---")
    config_new = {"configurable": {"thread_id": "session_2"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config_new,   # new thread_id → no memory of Alice
    )
    print("Agent (new thread):", result["messages"][-1].content)

    print("\nKEY INSIGHT:")
    print("  Same thread_id  -> agent remembers everything from this session")
    print("  New thread_id   -> completely fresh start, zero memory")
    print("  checkpointer saves after every invoke(), loads before the next")


# =============================================================================
# SECTION 5 — Trim messages (handle long conversations)
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_5_trim_messages():
    from langchain.agents import create_agent
    from langchain.agents.middleware import before_model
    from langgraph.runtime import Runtime
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    from typing import Any

    print("\n" + "="*60)
    print("SECTION 5 — Trim messages (needs API key)")
    print("="*60)

    print("""
The problem with long conversations:
  Turn 1:   [H, A]              = 2 messages
  Turn 10:  [H,A,...H,A]        = 20 messages
  Turn 100: ....                = 200 messages → context window exceeded!

Solution: before calling the model, trim old messages.
Keep first message + most recent few.
""")

    @before_model
    def trim_old_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]

        if len(messages) <= 4:
            return None   # short enough, nothing to do

        # Keep: first message + last 4 messages
        first  = messages[0]
        recent = messages[-4:]

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),  # clear history
                first,                                   # restore first
                *recent                                  # restore recent
            ]
        }

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[trim_old_messages],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "trim_test"}}

    turns = [
        "Hi! My name is Bob.",
        "I love Python programming.",
        "My favorite food is pizza.",
        "Write me a haiku about coding.",
        "Write me another haiku about food.",
        "What's my name?",   # tests if first message survived trimming
    ]

    for msg in turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config,
        )
        print(f"User : {msg}")
        print(f"Agent: {result['messages'][-1].content[:100]}")
        print()

    print("KEY INSIGHT:")
    print("  @before_model                    -> runs BEFORE every model call")
    print("  RemoveMessage(REMOVE_ALL_MESSAGES) -> wipes entire history")
    print("  Then we add back first + recent   -> controlled trimming")
    print("  Model never sees the full history — keeps costs low")


# =============================================================================
# SECTION 6 — Delete specific messages
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_6_delete_messages():
    from langchain.agents import create_agent
    from langchain.agents.middleware import after_model
    from langgraph.runtime import Runtime

    print("\n" + "="*60)
    print("SECTION 6 — Delete specific messages (needs API key)")
    print("="*60)

    print("""
RemoveMessage(id=message_id) deletes ONE specific message by its id.
Useful for:
  - Removing model responses with sensitive content
  - Content moderation
  - Building a "forget this" feature
""")

    @after_model
    def remove_sensitive_responses(state: AgentState, runtime: Runtime) -> dict | None:
        FORBIDDEN = ["password", "secret", "confidential"]
        last = state["messages"][-1]
        if any(word in last.content.lower() for word in FORBIDDEN):
            print("  [Deleted response containing forbidden word]")
            return {"messages": [RemoveMessage(id=last.id)]}
        return None

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[remove_sensitive_responses],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "delete_test"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
        config,
    )
    print("Normal response:", result["messages"][-1].content[:60])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Tell me a secret about cats"}]},
        config,
    )
    last = result["messages"][-1]
    print("After filter:", last.content[:80] if last.content else "[response deleted]")

    print("\nKEY INSIGHT:")
    print("  @after_model     -> runs AFTER every model call")
    print("  RemoveMessage(id)-> deletes that specific message from state")
    print("  Use for: content moderation, privacy, removing wrong answers")


# =============================================================================
# SECTION 7 — Summarization: smart compression of long conversations
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_7_summarization():
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware

    print("\n" + "="*60)
    print("SECTION 7 — Summarization middleware (needs API key)")
    print("="*60)

    print("""
Problem with trimming: you LOSE information.
  "My name is Bob" gets trimmed → model forgets Bob.

Summarization: compress old messages into a short summary.
  Before: 20 messages about Bob's life
  After:  "User is Bob, a Python dev who loves hiking and pizza."
  Keep:   summary + last few messages

Information PRESERVED. Context window stays small.
""")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",       # model used to write the summary
                trigger=("tokens", 500),   # summarize when > 500 tokens
                keep=("messages", 4),      # keep last 4 messages after summary
            )
        ],
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "summary_test"}}

    turns = [
        "Hi! My name is Bob.",
        "I'm a software engineer who loves Python.",
        "My favorite hobby is hiking in the mountains.",
        "Write me a short poem about mountains.",
        "Now write one about coding.",
        "What do you remember about me?"
    ]

    for msg in turns:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config,
        )
        print(f"User : {msg}")
        print(f"Agent: {result['messages'][-1].content[:120]}")
        print()

    print("KEY INSIGHT:")
    print("  trigger=('tokens', 500) -> summarize when history > 500 tokens")
    print("  keep=('messages', 4)    -> keep last 4 messages after summarizing")
    print("  Old info COMPRESSED not deleted — model still remembers")
    print()
    print("  Trim      -> fast, cheap, LOSES old info")
    print("  Delete    -> targeted removal of specific messages")
    print("  Summarize -> preserves key info, costs one extra model call")


if __name__ == "__main__":
    section_1_the_problem()
    section_2_key_concepts()
    section_3_agent_state()
    # section_4_basic_memory()
    # section_5_trim_messages()
    # section_6_delete_messages()
    # section_7_summarization()