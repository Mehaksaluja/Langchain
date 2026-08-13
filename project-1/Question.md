Project: CLI Personal Assistant
What you're building

A command-line chatbot that runs in your terminal. The user types messages, the bot responds. It has real tools, remembers the conversation, and can return structured output — all built without create_agent. You manually write the tool-call loop.

Exact behavior it should have
You: Hi, my name is Arjun
Bot: Hello Arjun! How can I help you today?

You: what is 1847 * 293
Bot: 1847 * 293 = 541,171

You: what is today's date
Bot: Today is Thursday, August 13, 2026

You: save note: buy groceries tomorrow
Bot: Note saved ✓

You: save note: call dentist on Monday
Bot: Note saved ✓

You: show notes
Bot: Your notes:
     1. buy groceries tomorrow
     2. call dentist on Monday

You: what is my name
Bot: Your name is Arjun

You: summarize
Bot: {"total_messages": 10, "topics": ["greeting", "math", "notes"], "notes_count": 2}

You: quit
Bot: Goodbye!
Tools you must build (4 tools)

1. calculator

Input: a math expression as a string ("1847 * 293")
Output: the result as a string ("541171")
Use Python's eval() to compute it
Handle errors — if expression is invalid, return "Invalid expression"

2. get_date

Input: nothing
Output: today's date as a string
Use Python's datetime module

3. save_note

Input: the note text as a string
Output: "Note saved ✓"
Store notes in a plain Python list defined at the top of your file

4. show_notes

Input: nothing
Output: all saved notes as a numbered string
If no notes, return "No notes saved yet"
Structured output

When the user types "summarize", instead of a normal text response, use with_structured_output to return a Pydantic model with these exact fields:

python
class ConversationSummary(BaseModel):
    total_messages: int      # total number of messages in history so far
    topics: list[str]        # list of topics discussed (model decides)
    notes_count: int         # how many notes have been saved

Print the result as a dict using .model_dump().

Memory requirement
Keep a messages list at the top of your script
Every turn: append the HumanMessage, run the loop, append the AIMessage (and ToolMessage if a tool was called)
The model must receive the full history on every call — that's how it remembers your name
The tool-call loop you must write manually

This is the most important part. For every user input:

1. Append HumanMessage to history
2. Call model_with_tools.invoke(history)
3. Append the AIMessage to history
4. Check if ai_message.tool_calls is not empty
5. If yes:
   - Find which tool was called
   - Run it with the given args
   - Wrap result in ToolMessage (matching tool_call_id)
   - Append ToolMessage to history
   - Call model again with updated history to get final answer
   - Append that final AIMessage to history
6. Print the final text response

No create_agent. Write this loop yourself inside a while True: input loop.

File structure

Single file: assistant.py

assistant.py
  ├── imports
  ├── notes = []               ← shared list for save_note / show_notes
  ├── messages = []            ← conversation history
  ├── Tool definitions (4)
  ├── ConversationSummary Pydantic model
  ├── model setup
  ├── model_with_tools = model.bind_tools([...])
  ├── structured_model = model.with_structured_output(ConversationSummary)
  └── main loop (while True)
Rules
No create_agent — write the loop manually
No LangChain memory/checkpointer — manage the messages list yourself
The summarize command must use with_structured_output, not a plain text response
Handle the case where the model answers directly (no tool call) — your loop must work for both cases
quit or exit should end the program cleanly
What you'll use from what you learned
Concept	Where used
Messages	HumanMessage, AIMessage, ToolMessage — building history manually
Tools	@tool decorator on all 4 tools
Tool-call loop	The 4-step manual loop from Messages Section 8
Structured output	with_structured_output for the summarize command
Memory	Manual messages list passed to every model.invoke()
How to run
bash
export OPENAI_API_KEY="sk-..."
python assistant.py
What counts as done
 All 4 tools work correctly when the model calls them
 Conversation history is maintained — model remembers name across turns
 summarize returns a Pydantic object printed as a dict
 Tool errors are handled gracefully (bad math expression, empty notes)
 The loop works whether or not the model calls a tool
 quit exits cleanly