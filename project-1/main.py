from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, ToolMessage
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
load_dotenv()

notes = []
messages = []

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Use for airthmatic"""
    try:
        return str(eval(expression))
    except:
        return "Invalid Expression"

@tool
def get_date():
    """Get today's date"""
    return datetime.date.today().strftime("%B %d, %Y")

@tool
def save_note(note: str) -> str:
    """Save the notes in the list."""
    notes.append(note)
    return "Note Saved!"

@tool
def show_notes():
    """Show all saved notes."""
    if not notes:
        return "No Notes saved yet!"
    return "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))


model = init_chat_model("gpt-4o-mini")
model_with_tools = model.bind_tools([calculator, get_date, save_note, show_notes])

def run_turn(user_input: str):
    messages.append(HumanMessage(user_input))
    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)

    if ai_message.tool_calls:
        tool_map = {
            "calculator": calculator,
            "get_date": get_date,
            "save_note": save_note,
            "show_notes": show_notes,
        }

        for tool_call in ai_message.tool_calls:
            tool_fn = tool_map[tool_call["name"]]
            result = tool_fn.invoke(tool_call["args"])

            tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )

            messages.append(tool_msg)
            final = model_with_tools.invoke(messages)
            messages.append(final)
            return final.text

    return ai_message.text

class ConversationSummary(BaseModel):
    total_messages: int = Field(description="Total number of messages in history")
    topics: list[str] = Field(description="List of topics discussed")
    notes_count: int = Field(description="Number of notes saved")

structured_model = model.with_structured_output(ConversationSummary)

def run_summarize():
    result = structured_model.invoke(
        messages + [HumanMessage("Summarize our conversation so far")]
    )
    return result.model_dump()


def main():
    print("Assistant ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Goodbye!")
            break

        if user_input.lower() == "summarize":
            print("Bot:", run_summarize())
            continue

        response = run_turn(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
