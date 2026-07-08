import os
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone

from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import google.generativeai as genai
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PORT = int(os.environ.get("PORT", 10000))
DB_PATH = os.environ.get("DB_PATH", "gurusigma.db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite Database
# ---------------------------------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            task TEXT NOT NULL,
            time_str TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"**{r.get('title', '')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"

def remember_fact(user_id: str, fact: str) -> str:
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO memories (user_id, fact, created_at) VALUES (?, ?, ?)",
            (str(user_id), fact, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return f"Remembered: {fact}"
    except Exception as e:
        return f"Failed to save: {str(e)}"

def get_memory(user_id: str) -> str:
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT fact, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
            (str(user_id),),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No memories saved yet."
        facts = [f"• {row[0]} (saved {row[1][:10]})" for row in rows]
        return "\n".join(facts)
    except Exception as e:
        return f"Failed to retrieve memories: {str(e)}"

def set_reminder(user_id: str, task: str, time_str: str) -> str:
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO reminders (user_id, task, time_str, created_at) VALUES (?, ?, ?, ?)",
            (str(user_id), task, time_str, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return f"Reminder set: '{task}' at {time_str}"
    except Exception as e:
        return f"Failed to set reminder: {str(e)}"

def get_reminders(user_id: str) -> str:
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT task, time_str, created_at FROM reminders WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
            (str(user_id),),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No reminders set."
        items = [f"• {row[0]} — {row[1]} (created {row[2][:10]})" for row in rows]
        return "\n".join(items)
    except Exception as e:
        return f"Failed to retrieve reminders: {str(e)}"

# ---------------------------------------------------------------------------
# Gemini Setup with Function Calling
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a powerful personal AI assistant named Gurusigmabot. "
    "You can search the web, remember things about the user, set reminders, "
    "help with tasks, answer questions, write content, analyze ideas, and help make money online. "
    "Be smart, friendly, and proactive. Always use tools when needed. "
    "When the user tells you something personal, use remember_fact. "
    "When the user asks for current info, use web_search. "
    "When the user asks to be reminded, use set_reminder."
)

TOOL_DECLARATIONS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="web_search",
                description="Search the web for current information.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={"query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Search query")},
                    required=["query"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="remember_fact",
                description="Save an important fact about the user to persistent memory.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "user_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="User ID"),
                        "fact": genai.protos.Schema(type=genai.protos.Type.STRING, description="Fact to remember"),
                    },
                    required=["user_id", "fact"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="get_memory",
                description="Retrieve saved facts about the user.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={"user_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="User ID")},
                    required=["user_id"],
                ),
            ),
            genai.protos.FunctionDeclaration(
                name="set_reminder",
                description="Set a reminder for the user.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "user_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="User ID"),
                        "task": genai.protos.Schema(type=genai.protos.Type.STRING, description="Reminder task"),
                        "time_str": genai.protos.Schema(type=genai.protos.Type.STRING, description="When to remind"),
                    },
                    required=["user_id", "task", "time_str"],
                ),
            ),
        ]
    )
]

TOOL_FUNCTIONS = {
    "web_search": lambda args: web_search(args["query"]),
    "remember_fact": lambda args: remember_fact(args["user_id"], args["fact"]),
    "get_memory": lambda args: get_memory(args["user_id"]),
    "set_reminder": lambda args: set_reminder(args["user_id"], args["task"], args["time_str"]),
}

user_chats: dict = {}

def get_or_create_chat(user_id: str):
    if user_id not in user_chats:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name= gemini-2.0-flash
            tools=TOOL_DECLARATIONS,
            system_instruction=SYSTEM_PROMPT,
        )
        user_chats[user_id] = model.start_chat(enable_automatic_function_calling=False)
    return user_chats[user_id]

async def process_with_gemini(user_id: str, message: str) -> str:
    chat = get_or_create_chat(user_id)
    augmented_message = f"[User ID: {user_id}] {message}"
    try:
        response = chat.send_message(augmented_message)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        if user_id in user_chats:
            del user_chats[user_id]
        return f"Sorry, I encountered an error: {str(e)}"

    for _ in range(5):
        if not response.candidates:
            return "Could not generate a response. Please try again."
        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if p.function_call.name]
        if not function_calls:
            text_parts = [p.text for p in parts if p.text]
            return "\n".join(text_parts) if text_parts else "Done!"
        function_responses = []
        for part in function_calls:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)
            logger.info(f"Tool call: {fn_name}({fn_args})")
            result = TOOL_FUNCTIONS[fn_name](fn_args) if fn_name in TOOL_FUNCTIONS else f"Unknown: {fn_name}"
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn_name, response={"result": result}
                    )
                )
            )
        try:
            response = chat.send_message(function_responses)
        except Exception as e:
            return f"Error processing tool results: {str(e)}"
    return "Processing complete."

# ---------------------------------------------------------------------------
# Telegram Handlers
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Hey! I'm Gurusigmabot* — your personal AI assistant.\n\n"
        "I can:\n🔍 Search the web\n🧠 Remember things about you\n⏰ Set reminders\n"
        "✍️ Write content & analyze ideas\n💡 Help with anything\n\nJust talk to me! /help for commands.",
        parse_mode="Markdown",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Commands:*\n/start /help /clear /memory /remind\n\n"
        "*What I can do:*\n• Answer questions (searches web if needed)\n"
        "• Remember facts ('Remember I like Python')\n• Set reminders\n"
        "• Write emails, code, content\n• Help make money online\n\nJust message naturally!",
        parse_mode="Markdown",
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id in user_chats:
        del user_chats[user_id]
    await update.message.reply_text("🗑️ Conversation cleared! Saved memories are still intact.")

async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    memories = get_memory(user_id)
    await update.message.reply_text(f"🧠 *Your Memories:*\n\n{memories}", parse_mode="Markdown")

async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    reminders = get_reminders(user_id)
    await update.message.reply_text(f"⏰ *Your Reminders:*\n\n{reminders}", parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = str(update.effective_user.id)
    message = update.message.text
    await update.message.chat.send_action("typing")
    response = await process_with_gemini(user_id, message)
    if len(response) > 4000:
        for chunk in [response[i:i+4000] for i in range(0, len(response), 4000)]:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(response)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update error: {context.error}")

# ---------------------------------------------------------------------------
# Flask Keep-Alive
# ---------------------------------------------------------------------------

flask_app = Flask(__name__)

@flask_app.route("/")
def health():
    return json.dumps({"status": "alive", "bot": "Gurusigmabot"}), 200, {"Content-Type": "application/json"}

@flask_app.route("/health")
def health_check():
    return json.dumps({"status": "healthy"}), 200, {"Content-Type": "application/json"}

def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required")
    init_db()
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info(f"Flask keep-alive started on port {PORT}")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("memory", memory_command))
    application.add_handler(CommandHandler("remind", remind_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    logger.info("Gurusigmabot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
