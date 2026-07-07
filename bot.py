import os
import logging
from collections import defaultdict
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
MODEL_NAME = 'gemini-1.5-flash'
MAX_HISTORY = 10

SYSTEM_PROMPT = 'You are Gurusigmabot, a helpful personal AI assistant. You help with tasks, answer questions, write content, and assist with making money online. Be friendly and conversational.'

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT)
conversation_history = defaultdict(list)

def _trim_history(user_id):
    history = conversation_history[user_id]
    if len(history) > MAX_HISTORY:
        trim_to = MAX_HISTORY if MAX_HISTORY % 2 == 0 else MAX_HISTORY - 1
        conversation_history[user_id] = history[-trim_to:]

async def _ask_gemini(user_id, user_text):
    history = conversation_history[user_id]
    chat = model.start_chat(history=list(history))
    response = await chat.send_message_async(user_text)
    reply = response.text.strip()
    conversation_history[user_id].append({'role': 'user', 'parts': [user_text]})
    conversation_history[user_id].append({'role': 'model', 'parts': [reply]})
    _trim_history(user_id)
    return reply

async def start(update, context):
    user = update.effective_user
    name = user.first_name if user else 'there'
    await update.message.reply_text(f'Namaste {name}! I am Gurusigmabot, your personal AI assistant. Just send me a message!')

async def help_command(update, context):
    await update.message.reply_text('/start - Welcome\n/help - This message\n/clear - Clear history\n\nJust type anything!')

async def clear_command(update, context):
    conversation_history[update.effective_user.id].clear()
    await update.message.reply_text('History clear ho gayi!')

async def handle_message(update, context):
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    try:
        reply = await _ask_gemini(user_id, user_text)
        await update.message.reply_text(reply)
    except Exception as exc:
        logger.error('Gemini error: %s', exc)
        await update.message.reply_text(f'Error: {exc}')

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('clear', clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info('Gurusigmabot starting...')
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()