import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from Predicter import Predicter
import os
from dotenv import load_dotenv

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    encoding='utf-8',
    filemode='a',
    filename="log.txt"
)


def predict_text(text):
    prediction, max_predict_value, margin, reply, tag = predictor.predict_text(text)
    logging.info('New message:')
    logging.info(text)
    logging.info(prediction)
    logging.info(f'max pred - {max_predict_value}' )
    logging.info(f'margin - {margin}')
    logging.info(f'tag - {tag}')
    return reply
        

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = """Я простецкий чат бот, дублирующий информацию из интернета про котов.
Задайте вопрос на одну из тем: уход, питание, здоровье, поведение, дрессировка, игрушки, пространство для кошки, законодательство.
А я попробую ответить)"""
    user = update.message.from_user
    logging.info( f"User {user.first_name} (@{user.username}) with ID {user.id} started bot.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = predict_text(update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)




if __name__ == '__main__':
    predictor = Predicter()
    predictor.load_model()
    # predictor.evaluate()
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    application = ApplicationBuilder().token(token).read_timeout(40).write_timeout(40).build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
 
    application.run_polling()