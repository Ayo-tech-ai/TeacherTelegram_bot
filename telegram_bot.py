import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# -----------------------------
# Load the model and tokenizer only once
# -----------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# Enable logging
# -----------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
MAX_WORDS = 400
MAX_QUESTIONS = 5
MIN_CHUNK_WORDS = 50

# -----------------------------
# Start Command Handler
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I'm your AI Question Generator Bot.\n\n"
        "📄 Send me a short passage (max 400 words), and I’ll generate up to 5 comprehension questions based on it.\n\n"
        "Use the format: \n\n"
        "/ask [number_of_questions]\n[Your passage here]"
    )

# -----------------------------
# Ask Command Handler
# -----------------------------
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text("❗Please specify the number of questions (e.g., /ask 3 followed by your passage).")
            return

        num_questions = int(context.args[0])
        if num_questions > MAX_QUESTIONS:
            await update.message.reply_text(f"⚠️ Max allowed questions is {MAX_QUESTIONS}.")
            return

        passage = update.message.text.partition('\n')[2].strip()
        if not passage:
            await update.message.reply_text("❗Please include the passage below the command.")
            return

        words = passage.split()
        word_count = len(words)

        if word_count > MAX_WORDS:
            await update.message.reply_text(f"❌ Your passage has {word_count} words. Please limit to {MAX_WORDS} words.")
            return

        # Calculate chunk size by percentage
        chunk_size = max(len(words) // num_questions, MIN_CHUNK_WORDS)
        chunks = [" ".join(words[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_questions)]

        # Handle leftover words
        leftover_index = num_questions * chunk_size
        if leftover_index < len(words):
            chunks[-1] += " " + " ".join(words[leftover_index:])

        questions = []
        for chunk in chunks:
            prompt = (
                f"Based on the following passage chunk, generate 1 clear, fact-based comprehension question. "
                f"Avoid repetition and do not invent facts:\n\n{chunk}"
            )
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                num_return_sequences=1
            )
            question = tokenizer.decode(output[0], skip_special_tokens=True)
            questions.append(question.strip())

        response = "\n\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
        await update.message.reply_text(f"✅ Here are your questions:\n\n{response}")

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ An error occurred while generating the questions.")

# -----------------------------
# Main Entrypoint
# -----------------------------
if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_TOKEN:
        raise ValueError("Please set the TELEGRAM_BOT_TOKEN environment variable.")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))

    print("🤖 Bot is running...")
    app.run_polling()
