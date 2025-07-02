import logging
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# -----------------------------
# Load a lightweight, tiny T5 model
# -----------------------------
model_name = "google/flan-t5-xs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# Enable logging
# -----------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
# /start handler
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! I'm your AI Question Generator Bot (Tiny‑T5 edition).\n\n"
        "📄 Send a passage (max 400 words) via:\n"
        "/ask [number_of_questions]\\n[your passage]",
        parse_mode="Markdown"
    )

# -----------------------------
# /ask handler
# -----------------------------
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            return await update.message.reply_text("⚠️ Please specify the number of questions first.")

        num_q = int(context.args[0])
        if num_q > MAX_QUESTIONS:
            return await update.message.reply_text(f"⚠️ Max is {MAX_QUESTIONS} questions.")

        passage = update.message.text.partition("\n")[2].strip()
        if not passage:
            return await update.message.reply_text("⚠️ Please send a passage after the command.")

        words = passage.split()
        if len(words) > MAX_WORDS:
            return await update.message.reply_text(f"❌ Limit reached: {len(words)} words (max {MAX_WORDS}).")

        chunk_size = max(len(words)//num_q, MIN_CHUNK_WORDS)
        chunks = [" ".join(words[i*chunk_size:(i+1)*chunk_size]) for i in range(num_q)]
        if len(words) % num_q:
            leftovers = words[num_q*chunk_size :]
            chunks[-1] += " " + " ".join(leftovers)

        questions = []
        for chunk in chunks:
            prompt = (
                "Generate 1 concise, factual comprehension question based on this chunk:\n\n"
                f"{chunk}"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            out = model.generate(**inputs, max_new_tokens=60)
            q = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            questions.append(q)

        resp = "\n\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        await update.message.reply_text(f"✅ Here are your question(s):\n\n{resp}")

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Something went wrong.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN env var.")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))
    print("🤖 Bot running (Tiny‑T5)…")
    app.run_polling()
