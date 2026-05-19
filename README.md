# 🌟 SJVN Suvidha — Your Personal AI Document Assistant

Welcome to **SJVN Suvidha**! We built this smart chatbot to make navigating complex HR policies, Delegation of Powers (DoP), and company guidelines a breeze. 

The best part? **It’s 100% free, entirely local, and completely private.** Because it runs right on your own hardware, you don't have to worry about API costs, internet connections, or your sensitive data leaking out.

---

## ✨ Why You'll Love It (Key Features)

* 🧠 **It Learns from You:** Spot a mistake? Give it a thumbs down and tell it the correct answer. The AI saves this to its memory (via SQLite) and will instantly get it right the next time you ask.
* 🎤 **Talk and Listen:** Don't feel like typing? Just use your microphone to ask questions, and let the Text-to-Speech (TTS) feature read the answers back to you.
* 🇮🇳 **Speaks Your Language:** Feel free to ask questions in English, Hindi (Devanagari), or Hinglish! The system automatically translates everything behind the scenes to find the perfect answer.
* 📊 **Smart Tables & 1-Click Exports:** When a document has structured data, the AI creates neat Markdown tables. You can even download them as a PDF or CSV file with a single click.
* 🔗 **Verifiable Answers:** No more guessing where an answer came from. Every response includes a direct citation—just click the link to jump to the exact page in the original PDF.
* 🛡️ **Built-in Admin Dashboard:** A password-protected portal lets admins manage the dynamic FAQ sidebar and keep an eye on audit logs (like user IPs and queries).
* ⚡ **Runs on Everyday Hardware:** We've optimized the routing and context windows so this beast runs smoothly even on standard 4GB laptop GPUs (like an RTX 3050).

---

## 🚀 Get Started in 5 Minutes

Ready to spin it up? Just follow these quick steps:

### Step 1: Get Ollama
First, download our local AI engine from **[Ollama's website](https://ollama.com/download)**. *(It's available for Windows, Mac, and Linux!)*

### Step 2: Grab the AI Models
Open up your terminal or command prompt. You'll need two models to make the magic happen: one to read the documents, and one to talk to you.

**1. Pull the Document Reader (Required):**
```bash
ollama pull nomic-embed-text
```

**2. Pull the Chat Model:**
Pick ONE of these depending on how much RAM your computer has:
* **16 GB+ RAM** | Qwen 2.5 (14B) | `ollama pull qwen2.5:14b` | ⭐⭐⭐⭐⭐ Best Quality
* **8 GB+ RAM** | Qwen 2.5 (7B)  | `ollama pull qwen2.5:7b`  | ⭐⭐⭐⭐ Great Quality
* **4 GB+ RAM** | Qwen 2.5 (3B)  | `ollama pull qwen2.5:3b`  | ⭐⭐⭐ Fast & Good

*(Pro-tip: If you're on a standard laptop, we highly recommend `qwen2.5:3b`—it's heavily optimized for this specific app!)*

### Step 3: Keep Ollama Running
Leave this command running in the background while you use the chatbot:
```bash
ollama serve
```

### Step 4: Install the Python Tools
Open a fresh terminal window and install the required libraries:
```bash
pip install flask pdfplumber cryptography numpy
```

### Step 5: Drop in Your PDFs
Place your official company PDFs into the `documents/` folder. It should look something like this:
```text
documents/
  dop.pdf
  5889_LEAVE_RULE_Policy.pdf
  Procurement_Policy.pdf
  (add any other PDFs here too!)
```

### Step 6: Start the Chatbot!
Run the app using Python:
```bash
python app.py
```
*(Note: The very first time you run this, it will take a minute or two to read and index all your PDFs. Let it do its thing!)*

### Step 7: Start Chatting
Open your favorite web browser and head over to:
**http://localhost:5000**

---

## 🔐 For the Admins (Dashboard & FAQs)

SJVN Suvidha comes with a secure backend so you can monitor usage and manage the FAQ sidebar.

* **How to access:** Click the **⚙ Admin** button in the sidebar.
* **Default Password:** `SJVN@Admin2024` *(You can change this in the environment variables).*
* **What you can do:** - Check real-time Audit Logs (timestamps, user IPs, and what people are asking).
  - Add or delete questions from the sidebar's FAQ database.

---

## 🎛️ Power User Tweaks (Environment Variables)

Want to customize how the server runs without touching the code? Set these environment variables before running `app.py`:

* `OLLAMA_HOST`: Hosting Ollama on a different machine? Point to it here! *(Default: http://localhost:11434)*
* `OLLAMA_MODEL`: Force the app to use a specific model. *(Default: Auto-detects)*
* `PORT`: Change the web port if 5000 is taken. *(Default: 5000)*
* `ADMIN_PASSWORD`: Update your dashboard login password. *(Default: SJVN@Admin2024)*

---

## 🛠️ Oops! Troubleshooting Guide

Hit a snag? Here are some quick fixes:

**"My microphone isn't working / Access is denied"**
Modern browsers block microphones on non-HTTPS websites. 
* If you are on the host PC: Make sure you are using `http://localhost:5000` (browsers always trust localhost).
* If you are testing from your phone: Use a tool like Ngrok (`ngrok http 5000`) to create a secure HTTPS tunnel.

**"The AI keeps repeating the same sentence over and over"**
Double-check that you are using a `qwen2.5` model! The `llm.py` file has a strict `repeat_penalty` designed specifically for Qwen to stop it from looping.

**"It keeps saying '⚠️ This information was not found'"**
Our AI is strictly trained *not* to hallucinate or make things up. If it says this, it genuinely can't find the answer in the PDFs you provided.
* Try rephrasing your question slightly.
* Make sure you have the correct document selected in the left sidebar.

**"I added a new PDF, but it doesn't know the answers to it yet!"**
The app has an auto-watcher, but if it ever gets stuck, you can force it to wipe and rebuild its memory by running this:
```bash
python app.py --reindex
```
