# SJVN Suvidha — AI Document Assistant
**Powered by Ollama (100% Free, Local, & Private)**

SJVN Suvidha is a production-grade, Retrieval-Augmented Generation (RAG) chatbot designed to parse, search, and explain complex HR policies, Delegation of Powers (DoP), and company guidelines. It runs entirely on your local hardware—meaning zero API costs, no internet dependency, and complete data privacy.

---

## 🌟 Key Features

* 🧠 **Self-Learning Memory Loop:** Includes a feedback UI (👍/👎). If the AI gives a wrong answer, users can submit a correction. The system permanently saves this to a SQLite database and instantly overrides the AI the next time that question is asked.
* 🎤 **Voice-to-Text & Read Aloud:** Full microphone integration for hands-free querying, plus Text-to-Speech (TTS) to read answers aloud.
* 🇮🇳 **Bilingual Semantic Search:** Seamlessly handles queries in English, Hindi (Devanagari), or Hinglish. Automatically translates queries in the backend for perfect vector math matching.
* 📊 **Smart Table Extraction & Export:** AI generates structured Markdown tables from documents, complete with **1-click PDF and CSV download** buttons.
* 🔗 **Deep-Link Citations:** Every answer cites its source. Clicking the citation opens the exact page in the original PDF directly in your browser.
* 🛡️ **Admin Dashboard & Audit Logs:** A password-protected portal to manage dynamic FAQs and view a history of all user queries and IP addresses.
* ⚡ **Optimized for Low VRAM:** Custom context-window tuning and aggressive vector routing allow this to run smoothly even on 4GB GPUs (like an RTX 3050).

---

## 🚀 Quick Setup (5 Minutes)

### Step 1 — Install Ollama
Download the local AI engine from: **https://ollama.com/download**
*(Available for Windows, Mac, Linux)*

### Step 2 — Pull the Required Models
Open your terminal/command prompt. You must pull **two** models: the embedding model (for the Vector DB) and the language model (for the chat).

**1. Pull the Vector Embedding Model (Required):**
ollama pull nomic-embed-text

**2. Pull the Chat Model (Pick ONE based on your RAM):**
- 16 GB+ RAM | Qwen 2.5 (14B) | ollama pull qwen2.5:14b | ⭐⭐⭐⭐⭐ Best
- 8 GB+ RAM  | Qwen 2.5 (7B)  | ollama pull qwen2.5:7b  | ⭐⭐⭐⭐ Great
- 4 GB+ RAM  | Qwen 2.5 (3B)  | ollama pull qwen2.5:3b  | ⭐⭐⭐ Fast & Good
(Recommendation: qwen2.5:3b is highly optimized for this specific codebase if you are running on a standard laptop GPU.)

### Step 3 — Start Ollama
Keep this terminal open in the background while using the chatbot:
ollama serve

### Step 4 — Install Python Dependencies
Open a new terminal window and install the required Python libraries:
pip install flask pdfplumber cryptography numpy

### Step 5 — Add Your Documents
Place your official company PDF files into the documents/ folder.
documents/
  dop.pdf
  5889_LEAVE_RULE_Policy.pdf
  Procurement_Policy.pdf
  (add any other PDFs here)

### Step 6 — Start the Chatbot
python app.py
*(Note: The first time you run this, it will automatically read, chunk, and mathematically index all your PDFs. This may take a minute or two.)*

### Step 7 — Open the App
Open your web browser and go to:
http://localhost:5000

---

## 🔐 Admin & FAQ Management

SJVN Suvidha includes a secure backend for administrators to monitor usage and update the dynamic FAQ sidebar.

* **Access the Admin Panel:** Click the **⚙ Admin** button in the sidebar.
* **Default Password:** SJVN@Admin2024 (Can be changed in app.py environment variables)
* **Capabilities:** - View real-time Audit Logs (User IPs, timestamps, and queries).
  - Add or delete questions from the sidebar FAQ database.

---

## 🎛️ Environment Variables (Optional)

You can customize the server behavior without changing the code by setting these environment variables before running app.py:
- OLLAMA_HOST: Point to a remote Ollama server if hosting elsewhere. (Default: http://localhost:11434)
- OLLAMA_MODEL: Force the app to use a specific model. (Default: Auto-detects)
- PORT: Change the HTTP port. (Default: 5000)
- ADMIN_PASSWORD: Change the dashboard login password. (Default: SJVN@Admin2024)

---

## 🛠️ Troubleshooting

**"Microphone access is denied / Voice isn't working"**
Browsers block microphones on non-HTTPS sites. To fix this:
* If testing on the same PC: Access the app via http://localhost:5000 (Browsers trust localhost).
* If testing on your phone: Use a tool like Ngrok (ngrok http 5000) to create a secure HTTPS tunnel to your phone.

**"The AI is looping or repeating the same sentence"**
Ensure you are using the qwen2.5 family of models. The llm.py file has a strict repeat_penalty configured specifically for Qwen models to prevent degeneration. 

**"It says '⚠️ This information was not found'"**
The AI is strictly hardcoded not to hallucinate. If it says this, it cannot find the text in the PDFs. 
* Try rephrasing the question.
* Ensure the correct document is selected in the left sidebar.

**"I added a new PDF but it doesn't know the answers"**
The app has a watcher, but if it gets stuck, you can force a hard wipe and rebuild of the Vector Database by running:
python app.py --reindex
