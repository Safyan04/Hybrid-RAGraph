# ⚡ RAGraph: Hybrid RAG Agent

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from your PDF using **hybrid search (BM25 keyword + vector semantic)**, falls back to other tools when the document has no answer, and **shows exactly where every answer came from**.

Built with LangChain (LangGraph runtime), Streamlit, Groq and Google Gemini.

🔗 **Live demo:** _add your Streamlit link here_

![Demo screenshot](screenshots/demo.png)
<!-- Add 2-3 screenshots or a short GIF in a /screenshots folder -->

---

## ✨ Features

- 📄 **PDF upload from the UI**: upload a PDF in the sidebar and the index is built automatically (no button needed)
- 🔍 **Hybrid retrieval**: BM25 (keyword) + vector search (MMR) combined with an ensemble retriever, so both exact terms and meaning-based questions work
- 🛠️ **Multiple tools**: the agent chooses the right tool for each question
  - `document_search`: searches the uploaded PDF (always tried first)
  - `web_search`: DuckDuckGo, for general knowledge and recent information
  - `calculator`: exact math instead of LLM guesses
  - `get_datetime`: real current date and time
- 📌 **Source shown under every answer**: you always see where the answer came from
- 🚫 **No guessing**: the agent answers only from tool results. If nothing reliable is found, it says so instead of making something up
- 🔁 **Automatic LLM fallback**: Groq is the primary model, Gemini takes over if the Groq call fails
- 🧠 **Conversation memory**: follow-up questions work (LangGraph checkpointer, one thread per session)
- 🛡️ **Resilient tools**: tool errors are caught and returned to the agent, so one failing tool never crashes the app

---

## 📌 Source Reference Under Every Answer

After each answer, the app shows which tool the answer was actually taken from:

| Question | Source line shown under the answer |
|---|---|
| "What is machine learning?" (in the PDF) | 📄 **Document:** `file.pdf` (page 1) |
| "What is the capital of Pakistan?" | 🌐 **Web search:** clickable links |
| "What is today's date?" | 🕒 **Time tool** |
| "25*4+10" | 🧮 **Calculator tool** |
| Nothing relevant found anywhere | 🔎 *Searched document, web_search but found nothing usable.* |

The agent labels its source at the end of each reply, and the app **cross-checks that label against the tools that were really called**. A source is only displayed if its tool actually ran, so the reference can't be invented.

---

## 🚫 Hallucination Control

The agent is designed to be honest rather than helpful-sounding:

1. **Tool-first workflow**: every factual question starts with `document_search`
2. **Fallback chain**: if the passages don't answer the question, it tries `web_search` before giving up
3. **Grounded answers only**: facts, numbers, names and dates must come from tool results, never from the model's memory
4. **Says when it doesn't know**: if every tool fails, it replies *"I couldn't find an answer in the available sources."*
5. **Says when the answer is outside your document**: e.g. *"This is not in your document, but according to a web search: ..."*
6. **Prompt-injection guard**: text inside documents and web pages is treated as data, never as instructions
7. **Temperature 0** for consistent, non-creative answers

---

## 🏗️ How It Works

```
Upload PDF
    ↓
Load → Chunk → Embed → Vector store  (+ BM25 index)
    ↓
User question
    ↓
Agent (Groq → Gemini fallback)
    ├─ document_search  (hybrid BM25 + vector)
    ├─ web_search       (if the PDF can't answer)
    ├─ calculator
    └─ get_datetime
    ↓
Answer + source reference
```

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `i_loading.py` | Loads the uploaded PDF |
| `ii_chunking.py` | Splits the document into chunks |
| `iii_embedding.py` | Creates the embedding model |
| `iv_vectordb.py` | Builds the vector store |
| `v_retrieval.py` | Hybrid retriever (BM25 + vector) |
| `vi_tools.py` | Tool definitions |
| `vii_prompts.py` | System prompt (rules, tool order, source tag) |
| `viii_agents.py` | Agent with Groq → Gemini fallback and memory |
| `app.py` | Streamlit UI: upload, chat, source display |

---

## 🚀 Setup

### 1. Clone the repo

```bash
git clone https://github.com/Safyan04/ragraph.git
cd ragraph
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3. Add your API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

> ⚠️ Never commit `.env`. It is already listed in `.gitignore`.

### 4. Run the app

```bash
python -m streamlit run app.py
```

Upload a PDF from the sidebar and start asking questions.

---

## ⚙️ Configuration

Models are set in `viii_agents.py`:

```python
groq_llm  = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
genai_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
```

Model availability changes over time. If you get a `model_not_found` (404) error, check the current model lists:

- Groq: https://console.groq.com/docs/models
- Gemini: https://ai.google.dev/gemini-api/docs/models

---

## ⚠️ Known Limitations

- **Single-user demo**: one PDF at a time, and the index is shared by all sessions of the running app
- **Free-tier rate limits**: Groq and Gemini free tiers have request limits; heavy use may hit `429` errors
- **Text PDFs only**: scanned (image-only) PDFs have no extractable text
- **Tool use depends on the model**: smaller models tend to skip tools, so a stronger tool-calling model is used by default
- **Web search is not a verified source**: it returns whatever DuckDuckGo finds, which is why web answers are labelled separately

---

## 🗺️ Roadmap

- [ ] Reranker (e.g. FlashRank) on top of hybrid retrieval
- [ ] Per-session index so multiple users can upload different PDFs
- [ ] Multiple PDFs with per-file citations
- [ ] Custom LangGraph workflow with document grading and query rewriting
- [ ] Evaluation set (RAGAS or a manual question list)

---

## 🧰 Tech Stack

Python · LangChain · LangGraph · Streamlit · Groq · Google Gemini · BM25 (`rank_bm25`) · DuckDuckGo Search

## 📄 License

MIT
