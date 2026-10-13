
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import main as rag

app = FastAPI(title="Local RAG AI Chatbot")

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "rag_index.json"
DOCUMENTS_DIR = BASE_DIR / "documents"

MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local RAG AI</title>
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #0b1020;
    color: #edf2ff;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    width: min(900px, 96%);
    height: 92vh;
    display: flex;
    flex-direction: column;
    background: #121a2d;
    border: 1px solid #26334d;
    border-radius: 18px;
    overflow: hidden;
}

header {
    padding: 22px;
    border-bottom: 1px solid #26334d;
}

h1 {
    margin: 0 0 8px;
    font-size: 24px;
}

.subtitle {
    color: #9eacc7;
    font-size: 13px;
}

.upload-panel {
    display: flex;
    gap: 10px;
    padding: 14px;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid #26334d;
}

input {
    min-width: 0;
    padding: 13px;
    border-radius: 10px;
    border: 1px solid #35435f;
    background: #0b1020;
    color: white;
    font-size: 15px;
}

#file-input {
    flex: 1;
    min-width: 220px;
}

#question {
    flex: 1;
}

button {
    border: 0;
    border-radius: 10px;
    padding: 13px 18px;
    background: #5378ff;
    color: white;
    font-weight: bold;
    cursor: pointer;
}

button:disabled {
    opacity: .55;
    cursor: wait;
}

#upload-status {
    min-height: 25px;
    padding: 5px 16px;
    font-size: 12px;
    color: #a5b4fc;
    overflow-wrap: anywhere;
}

#messages {
    flex: 1;
    overflow-y: auto;
    padding: 22px;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.message {
    max-width: 88%;
    padding: 13px 16px;
    border-radius: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.user {
    align-self: flex-end;
    background: #3659d9;
}

.bot {
    align-self: flex-start;
    background: #202b42;
}

form {
    display: flex;
    gap: 10px;
    padding: 16px;
    border-top: 1px solid #26334d;
}

footer {
    padding: 0 16px 14px;
    color: #8795b0;
    font-size: 12px;
}

@media(max-width:560px) {
    .container {
        width: 98%;
        height: 96vh;
    }

    #file-input {
        min-width: 100%;
    }

    #upload-button {
        width: 100%;
    }

    .message {
        max-width: 96%;
    }
}
</style>
</head>

<body>
<div class="container">

    <header>
        <h1>Local RAG AI</h1>
        <div class="subtitle">
            Chat with your documents using local AI
        </div>
    </header>

    <section class="upload-panel">
        <input
            type="file"
            id="file-input"
            accept=".pdf,.txt,.md"
        >
        <button type="button" id="upload-button">
            Upload &amp; Index
        </button>
    </section>

    <div id="upload-status">
        Select a PDF, TXT, or Markdown file to get started.
    </div>

    <main id="messages">
        <div class="message bot">
            Hello! Upload a document, then ask me questions about its content.
        </div>
    </main>

    <form id="chat-form">
        <input
            id="question"
            placeholder="Ask about your documents..."
            autocomplete="off"
            required
        >
        <button id="send-button" type="submit">Send</button>
    </form>

    <footer>
        Qwen 2.5 · Nomic embeddings · Local document retrieval
    </footer>

</div>

<script>
const messages = document.getElementById("messages");
const fileInput = document.getElementById("file-input");
const uploadButton = document.getElementById("upload-button");
const uploadStatus = document.getElementById("upload-status");
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");

function addMessage(text, role) {
    const element = document.createElement("div");
    element.className = "message " + role;
    element.textContent = text;

    messages.appendChild(element);
    messages.scrollTop = messages.scrollHeight;

    return element;
}

uploadButton.addEventListener("click", async () => {
    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please select a file first.";
        return;
    }

    uploadButton.disabled = true;
    uploadStatus.textContent = "Uploading and indexing... Please wait.";

    const body = new FormData();
    body.append("file", file);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: body
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        uploadStatus.textContent =
            data.message + " Total chunks: " + data.chunks;

        addMessage(
            "Document indexed: " + data.filename,
            "bot"
        );

        fileInput.value = "";

    } catch (error) {
        uploadStatus.textContent = "Error: " + error.message;
    } finally {
        uploadButton.disabled = false;
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) return;

    addMessage(question, "user");
    questionInput.value = "";
    sendButton.disabled = true;

    const waiting = addMessage(
        "Searching documents and generating an answer...",
        "bot"
    );

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Chat failed.");
        }

        waiting.textContent = data.answer;

    } catch (error) {
        waiting.textContent = "Error: " + error.message;
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
        messages.scrollTop = messages.scrollHeight;
    }
});
</script>
</body>
</html>
"""


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "index_exists": INDEX_FILE.exists(),
        "chat_model": rag.CHAT_MODEL,
        "embedding_model": rag.EMBED_MODEL,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="No index found. Upload a document first."
        )

    try:
        records = json.loads(
            INDEX_FILE.read_text(encoding="utf-8")
        )

        if not records:
            raise HTTPException(
                status_code=503,
                detail="The document index is empty."
            )

        answer = rag.chat_with_model(question, records)

        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}"
        )


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    filename = Path(file.filename.replace("\\", "/")).name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 15 MB."
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / filename

    try:
        destination.write_bytes(content)

        # Rebuild the index using documents read by main.py.
        rag.build_index()

        if not INDEX_FILE.exists():
            raise RuntimeError(
                "Index file was not created. Check build_index() in main.py."
            )

        records = json.loads(
            INDEX_FILE.read_text(encoding="utf-8")
        )

        return {
            "message": f"{filename} uploaded and indexed successfully.",
            "filename": filename,
            "chunks": len(records)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {exc}"
        )
    finally:
        await file.close()
