# 📚 Document-based Question Answering App

This is a Flask-based web app that allows you to upload `.pdf`, `.pptx`, or `.docx` documents and ask questions based on their content. The app uses **LangChain**, **HuggingFace embeddings**, and **Cohere LLM** to perform Retrieval-Augmented Generation (RAG) and return accurate answers from your files.

---

## 🔧 Features

- Upload and process `.pdf`, `.docx`, or `.pptx` files
- Extract and chunk the text using LangChain's text splitter
- Embed chunks using HuggingFace sentence-transformers
- Store and retrieve data using FAISS vectorstore
- Use Cohere's LLM for accurate, context-aware answers

---

## 📁 Folder Structure

project-folder/
│
├── app.py # Flask server with RAG pipeline
├── templates/
│ └── index.html # HTML form for file upload and question input
├── uploads/ # Uploaded files get saved here
├── .env # Your API key for Cohere
└── README.md # You're reading this!


---

## 🧠 How it Works

1. **Upload** a supported document
2. **Text is extracted** and **chunked** for better processing
3. Text chunks are embedded and stored in a **FAISS** index
4. A question is passed to a **RAG chain**
5. The app retrieves relevant text and queries **Cohere LLM**
6. The answer is displayed on the webpage

---

## 🚀 Getting Started

### 🔨 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```
Required Python packages:

Flask

python-dotenv

PyPDF2

python-pptx

python-docx

langchain

sentence-transformers

cohere

🌐 Set Up .env
Create a .env file in your root directory:

ini
Copy
Edit
cohere_api_key=your_cohere_api_key_here
▶️ Run the App
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser
