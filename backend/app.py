import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# PDF processing with smaller chunks
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    documents = splitter.split_documents(pages)
    return [doc.page_content for doc in documents]

# Setup
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore_path = "chroma_data"
pdf_path = "Handbook.pdf"

# Build or load vector DB
if os.path.exists(os.path.join(vectorstore_path, "index")):
    print("🔄 Loading existing vector database...")
    chroma_vector_database = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)
else:
    print("✨ Creating vector database from handbook...")
    all_docs = []

    if os.path.exists(pdf_path):
        handbook_chunks = process_pdf(pdf_path)
        print("✅ Sample handbook chunks:")
        for i, chunk in enumerate(handbook_chunks[:3]):
            print(f"\n📄 Handbook Chunk {i + 1}:\n{chunk[:300]}...\n")
        for chunk in handbook_chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": "handbook"}))
    else:
        print("❌ Handbook.pdf not found!")
        handbook_chunks = []

    # Create vectorstore from handbook only
    chroma_vector_database = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        persist_directory=vectorstore_path
    )
    chroma_vector_database.persist()
    print("💾 Vectorstore persisted.")

# Diagnostic check
print("\n🔍 Verifying handbook chunks in DB...")
all_data = chroma_vector_database.get()
docs = all_data['documents']
metas = all_data['metadatas']
handbook_count = sum(1 for m in metas if m.get("source") == "handbook")
print(f"✅ Vectorstore loaded: {handbook_count} handbook chunks\n")

# Retriever
retriever = chroma_vector_database.as_retriever(
    search_kwargs={"k": 5, "filter": {"source": "handbook"}}
)

# LLM
llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', temperature=0.7)

SYSTEM_TEMPLATE = """
You are a knowledgeable and helpful university assistant for LUMS.
Provide concise and informative answers to questions about campus facilities, academic programs, events, student services, admission processes, university culture, and policies outlined in the student handbook.
Use only the retrieved context to support your responses.
"""

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"answer": "No query provided."}), 400

    try:
        global chat_history
        if 'chat_history' not in globals():
            chat_history = ChatMessageHistory()

        # Retrieve top chunks
        top_docs = retriever.get_relevant_documents(user_query)
        context = " ".join([doc.page_content for doc in top_docs])

        # Debug: show chunks
        print("\n--- 🔍 Retrieved Chunks from Handbook ---")
        retrieved_chunks = []
        for i, doc in enumerate(top_docs):
            source = doc.metadata.get("source", "unknown")
            chunk = doc.page_content[:300] + "..."
            print(f"\nChunk {i+1} [source: {source}]:\n{chunk}")
            retrieved_chunks.append(f"[{source}] {chunk}")

        # LLM prompt
        messages = [SystemMessage(content=SYSTEM_TEMPLATE)]
        messages.extend(chat_history.messages)
        messages.append(HumanMessage(content=user_query))

        prompt = ChatPromptTemplate.from_messages(messages).format_prompt(context=context)
        assistant_response = llm(prompt.to_messages())  # warning ok

        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(assistant_response.content)

        return jsonify({
            "answer": assistant_response.content,
            "retrieved_contexts": retrieved_chunks
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global chat_history
    chat_history = ChatMessageHistory()
    return jsonify({"message": "Chat history reset."})

# ✅ Diagnostic route to search for Vice Chancellor chunks
@app.route('/search_vc', methods=['GET'])
def search_vc():
    all_data = chroma_vector_database.get()
    matches = []

    for doc, meta in zip(all_data['documents'], all_data['metadatas']):
        lower_doc = doc.lower()
        if "vice chancellor" in lower_doc or "ali cheema" in lower_doc:
            matches.append(doc[:500])

    if matches:
        return jsonify({"matches": matches})
    else:
        return jsonify({"message": "No relevant chunks found."}), 404

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    app.run(debug=True)
