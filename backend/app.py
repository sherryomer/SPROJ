import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
from typing import List
import json
import pandas as pd
import tabula
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Constants
VECTORSTORE_PATH = "chroma_db"
PDF_FILES = ["Handbook.pdf", "course_schedule.pdf"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BATCH_SIZE = 100

def process_course_schedule(pdf_path: str) -> List[Document]:
    """Process course schedule PDF specifically for tabular data"""
    try:
        # Extract tables from PDF using tabula
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        documents = []
        
        for table_idx, table in enumerate(tables):
            # Clean the table data
            table = table.fillna('')
            table = table.astype(str)
            
            # Create more semantic text representation
            table_text = []
            headers = table.columns.tolist()
            
            for _, row in table.iterrows():
                # Create a more natural language representation
                row_text = []
                for header, value in zip(headers, row):
                    if value.strip():  # Only include non-empty values
                        row_text.append(f"{header}: {value.strip()}")
                table_text.append(" | ".join(row_text))
            
            table_content = "\n".join(table_text)
            
            doc = Document(
                page_content=table_content,
                metadata={
                    "source_type": "course_schedule",
                    "table_index": table_idx,
                    "is_tabular": True,
                    "headers": headers  # Store headers in metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"Processed {len(tables)} tables from course schedule")
        return documents
        
    except Exception as e:
        logger.error(f"Error processing course schedule: {str(e)}")
        raise

def process_handbook(pdf_path: str) -> List[Document]:
    """Process handbook PDF using standard text processing"""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        for page in pages:
            page.metadata["source_type"] = "handbook"
            
        return splitter.split_documents(pages)
        
    except Exception as e:
        logger.error(f"Error processing handbook: {str(e)}")
        raise

def process_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Process multiple PDFs with specialized handling for each type"""
    documents = []
    
    for pdf_path in pdf_paths:
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
                
            logger.info(f"Loading PDF: {pdf_path}")
            
            # Use specialized processing for course schedule
            if "course_schedule" in pdf_path.lower():
                pdf_documents = process_course_schedule(pdf_path)
            else:
                pdf_documents = process_handbook(pdf_path)
                
            documents.extend(pdf_documents)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
            
    logger.info(f"Total documents after merging PDFs: {len(documents)}")
    return documents

def create_vector_store(documents: List[Document], embedding_model) -> Chroma:
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=VECTORSTORE_PATH
    )

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {len(texts)//BATCH_SIZE + 1}")
        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas
        )

    vector_store.persist()
    return vector_store

def initialize_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    if os.path.exists(VECTORSTORE_PATH):
        logger.info("Loading existing vector store...")
        return Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embedding_model
        )
    else:
        logger.info("Creating new vector store from PDFs...")
        documents = process_pdfs(PDF_FILES)
        return create_vector_store(documents, embedding_model)

# Initialize components
try:
    chroma_vector_database = initialize_vector_store()
    retriever = chroma_vector_database.as_retriever(
        search_kwargs={"k": 10}
    )
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
    logger.info("Initialization completed successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

SYSTEM_TEMPLATE = """
You are a knowledgeable and helpful university assistant for LUMS.

Your job is to answer questions about:
- Campus facilities
- Academic programs
- Events and student life
- Student services
- Admission processes
- Course schedules and timings
- University culture and policies

For course-related queries (e.g., course timings, schedules, instructors):
1. Always check the course schedule information first
2. When searching for course information:
   - Look for exact course codes (e.g., "CS101", "MATH101")
   - Also consider partial matches (e.g., "CS" for "CS101")
   - Check for variations in course names
3. If the course information is found in the context:
   - Provide all available sections
   - Include complete details: course code, name, timings, days, instructor
   - Format the response clearly with bullet points
4. If the course is not found:
   - First verify if the course code/name is correct
   - Check if there might be a typo in the query
   - Only then state that the course information is not available
5. For ambiguous queries:
   - Ask for clarification about which course/section
   - Provide available options if multiple matches exist

For other LUMS-related queries:
- Always prioritize using the retrieved context to answer questions
- If no useful context is available but the question is still related to LUMS, you may respond factually using your general knowledge
- If the question is unrelated to LUMS, politely indicate that you are designed only for LUMS-related queries

Maintain a concise, helpful, and professional tone at all times.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
human_message_prompt = HumanMessagePromptTemplate.from_template("{context}\n\nQuestion: {query}")

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_query = data.get('query', '')
        if not user_query:
            return jsonify({"error": "No query provided."}), 400

        global chat_history
        if 'chat_history' not in globals():
            chat_history = ChatMessageHistory()

        # Enhanced course schedule detection
        course_keywords = [
            "when", "time", "schedule", "class", "instructor", "taught",
            "course", "section", "timing", "day", "days", "meeting",
            "acct", "cs", "math", "econ", "mgmt", "sds", "engl", "hist",
            "morn", "afternoon", "evening", "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday"
        ]
        
        # Check if query is course-related
        lower_query = user_query.lower()
        is_course_query = any(keyword in lower_query for keyword in course_keywords)
        
        if is_course_query:
            logger.info("Applying Course Schedule filter for query")
            # Use a more specific filter for course schedule
            filtered_retriever = chroma_vector_database.as_retriever(
                search_kwargs={
                    "k": 15,
                    "filter": {"source_type": "course_schedule"},
                    "score_threshold": 0.3  # Lower threshold
                }
            )
            top_docs = filtered_retriever.get_relevant_documents(user_query)
            
            # If no relevant documents found, try without the filter
            if not top_docs:
                logger.info("No course schedule documents found, trying general search")
                top_docs = retriever.get_relevant_documents(user_query)
        else:
            top_docs = retriever.get_relevant_documents(user_query)

        context = " ".join([doc.page_content for doc in top_docs])

        messages = [
            system_message_prompt.format(),
            *chat_history.messages,
            human_message_prompt.format(context=context, query=user_query)
        ]

        assistant_response = llm(messages)

        def format_response(text):
            import re
            text = re.sub(r"^#{1,6}\s*(.+?):?\s*$", r"\1:", text, flags=re.MULTILINE)
            text = text.replace("**", "").replace("*", "")
            text = re.sub(r"^\d+\.\s*", "- ", text, flags=re.MULTILINE)
            text = re.sub(r"^-\s*", "- ", text, flags=re.MULTILINE)
            text = re.sub(r"\n{2,}", "\n\n", text)
            return text.strip()

        formatted_answer = format_response(assistant_response.content)
        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(formatted_answer)

        return jsonify({
            "answer": formatted_answer,
            "retrieved_contexts": [doc.page_content[:300] + "..." for doc in top_docs],
            "raw_context": context
        })

    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test_rag', methods=['POST'])
def test_rag():
    try:
        data = request.json
        test_query = data.get('query', '')

        if not test_query:
            return jsonify({"error": "No test query provided"}), 400

        docs = retriever.get_relevant_documents(test_query)
        seen_contents = set()
        duplicate_count = 0
        results = []

        for i, doc in enumerate(docs):
            if doc.page_content in seen_contents:
                duplicate_count += 1
            else:
                seen_contents.add(doc.page_content)

            results.append({
                "document_number": i + 1,
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "content_hash": hash(doc.page_content),
                "relevance_score": "N/A"
            })

        return jsonify({
            "query": test_query,
            "documents_retrieved": len(docs),
            "unique_documents": len(seen_contents),
            "duplicate_count": duplicate_count,
            "results": results
        })

    except Exception as e:
        logger.error(f"Error in test_rag endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        global chat_history
        chat_history = ChatMessageHistory()
        return jsonify({"message": "Chat history reset successfully."})
    except Exception as e:
        logger.error(f"Error resetting chat history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    try:
        all_data = chroma_vector_database.get()
        return jsonify({
            "status": "running",
            "total_documents": len(all_data['documents']),
            "vectorstore_path": VECTORSTORE_PATH,
            "pdfs_loaded": PDF_FILES
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    app.run(debug=False)
