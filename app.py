import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import requests
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Function to clean text data
def cleaning(text):
    """
    Cleans the text by removing unwanted characters and redundant whitespace.
    """
    text = text.replace('\n', ' ').replace('^', ' ').replace('[edit]', ' ')
    return ' '.join(text.split())

# Scrape and clean data from the LUMS Wikipedia page
url = "https://en.wikipedia.org/wiki/Lahore_University_of_Management_Sciences"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

content = soup.find('div', {'id': 'bodyContent'}).get_text()
paragraphs = content.split('\n\n')

new_cleaned_text = []
for paragraph in paragraphs:
    if paragraph.strip():
        new_cleaned_text.append(cleaning(paragraph))

# Initialize vector database
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
chroma_vector_database = Chroma(embedding_function=embedding_model, persist_directory="chroma_data")

# Add cleaned text to vector store
for chunk in [new_cleaned_text[i:i + 100] for i in range(0, len(new_cleaned_text), 100)]:
    chroma_vector_database.add_texts(texts=chunk)

retriever = chroma_vector_database.as_retriever(search_kwargs={"k": 5})

# Initialize the chatbot
llm = ChatOpenAI(model='gpt-4o-mini-2024-07-18', temperature=0.7)

SYSTEM_TEMPLATE = """
You are a knowledgeable and helpful university assistant for LUMS. Provide concise and informative answers to questions about campus facilities, academic programs, events, student services, admission processes, and university culture. Remember previous interactions to provide relevant and personalized responses.
"""

@app.route('/query', methods=['POST'])
def query():
    """
    Handle chatbot queries by retrieving relevant documents and generating a response.
    """
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"answer": "No query provided. Please ask a question."}), 400

    try:
        # Use LangChain's ChatMessageHistory to maintain chat history
        if 'chat_history' not in globals():
            global chat_history
            chat_history = ChatMessageHistory()

        # Retrieve top documents
        top_docs = retriever.get_relevant_documents(user_query)
        context = " ".join([doc.page_content for doc in top_docs])

        # Build the message history
        messages = [SystemMessage(content=SYSTEM_TEMPLATE)]
        messages.extend(chat_history.messages)
        messages.append(HumanMessage(content=user_query))

        # Generate response using LangChain
        prompt_template = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = prompt_template.format_prompt(context=context)  # Properly format the prompt
        assistant_response = llm(formatted_prompt.to_messages())  # Generate response

        # Update chat history
        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(assistant_response.content)

        return jsonify({"answer": assistant_response.content})
    except Exception as e:
        print("Error during query processing:", str(e))
        return jsonify({"answer": f"Error: {str(e)}"}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset the chat history.
    """
    global chat_history
    chat_history = ChatMessageHistory()
    return jsonify({"message": "Chat history has been reset."})

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    app.run(debug=True)
