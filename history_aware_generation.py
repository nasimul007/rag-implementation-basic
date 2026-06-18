import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


persist_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
llm_model = os.getenv("LLM_MODEL")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  #only return results with similarity score >= 0.3
    }
)

llm = ChatOllama(
    model=llm_model,
    temperature=0,
)


def generate_combined_input(query, relevant_docs):
    return f"""Based on following retrieved documents, answer the question: {query}\n\n
    Documents: {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
    Please provide a clear, helpful answer using only the information from these documents. 
    If you can't find the answer in the documents, 
    say "I don't have enough information to answer that question based on the provided documents."
    """

system_message = "You are a helpful assistant, that answers question based on the provided documents. " \
"Always use only the information from the documents to answer the question. "


chat_history = []

def generate_question(user_query):
    # Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(
                content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                ),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_query}")
        ]

        ai_msg = llm.invoke(messages)
        search_question = ai_msg.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_query

    return search_question


def ask_question(user_query):
    #  -----Step 1: Make the question clear using conversation history------
    search_question = generate_question(user_query)
    # print(f"User Query: {search_question}\n")

    # ---------Step 2: Find relevant documents--------
    relevant_docs = retriever.invoke(search_question)
    print(f"Found {len(relevant_docs)} relevant documents:")

    # for i, doc in enumerate(relevant_docs, 1):
    #     # Show first 2 lines of each document
    #     lines = doc.page_content.split('\n')[:2]
    #     preview = '\n'.join(lines)
    #     print(f"  Doc {i}: {preview}...")

    # ---------Step 3: Create final prompt------------------
    combined_input = generate_combined_input(search_question, relevant_docs)
    # Define the messages for the model
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=combined_input),
    ]

    # ---------------Step 4: Get AI response------------------
    ai_msg = llm.invoke(messages)
    answer = ai_msg.content.strip()
    print(f"\nAI Response: {answer}\n\n")

    # --------------------Step 5: Remember this conversation-----------------------
    chat_history.append(HumanMessage(content=search_question))
    chat_history.append(AIMessage(content=answer))

    # print(f"\nchat_history: {chat_history}\n\n")

def start_chat():
    print("Ask me questions! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Hope to see you again!")
            break
        
        ask_question(user_input)


if __name__ == "__main__":
    start_chat()
    pass



# example questions to make the conversation history aware:

# 1. What was NVIDIA's first graphics accelerator called?
# 2. when it was developed?
# 3. Revenue from it?
# 4. who was the founder of the company?
# 5. tell me more about them

# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
# 9. "When did Microsoft reached a trillion-dollar market cap?"