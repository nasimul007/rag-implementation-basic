from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime


persist_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "What was the original name of Microsoft before it became Microsoft?\n"

# retriever = db.as_retriever(search_kwargs={"k": 3})
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
            "k": 5,
            "score_threshold": 0.3  #only return results with similarity score >= 0.3
        }
    )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on following retrieved documents, answer the question: {query}\n\n
Documents: {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, 
say "I don't have enough information to answer that question based on the provided documents."
"""

system_message = "You are a helpful assistant, that answers question based on the provided documents. " \
"Always use only the information from the documents to answer the question. "

# Define the messages for the model
messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=combined_input),
]

llm = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0,
)

print("Start Time: ", datetime.now().time())

ai_msg = llm.invoke(messages)

print("\n--- AI Response ---")
print(ai_msg.content)

print("End Time: ", datetime.now().time())


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