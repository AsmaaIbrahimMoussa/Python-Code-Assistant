import os
import time
import pickle
import streamlit as st
from typing_extensions import TypedDict
from datasets import load_dataset
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from qdrant_client.models import CollectionStatus

# Save and load vectorstore

def load_vectorstore(path="vectorstore.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

# Step 1: Prepare dataset
def prepare_dataset():
    data = load_dataset("openai_humaneval")["test"]
    return [
        Document(
            page_content=example["prompt"],
            metadata={"task_id": example["task_id"], "solution": example["canonical_solution"]}
        ) for example in data
    ]

# Step 2: Setup vectorstore
def setup_vectorstore(tasks):
    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/codebert-base",
        encode_kwargs={"normalize_embeddings": True}
    )

    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("Missing QDRANT_API_KEY environment variable.")

    qdrant_client = QdrantClient(
        url="https://0762546b-450b-4d4c-bc82-0c41230701da.us-west-2-0.aws.cloud.qdrant.io",
        api_key=qdrant_api_key,
        timeout=180.0
    )

    collection_name = "humaneval_tasks"
    collections = qdrant_client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        qdrant_client.delete_collection(collection_name)
        while True:
            try:
                status = qdrant_client.get_collection(collection_name).status
                if status == CollectionStatus.GREEN:
                    break
            except:
                break
        time.sleep(1)

    vectorstore = Qdrant.from_documents(
        documents=tasks,
        embedding=embeddings,
        collection_name=collection_name,
        url="https://0762546b-450b-4d4c-bc82-0c41230701da.us-west-2-0.aws.cloud.qdrant.io",
        api_key=qdrant_api_key,
        prefer_grpc=False,
    )

    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        vectorstore.add_documents(batch)

    return embeddings, vectorstore

# Step 3: Code RAG Pipeline
class CodeRAGPipeline:
    def __init__(self, embeddings, vectorstore, client):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.client = client

    def retrieve_similar_tasks(self, query, k=3):
        query_embedding = self.embeddings.embed_query(query)
        return self.vectorstore.similarity_search_by_vector(query_embedding, k=k)

    def format_prompt(self, query, retrieved_docs):
        examples = ""
        for doc in retrieved_docs:
            examples += f"# Example Task:\n{doc.page_content}\n{doc.metadata['solution']}\n\n"
        return (
            "# Below are several examples of Python programming tasks and their solutions.\n"
            "# Use these to help write a function for the new task at the end.\n\n"
            f"{examples}"
            f"# New Task:\n{query}\n# Your Python solution:\n"
        )

# Step 4: Code generation node

def generate_code_node(state):
    query = state["input"]
    gemini_client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta")
    rag_pipeline = CodeRAGPipeline(embeddings, vectorstore, gemini_client)
    similar_docs = rag_pipeline.retrieve_similar_tasks(query)
    formatted_prompt = rag_pipeline.format_prompt(query, similar_docs)

    response = gemini_client.chat.completions.create(
        model="models/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful Python code generator, returning only the code."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.4,
        max_tokens=500
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

# Step 5: Code explanation node
def explain_code_node(state):
    code = state["input"]
    gemini_client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta")
    response = gemini_client.chat.completions.create(
        model="models/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful Python code explainer."},
            {"role": "user", "content": f"Explain the following Python code:\n{code}"}
        ],
        temperature=0.4,
        max_tokens=500
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}

# Step 6: LangGraph routing
def chat_node(state):
    return {"input": state["input"]}

def router_node(state):
    user_input = state["input"].lower()
    if "explain" in user_input or "explanation" in user_input:
        return {"__path__": "explain_code"}
    else:
        return {"__path__": "generate_code"}

class AppState(TypedDict):
    input: str
    messages: list

builder = StateGraph(state_schema=AppState)
builder.add_node("chat", RunnableLambda(chat_node))
builder.add_node("router", RunnableLambda(router_node))
builder.add_node("generate_code", RunnableLambda(generate_code_node))
builder.add_node("explain_code", RunnableLambda(explain_code_node))

builder.set_entry_point("chat")
builder.add_edge("chat", "router")
builder.add_conditional_edges(
    source="router",
    path=lambda state: state["__path__"],
    path_map={
        "generate_code": "generate_code",
        "explain_code": "explain_code"
    }
)

builder.add_edge("generate_code", END)
builder.add_edge("explain_code", END)
graph = builder.compile()

# Streamlit UI
def main():
    st.set_page_config(page_title="Python Code Assistant", layout="centered")
    st.title("Python Code Assistant")

    @st.cache_resource
    def get_data():
        tasks_path = "tasks.pkl"

        if os.path.exists(tasks_path) and os.path.getsize(tasks_path) > 0:
            try:
                with open(tasks_path, "rb") as f:
                    tasks = pickle.load(f)
            except Exception as e:
                os.remove(tasks_path)
                tasks = prepare_dataset()
                with open(tasks_path, "wb") as f:
                    pickle.dump(tasks, f)
        else:
            tasks = prepare_dataset()
            with open(tasks_path, "wb") as f:
                pickle.dump(tasks, f)

        emb, vs = setup_vectorstore(tasks)
        return emb, vs


    global embeddings, vectorstore
    embeddings, vectorstore = get_data()

    user_prompt = st.text_area("How can I help you with Python today?", height=250)

    if st.button("Get Answer"):
        if not user_prompt.strip():
            st.warning("Please enter a task.")
        else:
            with st.spinner("Processing..."):
                events = graph.stream({"input": user_prompt})
                final_state = None
                for event in events:
                    for value in event.values():
                        if value and "messages" in value:
                            final_state = value

                if final_state and "messages" in final_state and final_state["messages"]:
                    output = final_state["messages"][-1]["content"]
                    path = final_state.get("__path__")
                    st.success("Response generated!")
                    st.code(output, language="python")
                    st.download_button("Download Code", output, file_name="code.py")
                else:
                    st.error("No response generated.")

if __name__ == "__main__":
    main()
