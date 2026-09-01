import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(chunks, index_name="rag-pdf-index", namespace="default"):
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY environment variable missing in .env file!")

    pc = Pinecone(api_key=pinecone_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Clean existing namespace completely before upserting
    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace=namespace)
        time.sleep(2)  # Pinecone deletion latency delay
        print(f"✅ Cleared existing vectors in namespace: '{namespace}'")
    except Exception as e:
        print(f"ℹ️ Index clear log: {e}")

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=index_name,
        namespace=namespace
    )


def setup_rag_chain(index_name="rag-pdf-index", model_name="openai/gpt-oss-20b", top_k=6, namespace="default"):
    pinecone_key = os.getenv("PINECONE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not pinecone_key or not groq_key:
        raise ValueError("Missing API Keys in .env file!")

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=get_embeddings(),
        namespace=namespace
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": 20}
    )

    llm = ChatOpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        model=model_name,
        temperature=0.2
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question that can be understood without the chat history. Do NOT answer the question, just reframe it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    system_prompt = (
        "You are an intelligent, articulate, and conversational AI Assistant for document analysis.\n\n"
        "CORE RULE - DOCUMENT LOCK: You are 100% bound to the provided Context. Your entire knowledge for this task is ONLY what is written in the Context below. No outside knowledge allowed.\n\n"
        "GUIDELINES:\n"
        "1. CONVERSATIONAL & ENGAGING: Maintain a helpful, clear tone. Use headings, bullet points, or steps to explain answers thoroughly.\n"
        "2. STRICT ACCURACY: Answer SOLELY based on the provided Context. Do not infer, assume, or add external information.\n"
        "3. QUOTE WHEN POSSIBLE: If relevant, reference the specific part of the Context.\n"
        "4. GREETINGS: If the user says 'hi', 'hello', or small talk, respond warmly and state you are ready to analyze their document.\n"
        "5. ABSENCE FALLBACK - MANDATORY: If the answer is not found in the Context, you MUST reply EXACTLY with:\n"
        "   'The answer is not available in the provided document.'\n"
        "   Do not guess, do not elaborate, do not use outside info.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain), vectorstore