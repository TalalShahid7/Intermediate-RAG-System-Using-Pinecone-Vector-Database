import os
import re
import streamlit as st
from dotenv import load_dotenv
from loader import process_file
from rag_pipeline import build_vectorstore, setup_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Intermediate RAG System", page_icon="📘", layout="wide")

st.title("📘 Enterprise RAG System using Pinecone DB")

st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.slider("Chunk Size", 300, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200, 50)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 6)

index_name = st.sidebar.text_input("Pinecone Index", value="rag-pdf-index")
namespace_input = st.sidebar.text_input("Namespace Target", value="assignment-namespace")

selected_model = st.sidebar.selectbox(
    "LLM Model",
    ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct"]
)

uploaded_file = st.sidebar.file_uploader("Upload Document (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.sidebar.button("Process & Index Document", use_container_width=True):
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"):
        st.error("API Keys missing in .env file!")
    elif not uploaded_file:
        st.error("Please upload a document first.")
    else:
        try:
            with st.spinner("Processing & Indexing..."):
                chunks = process_file(uploaded_file, chunk_size, chunk_overlap)
                build_vectorstore(chunks, index_name, namespace_input)
                st.session_state.rag_chain, st.session_state.vectorstore = setup_rag_chain(
                    index_name, selected_model, top_k, namespace_input
                )
                st.session_state.all_chunks = chunks
                st.session_state.file_name = uploaded_file.name
                st.success(f"Successfully indexed {len(chunks)} chunks!")
        except Exception as e:
            st.error(f"Error: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_history = [
    HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
    for msg in st.session_state.messages
]

if user_query := st.chat_input("Ask something about the document..."):
    clean_query = user_query.strip().lower().rstrip("!.,?")
    is_greeting = clean_query in ["hi", "hello", "hey", "salam", "aoa", "good morning"]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if is_greeting:
            answer = "Hello! Upload a document and click 'Process & Index Document' to get started."
            st.markdown(answer)
        elif not st.session_state.rag_chain:
            answer = "Please index a document using the sidebar first."
            st.warning(answer)
        else:
            with st.spinner("Thinking..."):
                page_match = re.search(r'page\s*(\d+)', clean_query)
                page_handled = False

                if page_match and "all_chunks" in st.session_state:
                    requested_page = int(page_match.group(1))
                    
                    # 1. Check for PDFs with exact page numbers
                    matching_chunks = [
                        c for c in st.session_state.all_chunks 
                        if str(c.metadata.get("page")) == str(requested_page)
                    ]

                    # 2. Fallback for Non-PDF files (DOCX/TXT/CSV) where page metadata is "N/A"
                    if not matching_chunks and requested_page == 1:
                        matching_chunks = [
                            c for c in st.session_state.all_chunks 
                            if c.metadata.get("page") == "N/A"
                        ]

                    if matching_chunks:
                        page_handled = True
                        context_str = "\n\n".join([c.page_content for c in matching_chunks])
                        
                        groq_key = os.getenv("GROQ_API_KEY")
                        llm = ChatOpenAI(
                            api_key=groq_key,
                            base_url="https://api.groq.com/openai/v1",
                            model=selected_model,
                            temperature=0.2
                        )
                        
                        # Added Chat History to direct page prompt for seamless flow
                        direct_prompt = ChatPromptTemplate.from_messages([
                            ("system", (
                                "You are an AI Document Assistant.\n"
                                "Summarize and explain the content from the provided document context clearly and comprehensively.\n"
                                "Use headings, key bullet points, and maintain an engaging tone.\n\n"
                                "Document Content:\n{context}"
                            )),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}")
                        ])
                        
                        chain = direct_prompt | llm
                        res = chain.invoke({
                            "context": context_str,
                            "chat_history": chat_history,
                            "input": user_query
                        })
                        answer = res.content
                        
                        st.markdown(answer)
                        with st.expander("📌 Source Attribution & Metadata"):
                            for idx, doc in enumerate(matching_chunks):
                                st.markdown(f"**Source {idx+1}:** `{doc.metadata.get('source', 'Doc')}` | **Page:** `{doc.metadata.get('page')}` | **Chunk ID:** `{doc.metadata.get('chunk_id')}`")
                                st.caption(f'"{doc.page_content.strip()[:200]}..."')
                                st.markdown("---")

                # Standard RAG search if page matching wasn't applicable or required
                if not page_handled:
                    response = st.session_state.rag_chain.invoke({"input": user_query, "chat_history": chat_history})
                    answer = response["answer"]
                    st.markdown(answer)

                    if "context" in response and response["context"] and "The answer is not available in the provided document." not in answer:
                        with st.expander("📌 Source Attribution & Metadata"):
                            results = st.session_state.vectorstore.similarity_search_with_score(user_query, k=top_k, namespace=namespace_input)
                            for idx, (doc, score) in enumerate(results):
                                st.markdown(f"**Source {idx+1}:** `{doc.metadata.get('source', 'Doc')}` | **Page:** `{doc.metadata.get('page', 'N/A')}` | **Chunk ID:** `{doc.metadata.get('chunk_id', 'N/A')}` | **Score:** `{score:.4f}`")
                                st.caption(f'"{doc.page_content.strip()[:200]}..."')
                                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})