
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


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Intermediate RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG System using Pinecone DB")



# 2. INITIALIZE SESSION STATE


if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "active_namespace" not in st.session_state:
    st.session_state.active_namespace = None



# 3. SIDEBAR CONFIGURATION


st.sidebar.header("⚙️ Configuration")

chunk_size = st.sidebar.slider(
    "Chunk Size", 300, 2000, 1000, 100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap", 0, 500, 200, 50
)

top_k = st.sidebar.slider(
    "Top-K Retrieval", 1, 10, 6
)

index_name = st.sidebar.text_input(
    "Pinecone Index",
    value="rag-pdf-index"
)

namespace_input = st.sidebar.text_input(
    "Namespace Target",
    value="assignment-namespace"
)

selected_model = st.sidebar.selectbox(
    "LLM Model",
    [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Document (PDF, DOCX, TXT, CSV)",
    type=["pdf", "docx", "txt", "csv"]
)

# 4. PROCESS AND INDEX DOCUMENT

if st.sidebar.button(
    "Process & Index Document",
    use_container_width=True
):

    if not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"):
        st.error("API keys missing in your .env file.")

    elif not uploaded_file:
        st.error("Please upload a document first.")

    else:
        try:
            with st.spinner("Processing document and creating embeddings..."):

                clean_filename = re.sub(
                    r"[^a-zA-Z0-9_-]",
                    "_",
                    uploaded_file.name
                )

                active_namespace = f"file-{clean_filename}"

                chunks = process_file(
                    uploaded_file,
                    chunk_size,
                    chunk_overlap
                )

                if not chunks:
                    raise ValueError(
                        "No readable text was extracted from this document."
                    )

                build_vectorstore(
                    chunks,
                    index_name,
                    active_namespace
                )

                rag_chain, vectorstore = setup_rag_chain(
                    index_name,
                    selected_model,
                    top_k,
                    active_namespace
                )

                st.session_state.rag_chain = rag_chain
                st.session_state.vectorstore = vectorstore
                st.session_state.all_chunks = chunks
                st.session_state.file_name = uploaded_file.name
                st.session_state.active_namespace = active_namespace

                # IMPORTANT:
                # Do not clear st.session_state.messages here.
                # Existing conversation history is preserved.

            st.success(
                f"Document processed successfully! Total chunks: {len(chunks)}"
            )

        except Exception as e:
            st.error(f"Document processing failed: {e}")

# 5. DISPLAY COMPLETE CHAT HISTORY

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. CHAT INPUT AND RESPONSE


user_query = st.chat_input("Ask something about your document...")

if user_query:

    clean_query = user_query.strip().lower().rstrip("!.,?")

    is_greeting = clean_query in [
        "hi", "hello", "hey", "salam", "aoa",
        "good morning", "good afternoon", "good evening"
    ]

    # Capture previous history BEFORE adding current question.
    chat_history = [
        HumanMessage(content=msg["content"])
        if msg["role"] == "user"
        else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]

    # Save the current question immediately.
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    # Show the current question.
    with st.chat_message("user"):
        st.markdown(user_query)

    answer = ""

    # 7. GENERATE THE ANSWER    

    with st.chat_message("assistant"):

        try:
            if is_greeting:

                if st.session_state.rag_chain:
                    answer = (
                        f"Hello! I am ready to analyze your "
                        f"**{st.session_state.file_name}** file. "
                        "What would you like to know?"
                    )
                else:
                    answer = (
                        "Hello! Upload a document and click "
                        "'Process & Index Document' to get started."
                    )

                st.markdown(answer)

            elif not st.session_state.rag_chain:

                answer = (
                    "Please upload and index a document "
                    "using the sidebar first."
                )

                st.warning(answer)

            else:
                page_match = re.search(
                    r"\bpage\s*(?:number\s*)?(\d+)\b",
                    clean_query
                )

                page_handled = False

                with st.spinner("Thinking..."):

                    # 8. PAGE-SPECIFIC QUESTIONS

                    if page_match:
                        page_handled = True
                        requested_page = int(page_match.group(1))

                        file_extension = os.path.splitext(
                            st.session_state.file_name
                        )[1].lower()

                        all_chunks = st.session_state.all_chunks

                        if file_extension == ".docx":

                            answer = (
                                f"I cannot reliably identify Word Page "
                                f"{requested_page} because the current "
                                "DOCX loader does not preserve page-by-page "
                                "layout.\n\n"
                                "**Please convert your Word document to PDF, "
                                "upload the PDF, and process it again.**"
                            )

                            st.warning(answer)

                        else:
                            matching_chunks = []

                            for chunk in all_chunks:
                                page_value = chunk.metadata.get("page")

                                try:
                                    chunk_page = int(page_value)
                                except (TypeError, ValueError):
                                    continue

                                if chunk_page == requested_page:
                                    matching_chunks.append(chunk)

                            if matching_chunks:

                                context_str = "\n\n".join(
                                    chunk.page_content
                                    for chunk in matching_chunks
                                )

                                llm = ChatOpenAI(
                                    api_key=os.getenv("GROQ_API_KEY"),
                                    base_url="https://api.groq.com/openai/v1",
                                    model=selected_model,
                                    temperature=0.2
                                )

                                direct_prompt = ChatPromptTemplate.from_messages([
                                    (
                                        "system",
                                        (
                                            "Answer using ONLY the supplied "
                                            "page context. Do not invent facts. "
                                            "If the answer is absent, reply "
                                            "exactly: The answer is not "
                                            "available in the provided document.\n\n"
                                            "Page context:\n{context}"
                                        )
                                    ),
                                    MessagesPlaceholder("chat_history"),
                                    ("human", "{input}")
                                ])

                                page_chain = direct_prompt | llm

                                result = page_chain.invoke({
                                    "context": context_str,
                                    "chat_history": chat_history,
                                    "input": user_query
                                })

                                answer = result.content
                                st.markdown(answer)

                            else:
                                available_pages = sorted({
                                    int(chunk.metadata["page"])
                                    for chunk in all_chunks
                                    if str(
                                        chunk.metadata.get("page", "")
                                    ).isdigit()
                                })

                                answer = (
                                    f"Page **{requested_page}** was not found "
                                    f"in the extracted document pages.\n\n"
                                    f"**Available page numbers:** {available_pages}"
                                )

                                st.warning(answer)

                    # 9. NORMAL RAG QUESTIONS

                    if not page_handled:

                        response = st.session_state.rag_chain.invoke({
                            "input": user_query,
                            "chat_history": chat_history
                        })

                        answer = response["answer"]
                        st.markdown(answer)

                        # 10. SOURCE ATTRIBUTION

                        if (
                            response.get("context")
                            and "The answer is not available in the provided document."
                            not in answer
                        ):

                            with st.expander(
                                "📄 Source Attribution & Metadata"
                            ):

                                active_ns = st.session_state.active_namespace

                                results = (
                                    st.session_state.vectorstore
                                    .similarity_search_with_score(
                                        user_query,
                                        k=top_k,
                                        namespace=active_ns
                                    )
                                )

                                for idx, (doc, score) in enumerate(results):

                                    st.markdown(
                                        f"**Source {idx + 1}:** "
                                        f"`{doc.metadata.get('source', 'Document')}` "
                                        f"| **Page:** "
                                        f"`{doc.metadata.get('page', 'N/A')}` "
                                        f"| **Chunk ID:** "
                                        f"`{doc.metadata.get('chunk_id', 'N/A')}` "
                                        f"| **Score:** `{score:.4f}`"
                                    )

                                    st.caption(doc.page_content[:200] + "...")
                                    st.markdown("---")

        except Exception as e:
            answer = f"Sorry, an error occurred: {e}"
            st.error(answer)

    
    # 11. SAVE ASSISTANT ANSWER
    

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Rerun displays the complete saved conversation.
    st.rerun()
