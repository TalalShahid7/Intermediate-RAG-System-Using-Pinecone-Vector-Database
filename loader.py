import os
import tempfile
import platform
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_file(uploaded_file, chunk_size=1000, chunk_overlap=200):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        docs = []

        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            total_text = "".join([doc.page_content for doc in docs]).strip()
            if len(total_text) < 100:
                images = convert_from_path(tmp_file_path)
                ocr_docs = []
                for page_num, img in enumerate(images):
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        ocr_docs.append(Document(
                            page_content=ocr_text,
                            metadata={"source": uploaded_file.name, "page": page_num + 1}
                        ))
                docs = ocr_docs
            else:
                for idx, doc in enumerate(docs):
                    doc.metadata["source"] = uploaded_file.name
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = idx + 1

        elif file_extension == ".docx":
            docs = Docx2txtLoader(tmp_file_path).load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = "N/A"

        elif file_extension == ".txt":
            docs = TextLoader(tmp_file_path, encoding="utf-8").load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = "N/A"

        elif file_extension == ".csv":
            docs = CSVLoader(tmp_file_path).load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["page"] = "N/A"

        else:
            raise ValueError(f"Unsupported format: {file_extension}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        final_chunks = text_splitter.split_documents(docs)

        for idx, chunk in enumerate(final_chunks):
            chunk.metadata["chunk_id"] = f"{uploaded_file.name}_chunk_{idx+1}"
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = "N/A"

        return final_chunks

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)