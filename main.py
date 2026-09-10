import glob
import os
import pathlib
import shutil
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

import pymupdf4llm
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file (if present)
load_dotenv()

# Constants configuration
PDF_DIR_ENV = os.getenv("PDF_DIR", "./pdf_files")
PDF_DIR = pathlib.Path(PDF_DIR_ENV)
MD_DIR = pathlib.Path("./md_files")
CHROMA_DB_DIR = pathlib.Path("./chroma_db_dir")

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")

def convert_pdf_to_md() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PDF_DIR.exists():
        logger.warning(f"PDF folder '{PDF_DIR}' does not exist. Ensure the correct path is set in the .env file")
        return

    logger.info(f"Starting PDF file conversion in: {PDF_DIR}")

    # Use rglob to find .pdf files in subdirectories (case-insensitive)
    pdf_files = [p for p in PDF_DIR.rglob("*") if p.suffix.lower() == '.pdf']
    
    if not pdf_files:
        logger.info(f"No PDF files found in {PDF_DIR}.")
        return

    for pdf_path in pdf_files:
        logger.info(f"Converting: {pdf_path.name}...")
        try:
            md_content = pymupdf4llm.to_markdown(str(pdf_path))
            output_file = MD_DIR / f"{pdf_path.stem}.md"
            output_file.write_text(md_content, encoding="utf-8")
            logger.info(f"Saved {output_file.name}")
        except Exception as e:
            logger.error(f"Error during conversion of {pdf_path}: {e}")

    logger.info("Conversion completed!")

def execute_chunking() -> List[Dict[str, Any]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks: List[Dict[str, Any]] = []
    
    if not MD_DIR.exists():
        logger.error(f"Directory {MD_DIR} does not exist. Run PDF conversion first.")
        return chunks

    md_files = list(MD_DIR.glob("*.md"))
    logger.info(f"Found {len(md_files)} Markdown files. Starting chunking...")

    for file_path in md_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            file_chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(file_chunks):
                structured_document = {
                    "text": chunk,
                    "metadata": {
                        "origin": file_path.name,
                        "block_number": i + 1
                    }
                }
                chunks.append(structured_document)
                logger.info(f"Chunk {i + 1} of {len(file_chunks)}: {structured_document}\n\n")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Chunking completed! Total chunks created: {len(chunks)}")
    return chunks

def save_in_db(chunk_list: List[Dict[str, Any]]) -> Chroma:
    logger.info(f"Starting vectorization of {len(chunk_list)} chunks...")

    if not chunk_list:
        logger.warning("No chunks to save in the DB.")
        return None

    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    texts = [c["text"] for c in chunk_list]
    metadata = [c["metadata"] for c in chunk_list]

    try:
        vector_db = Chroma.from_texts(
            texts=texts,
            metadatas=metadata,
            embedding=get_embeddings(),
            persist_directory=str(CHROMA_DB_DIR)
        )
        logger.info(f"Database created and saved in {CHROMA_DB_DIR}")
        return vector_db
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise

if __name__ == '__main__':
    # Uncomment the following line to enable PDF conversion at startup
    # convert_pdf_to_md()
    chunks_for_db = execute_chunking()
    if chunks_for_db:
        save_in_db(chunks_for_db)
