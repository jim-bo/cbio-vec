import gradio as gr
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Create the pdfs directory if it doesn't exist
if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

def get_pdf_files():
    """Gets the list of PDF files from the 'pdfs' directory."""
    return [f for f in os.listdir("pdfs") if f.endswith(".pdf")]

def index_pdfs():
    """Indexes the PDF files in the 'pdfs' directory."""
    pdf_files = get_pdf_files()
    if not pdf_files:
        return "No PDF files found in the 'pdfs' directory."

    success_files = []
    failed_files = []
    
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join("pdfs", pdf_file)
            if os.path.getsize(file_path) == 0:
                failed_files.append(f"{pdf_file} (file is empty)")
                continue

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(texts, embeddings, persist_directory="./db")
            db.persist()
            success_files.append(pdf_file)
        except Exception as e:
            failed_files.append(f"{pdf_file} (Error: {e})")

    status = ""
    if success_files:
        status += f"Successfully indexed: {', '.join(success_files)}\n"
    if failed_files:
        status += f"Failed to index: {', '.join(failed_files)}"
        
    return status if status else "No files were processed."

def search(query):
    """Searches the indexed PDFs for the given query."""
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="./db", embedding_function=embeddings)
    docs = db.similarity_search(query)
    results = ""
    for doc in docs:
        results += f"Source: {doc.metadata['source']}\n"
        results += f"Content: {doc.page_content}\n\n"
    return results

with gr.Blocks() as demo:
    gr.Markdown("# Simple Semantic Search App")
    with gr.Tab("Index PDFs"):
        pdf_files_display = gr.Textbox(label="Available PDF Files", interactive=False, value="\n".join(get_pdf_files()))
        index_button = gr.Button("Index PDFs")
        index_status = gr.Textbox(label="Indexing Status", interactive=False)
        index_button.click(index_pdfs, inputs=None, outputs=index_status)
    with gr.Tab("Search"):
        search_query = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search")
        search_results = gr.Textbox(label="Search Results", interactive=False)
        search_button.click(search, inputs=search_query, outputs=search_results)

if __name__ == "__main__":
    demo.launch()
