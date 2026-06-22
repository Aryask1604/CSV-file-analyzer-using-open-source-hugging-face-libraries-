"""
CSV File Analyzer using Open-Source Hugging Face Libraries
==========================================================

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask
natural-language questions about the contents of a CSV file.

Pipeline:
    1. Load CSV rows as documents (LangChain CSVLoader)
    2. Split into chunks (CharacterTextSplitter)
    3. Embed chunks with a Sentence-Transformers model
    4. Index embeddings in a FAISS vector store for similarity search
    5. Answer questions with Mistral-7B-Instruct via a
       ConversationalRetrievalChain that keeps chat history

Author: Arya Khamkar
Note: Reconstructed and cleaned up from the original project report.
"""

import argparse
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
TOP_K = 4  # number of similar chunks to retrieve per query


# ----------------------------------------------------------------------
# 1. Load and chunk the CSV
# ----------------------------------------------------------------------
def load_and_chunk_csv(csv_path: str):
    """Load a CSV file and split its rows into text chunks."""
    print(f"Loading CSV: {csv_path}")
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_docs = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} rows -> {len(chunked_docs)} chunks")
    return chunked_docs


# ----------------------------------------------------------------------
# 2. Build the Mistral-7B-Instruct LLM (4-bit quantized for low memory)
# ----------------------------------------------------------------------
def build_llm():
    """Load Mistral-7B-Instruct with 4-bit quantization and wrap it for LangChain."""
    print(f"Loading LLM: {LLM_MODEL_NAME} (this can take a few minutes the first time)")

    # 4-bit quantization keeps a 7B model loadable on a single consumer GPU.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True,
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)


# ----------------------------------------------------------------------
# 3. Build the FAISS retriever
# ----------------------------------------------------------------------
def build_retriever(chunked_docs):
    """Embed the chunks and build a FAISS similarity-search retriever."""
    print(f"Generating embeddings with: {EMBED_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    db = FAISS.from_documents(chunked_docs, embeddings)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever


# ----------------------------------------------------------------------
# 4. Assemble the conversational QA chain
# ----------------------------------------------------------------------
def build_qa_chain(llm, retriever):
    """Combine the LLM and retriever into a chat-history-aware QA chain."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant answering questions about a CSV dataset.\n"
            "Use only the context below to answer. If the answer is not in the "
            "context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return qa_chain


# ----------------------------------------------------------------------
# 5. Interactive query loop
# ----------------------------------------------------------------------
def chat_loop(qa_chain):
    """Run an interactive prompt that keeps chat history for context."""
    chat_history = []
    print("\nReady. Ask questions about your CSV. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            query = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        print("Answer: " + answer + "\n")
        chat_history.append((query, answer))


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a CSV file using an open-source LLM.")
    parser.add_argument("csv_path", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    chunked_docs = load_and_chunk_csv(args.csv_path)
    retriever = build_retriever(chunked_docs)
    llm = build_llm()
    qa_chain = build_qa_chain(llm, retriever)
    chat_loop(qa_chain)


if __name__ == "__main__":
    main()
