# warning
import warnings

warnings.filterwarnings("ignore")

import os
from together import Together
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
from pathlib import Path

"""
Do these steps:
1) Set up a Together API key from https://together.ai/
"""
together_api_key = os.environ.get("TOGETHER_API_KEY")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

def load_documents(data_dict):
    """Process both text strings and PDF files from the data dictionary."""
    documents = []
    filenames = []
    
    print(f"\nAttempting to load {len(data_dict)} documents...")
    for key, content in data_dict.items():
        print(f"\nProcessing: {key}")
        print(f"Content path: {content}")
        if isinstance(content, str):
            # Handle PDF files
            if content.lower().endswith('.pdf'):
                pdf_path = Path(content)
                print(f"PDF path exists: {pdf_path.exists()}")
                print(f"Absolute path: {pdf_path.absolute()}")
                if pdf_path.exists():
                    text = extract_text_from_pdf(pdf_path)
                    if text:
                        documents.append(text)
                        filenames.append(key)
                        print(f"✅ Successfully loaded PDF: {key}")
                    else:
                        print(f"❌ No text extracted from PDF: {key}")
                else:
                    print(f"❌ PDF file not found: {pdf_path}")
            # Handle direct text
            else:
                content = content.strip()
                if content:
                    documents.append(content)
                    filenames.append(key)
                    print(f"✅ Loaded text: {key}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents, filenames


def run_rag(data_dict: dict, prompt: str):
    """
    Run RAG system: process documents, create embeddings, search, and generate answer.

    """

    # Stage 0: Initialize Together AI client for LLM completions
    client = Together(api_key="407980b3daee11d57187bc919693b335417b40bb15d2ebe504ea8d7a4edb972b")#together_api_key)

    # Stage 1: Load sentence transformer model for creating embeddings
    # ------------------------------------------------------------
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )

    # Stage 2: Process documents into Vector Database
    # ------------------------------------------------------------
    documents, filenames = load_documents(data_dict)

    if not documents:
        return "No valid documents found in data dictionary!"

    # Create embeddings for all documents
    print("Creating embeddings...")
    embeddings = embedding_model.encode(documents)

    # Set up FAISS index for similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"✅ RAG system ready with {len(documents)} documents!")

    # Stage 3: Retrieve relevant documents
    # ------------------------------------------------------------
    query_embedding = embedding_model.encode([prompt])
    faiss.normalize_L2(query_embedding)

    # Get top similar documents
    scores, indices = index.search(query_embedding, min(3, len(documents)))

    # Stage 4: Build context from retrieved documents
    # ------------------------------------------------------------
    relevant_docs = []
    context_parts = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            doc_info = {
                "content": documents[idx],
                "filename": filenames[idx],
                "score": float(score),
            }
            relevant_docs.append(doc_info)
            context_parts.append(f"[{doc_info['filename']}]\n{doc_info['content']}")

    if not relevant_docs:
        return "No relevant documents found for the query."

    # Combine context
    context = "\n\n".join(context_parts)

    # Stage 5: Augment by running the LLM to generate an answer
    # ------------------------------------------------------------
    llm_prompt = f"""Answer the question based on the provided context documents.

Context:
{context}

Question: {prompt}

Instructions:
- Answer based only on the information in the context
- Answer should have at least three variables in the metadata, and mention the time (year) and location of the data
- If possible, inlcude the sample size of the data and describe what a sample is
- If the context doesn't contain enough information, say so
- Mention which document(s) you're referencing
- Start with Author and year of publication
- Add brackets to the document name


Answer:"""

    try:
        # Generate answer using Together AI
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content

        # Display source information
        print(f"\n📚 Most relevant source:")
        for doc in relevant_docs:
            print(f"  • {doc['filename']} (similarity: {doc['score']:.3f})")

        # Add source information to the answer
        sources_list = [doc["filename"] for doc in relevant_docs]
        sources_text = sources_list[0]
        full_answer = f"{answer}\n\n📄 Source Used: {sources_text}"

        return full_answer

    except Exception as e:
        return f"Error generating answer: {str(e)}"


if __name__ == "__main__":
    # Example usage with both text and PDFs
    data_dict = {
        "Christensen2020": "data/Christensen2020.pdf",
        "Leckie1995": "data/Leckie1995.pdf",  
        "Speir2014": "data/Speir2014.pdf",  
    }

    question = "Give me metadata that describes observed data on fishig activity!"
    answer = run_rag(data_dict, question)
    print(f"\n🤖 Answer: {answer}\n")
    print("-" * 50)