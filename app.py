from flask import Flask, render_template, request, jsonify
import os
from together import Together
import fitz  # PyMuPDF for PDF reading: 
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Set your Together API key
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

PDF_DIR = os.path.join(os.path.dirname(__file__), 'data')

print("Together API Key:", TOGETHER_API_KEY)

# Helper to extract text from PDFs
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text()
        text.append(page_text)
    return '\n'.join(text)

# RAG: Embed all PDFs, retrieve top relevant docs for the prompt
def retrieve_relevant_docs(pdf_files, user_prompt, top_k=3):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    documents = []
    filenames = []
    for f in pdf_files:
        text = extract_pdf_text(os.path.join(PDF_DIR, f))
        if text:
            documents.append(text)
            filenames.append(f)
    if not documents:
        return [], []
    
    # Embed all documents
    doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    # Embed the query
    query_embedding = embedding_model.encode([user_prompt], convert_to_numpy=True)
    # Normalize for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embedding)
    # Build FAISS index
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    scores, indices = index.search(query_embedding, min(top_k, len(documents)))
    relevant_docs = [documents[i] for i in indices[0]]
    relevant_filenames = [filenames[i] for i in indices[0]]
    return relevant_docs, relevant_filenames

# Helper to call Together AI with the custom prompt
def get_metadata_from_papers(user_prompt, relevant_docs, relevant_filenames):
    # Compose the system prompt
    system_prompt = (
        "Find the dataset used in the study in a research paper, respectively, and create scientific metadata to allow us to compare the data measurements and observations between the papers using the following format (you can incorporate new tags if you need),\n"
        "make sure the (categorical) attribute of each tag is 3 words max (unless it is the title or authors or journal),\n"
        "one tag should be the definition of a sample that is relevant for the tag sample size\n"
        "two-four additional tags should be: temporal and spatial extent, and temporal and spatial resolution (latter two if it is temporal or spatial data, respectively)\n"
        "only use a specific tag if you can provide an attribute for all data sets\n"
        "for each tag's attribute add the exact line from the paper and the page number as citation\n"
        "if two scientific papers are using the same dataset list the dataset only once but list both studies for the dataset\n"
        "add all columns for the data that are needed to understand the kind of data that needs to be used like different counts\n"
        "||title| sample size| {dataset} | \n|paper 1|\n|paper 2|"
    )
    # Build context from relevant docs
    context = "\n\n".join(f"[{fname}]\n{doc}" for fname, doc in zip(relevant_filenames, relevant_docs))
    # Compose the full prompt
    prompt = f"User prompt: {user_prompt}\n\nContext:\n{context}"
    # Call Together AI API
    client = Together(api_key=TOGETHER_API_KEY)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        print("User prompt:", user_prompt)
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        print("PDF files found:", pdf_files)
        relevant_docs, relevant_filenames = retrieve_relevant_docs(pdf_files, user_prompt, top_k=5)
        print("Relevant docs found:", relevant_filenames)
        try:
            result = get_metadata_from_papers(user_prompt, relevant_docs, relevant_filenames)
            print("Together AI result (first 500 chars):", result[:500])
        except Exception as e:
            result = f"Error: {e}"
            print("Error:", e)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
