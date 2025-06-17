# warning
import warnings

warnings.filterwarnings("ignore")

import os
from together import Together
import faiss
from sentence_transformers import SentenceTransformer
import requests

"""
This script demonstrates RAG-based metadata extraction for Zenodo search results.
"""
together_api_key = os.environ.get("TOGETHER_API_KEY")


def run_rag_zenodo(zenodo_results, prompt):
    """
    Run RAG system: process Zenodo descriptions, create embeddings, search, and generate answer for each result.
    """
    client = Together(api_key="407980b3daee11d57187bc919693b335417b40bb15d2ebe504ea8d7a4edb972b")
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    embedding_model = embedding_model.to('cpu')

    for item in zenodo_results:
        if item.get('description'):
            doc = {item['source']: item['description']}
            # Create embeddings for the description
            documents = list(doc.values())
            filenames = list(doc.keys())
            embeddings = embedding_model.encode(documents)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            query_embedding = embedding_model.encode([prompt])
            faiss.normalize_L2(query_embedding)
            scores, indices = index.search(query_embedding, min(1, len(documents)))
            context = documents[indices[0][0]]
            llm_prompt = f"""Answer the question based on the provided context document.

Context:
{context}

Question: {prompt}

Instructions:
- Answer based only on the information in the context
- Answer should have at least three variables in the metadata, and mention the time (year) and location of the data
- If possible, inlcude the sample size of the data and describe what a sample is
- If the context doesn't contain enough information, say so
- Start with Author and year of publication
- Add brackets to the document name

Answer:"""
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": llm_prompt}],
                    max_tokens=500,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
                item['rag_metadata'] = answer
                print(f"RAG answer for {item['source']}: {answer}")
            except Exception as e:
                item['rag_metadata'] = f"Error generating answer: {str(e)}"
                print(f"Error for {item['source']}: {e}")
    return zenodo_results

if __name__ == "__main__":
    # Example usage: search Zenodo and run RAG
    def search_zenodo(query, max_results=3):
        url = "https://zenodo.org/api/records"
        params = {"q": query, "size": max_results, "sort": "mostrecent"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("hits", {}).get("hits", [])
        results = []
        for item in items:
            metadata = item.get("metadata", {})
            files = item.get("files", [])
            size = str(files[0].get("size", "N/A")) if files else "N/A"
            results.append({
                "source": metadata.get("title", "Unknown Title"),
                "description": metadata.get("description", "No description."),
                "last_updated": item.get("updated", "N/A")[:10],
                "format": ", ".join(metadata.get("resource_type", {}).values()) if metadata.get("resource_type") else "N/A",
                "size": size,
                "url": item.get("links", {}).get("html", "https://zenodo.org")
            })
        return results

    query = "satellite"
    zenodo_results = search_zenodo(query)
    run_rag_zenodo(zenodo_results, query) 