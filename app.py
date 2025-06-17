from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
import requests

app = Flask(__name__)

# Sample metadata database (you can replace this with your actual database)
METADATA_DB = {
    "climate": [
        {
            "source": "NASA Climate Data",
            "description": "Global temperature and climate change data",
            "last_updated": "2024-03-15",
            "format": "CSV",
            "size": "2.5GB",
            "url": "https://climate.nasa.gov/data"
        },
        {
            "source": "NOAA Weather Database",
            "description": "Historical weather patterns and forecasts",
            "last_updated": "2024-03-14",
            "format": "JSON",
            "size": "1.8GB",
            "url": "https://www.noaa.gov/weather"
        }
    ],
    "health": [
        {
            "source": "WHO Health Statistics",
            "description": "Global health indicators and statistics",
            "last_updated": "2024-03-13",
            "format": "Excel",
            "size": "950MB",
            "url": "https://www.who.int/data"
        }
    ]
}

SUBSCRIPTIONS = []

@app.route('/')
def index():
    return render_template('index.html')

def search_zenodo(query, max_results=5):
    """Search Zenodo for datasets/papers matching the query."""
    url = "https://zenodo.org/api/records"
    params = {
        "q": query,
        "size": max_results,
        "sort": "mostrecent"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("hits", {}).get("hits", [])
        results = []
        for item in items:
            metadata = item.get("metadata", {})
            results.append({
                "source": metadata.get("title", "Unknown Title"),
                "description": metadata.get("description", "No description."),
                "last_updated": item.get("updated", "N/A")[:10],
                "format": ", ".join(metadata.get("resource_type", {}).values()) if metadata.get("resource_type") else "N/A",
                "size": str(item.get("files", [{}])[0].get("size", "N/A")),
                "url": item.get("links", {}).get("html", "https://zenodo.org")
            })
        return results
    except Exception as e:
        print(f"Zenodo search error: {e}")
        return []

@app.route('/search', methods=['POST'])
def search():
    print(f"Received form data: {request.form}")  # Debug print
    query = request.form.get('query', '').lower()
    phone = request.form.get('phone', '').strip()
    results = []
    
    # Store subscription if phone number is provided
    if phone:
        SUBSCRIPTIONS.append({'query': query, 'phone': phone})
        print(f"New subscription: {query} -> {phone}")
    
    # Search through local metadata
    for category, sources in METADATA_DB.items():
        if query in category.lower():
            results.extend(sources)
        else:
            for source in sources:
                if query in source['description'].lower() or query in source['source'].lower():
                    results.append(source)
    # Search Zenodo and add results
    zenodo_results = search_zenodo(query)
    print(f"Zenodo results: {zenodo_results}")  # Debug print
    results.extend(zenodo_results)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
