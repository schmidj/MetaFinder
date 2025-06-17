from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    results = []
    
    # Search through metadata
    for category, sources in METADATA_DB.items():
        if query in category.lower():
            results.extend(sources)
        else:
            for source in sources:
                if query in source['description'].lower() or query in source['source'].lower():
                    results.append(source)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
