# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import argparse
from together import Together
import textwrap
import json
import os
from datetime import datetime
import re


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    """
    Prompt the LLM to extract metadata from research paper content.
    
    Args:
        prompt (str): The prompt containing the paper content and instructions
        with_linebreak (bool): Whether to wrap the output with line breaks
        
    Returns:
        str: The LLM's response
    """
    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content

    if with_linebreak:
        # Wrap the output
        wrapped_output = textwrap.fill(output, width=50)
        return wrapped_output
    else:
        return output

def extract_json_from_response(response):
    """
    Extract JSON from the LLM's response.
    
    Args:
        response (str): The LLM's response containing JSON
        
    Returns:
        dict: The parsed JSON data
    """
    try:
        # Try to parse the response directly as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
    
    # If all parsing attempts fail, return None
    return None

def split_paper_into_sections(content):
    """
    Split the paper into logical sections for processing.
    
    Args:
        content (str): The full paper content
        
    Returns:
        dict: Dictionary containing different sections of the paper
    """
    sections = {
        'title_abstract': '',
        'methods': '',
        'results': '',
        'discussion': ''
    }
    
    # Extract title and abstract
    title_abstract_match = re.search(r'^(.*?)(?=Introduction)', content, re.DOTALL)
    if title_abstract_match:
        sections['title_abstract'] = title_abstract_match.group(1).strip()
    
    # Extract methods
    methods_match = re.search(r'Methods\n(.*?)(?=Results|Discussion)', content, re.DOTALL)
    if methods_match:
        sections['methods'] = methods_match.group(1).strip()
    
    # Extract results
    results_match = re.search(r'Results\n(.*?)(?=Discussion|Conclusion)', content, re.DOTALL)
    if results_match:
        sections['results'] = results_match.group(1).strip()
    
    # Extract discussion
    discussion_match = re.search(r'Discussion\n(.*?)(?=References|$)', content, re.DOTALL)
    if discussion_match:
        sections['discussion'] = discussion_match.group(1).strip()
    
    return sections

## FUNCTION 2: Extract Metadata from Research Paper
# -------------------------------------------------
def extract_metadata(file_path):
    """
    Extract metadata from a research paper text file using LLM.
    
    Args:
        file_path (str): Path to the research paper text file
        
    Returns:
        dict: Dictionary containing the extracted metadata
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split paper into sections
    sections = split_paper_into_sections(content)
    
    # Extract basic information from title and abstract
    basic_info_prompt = f"""Please analyze this section of a research paper and extract the following information in JSON format:
1. Basic Information:
   - Title
   - Authors
   - Publication dates (received, accepted, published)
   - DOI
   - Keywords

Here is the paper section:
{sections['title_abstract']}

Please provide the information in a structured JSON format. Return ONLY the JSON object, no additional text."""

    # Extract data information from methods and results
    data_info_prompt = f"""Please analyze these sections of a research paper and extract the following information in JSON format:
2. Data Information:
   - Variables measured/collected
   - Spatial extent of the study
   - Temporal extent of the study
   - Sampling resolution (spatial and temporal)
   - Sample size
   - Data collection methods
   - Data sources
   - Any data limitations or constraints

Here are the relevant paper sections:

Methods:
{sections['methods']}

Results:
{sections['results']}

Please provide the information in a structured JSON format. Return ONLY the JSON object, no additional text."""

    # Get LLM responses
    basic_info_response = prompt_llm(basic_info_prompt)
    data_info_response = prompt_llm(data_info_prompt)
    
    # Extract JSON from responses
    basic_info = extract_json_from_response(basic_info_response)
    data_info = extract_json_from_response(data_info_response)
    
    # Process basic info
    if basic_info:
        # Handle publication dates if they're in a nested structure
        if 'publication_dates' in basic_info:
            basic_info['dates'] = basic_info.pop('publication_dates')
    
    # Process data info
    if data_info and 'dataInformation' in data_info:
        # Flatten the nested structure
        data_info = data_info['dataInformation']
        # Convert camelCase keys to snake_case
        data_info = {
            'variables': data_info.get('variablesMeasuredCollected'),
            'spatial_extent': data_info.get('spatialExtent'),
            'temporal_extent': data_info.get('temporalExtent'),
            'sampling_resolution': {
                'spatial': data_info.get('SamplingResolution', {}).get('spatial'),
                'temporal': data_info.get('SamplingResolution', {}).get('temporal')
            },
            'sample_size': data_info.get('sampleSize'),
            'collection_methods': data_info.get('dataCollectionMethods'),
            'data_sources': data_info.get('dataSources'),
            'limitations': data_info.get('dataLimitations')
        }
    
    # Create metadata structure
    metadata = {
        'basic_info': basic_info or {
            'title': None,
            'authors': None,
            'dates': {
                'received': None,
                'accepted': None,
                'published': None
            },
            'doi': None,
            'keywords': None
        },
        'data_info': data_info or {
            'variables': None,
            'spatial_extent': None,
            'temporal_extent': None,
            'sampling_resolution': {
                'spatial': None,
                'temporal': None
            },
            'sample_size': None,
            'collection_methods': None,
            'data_sources': None,
            'limitations': None
        },
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    return metadata

## FUNCTION 3: Save Metadata to JSON
# -------------------------------------------------
def save_metadata(metadata, output_dir='results'):
    """
    Save metadata to a JSON file with timestamp in the filename.
    
    Args:
        metadata (dict): Metadata dictionary to save
        output_dir (str): Directory to save the output file
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{timestamp}.json')
    
    # Save metadata to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    return output_file

## FUNCTION 4: Process File and Extract Metadata
# -------------------------------------------------
def process_file(file_path):
    """
    Process the file and extract metadata.
    
    Args:
        file_path (str): Path to the research paper text file
        
    Returns:
        tuple: (metadata_json, output_file_path)
    """
    try:
        # Extract metadata
        metadata = extract_metadata(file_path)
        
        # Save metadata to JSON file
        output_file = save_metadata(metadata)
        
        # Print the extracted metadata
        print("\nExtracted Metadata:")
        print(json.dumps(metadata, indent=4))
        print(f"\nMetadata saved to: {output_file}")
        
        return metadata, output_file
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser(description="Extract metadata from research papers using LLM")
    parser.add_argument("-k", "--api_key", type=str, required=True,
                      help="Together API key for LLM access")
    parser.add_argument("-f", "--file", type=str, required=True,
                      help="Path to the research paper text file")
    args = parser.parse_args()

    # Initialize the Together client
    client = Together(api_key=args.api_key)

    # Process the file
    process_file(args.file)