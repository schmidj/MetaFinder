import gradio as gr
import json
import os
from together import Together
import argparse
from datetime import datetime
import re
import textwrap

# Global variables
client = None
current_metadata = None

# Import functions after client is defined
from minimal import (
    extract_metadata,
    save_metadata,
    request_additional_variable,
    remove_variable,
    prompt_llm,
    extract_json_from_response
)

def initialize_client(api_key):
    """Initialize the Together client with the provided API key."""
    global client
    try:
        client = Together(api_key=api_key)
        # Update the client in the minimal module
        import minimal
        minimal.client = client
        return "API key set successfully!"
    except Exception as e:
        return f"Error setting API key: {str(e)}"

def process_paper(paper_content):
    """Process the paper content and extract metadata."""
    global client, current_metadata
    
    try:
        # Extract metadata directly from content
        current_metadata = extract_metadata(paper_content)
        return json.dumps(current_metadata, indent=4)
    except Exception as e:
        return f"Error processing paper: {str(e)}"

def query_metadata(query):
    """Query the current metadata."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    try:
        parts = query.split('.')
        current = current_metadata
        
        for part in parts:
            current = current[part]
        
        if isinstance(current, dict):
            return json.dumps(current, indent=4)
        else:
            return str(current)
    except KeyError:
        return f"Error: Could not find '{query}' in the metadata"
    except Exception as e:
        return f"Error: {str(e)}"

def add_variable(variable_name):
    """Request additional information about a variable."""
    global current_metadata, client
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    if not client:
        return "Client not initialized. Please process a paper first."
    
    try:
        return request_additional_variable(current_metadata, variable_name)
    except Exception as e:
        return f"Error adding variable: {str(e)}"

def request_additional_variable(metadata, variable_name):
    """
    Request additional information about a specific variable from the LLM.
    
    Args:
        metadata (dict): The current metadata dictionary
        variable_name (str): The name of the variable to get more information about
    """
    global client
    
    prompt = f"""Please analyze the research paper and provide detailed information about the following variable: {variable_name}
    
    Please provide the information in JSON format with the following structure:
    {{
        "definition": "Clear definition of the variable",
        "measurement_method": "How it was measured/collected",
        "units": "Units of measurement if applicable",
        "importance": "Why this variable is important in the study"
    }}
    
    Return ONLY the JSON object, no additional text."""

    response = prompt_llm(prompt)
    additional_info = extract_json_from_response(response)
    
    if additional_info:
        # Add the new information to the metadata
        if 'additional_variables' not in metadata:
            metadata['additional_variables'] = {}
        metadata['additional_variables'][variable_name] = additional_info
        return json.dumps(additional_info, indent=4)
    else:
        return f"Could not extract additional information about {variable_name}"

def remove_variable_from_metadata(variable_path):
    """Remove a variable from the metadata."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    try:
        remove_variable(current_metadata, variable_path)
        return f"Removed {variable_path} from metadata"
    except Exception as e:
        return f"Error removing variable: {str(e)}"

def save_current_metadata():
    """Save the current metadata to a file."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available to save."
    
    try:
        output_file = save_metadata(current_metadata)
        return f"Metadata saved to: {output_file}"
    except Exception as e:
        return f"Error saving metadata: {str(e)}"

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

def extract_metadata(content):
    """
    Extract metadata from research paper content using LLM.
    
    Args:
        content (str): The research paper content
        
    Returns:
        dict: Dictionary containing the extracted metadata
    """
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

def prompt_llm(prompt, with_linebreak=False):
    """
    Prompt the LLM to extract metadata from research paper content.
    
    Args:
        prompt (str): The prompt containing the paper content and instructions
        with_linebreak (bool): Whether to wrap the output with line breaks
        
    Returns:
        str: The LLM's response
    """
    global client
    
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

# Create the Gradio interface
with gr.Blocks(title="Research Paper Metadata Extractor") as demo:
    gr.Markdown("# Research Paper Metadata Extractor")
    
    with gr.Tab("Process Paper"):
        with gr.Row():
            with gr.Column():
                paper_content = gr.Textbox(
                    label="Paper Content",
                    placeholder="Paste your research paper content here...",
                    lines=20
                )
                process_btn = gr.Button("Process Paper")
            
            with gr.Column():
                metadata_output = gr.Textbox(
                    label="Extracted Metadata",
                    lines=20,
                    interactive=False
                )
    
    with gr.Tab("Query Metadata"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter query (e.g., 'basic_info.title', 'data_info.variables')"
                )
                query_btn = gr.Button("Query")
                
                add_var_input = gr.Textbox(
                    label="Add Variable",
                    placeholder="Enter variable name to get more information"
                )
                add_var_btn = gr.Button("Add Variable")
                
                remove_var_input = gr.Textbox(
                    label="Remove Variable",
                    placeholder="Enter variable path to remove (e.g., 'data_info.variables')"
                )
                remove_var_btn = gr.Button("Remove Variable")
                
                save_btn = gr.Button("Save Metadata")
            
            with gr.Column():
                query_output = gr.Textbox(
                    label="Query Result",
                    lines=20,
                    interactive=False
                )
    
    # Set up event handlers
    process_btn.click(
        process_paper,
        inputs=[paper_content],
        outputs=metadata_output
    )
    
    query_btn.click(
        query_metadata,
        inputs=query_input,
        outputs=query_output
    )
    
    add_var_btn.click(
        add_variable,
        inputs=add_var_input,
        outputs=query_output
    )
    
    remove_var_btn.click(
        remove_variable_from_metadata,
        inputs=remove_var_input,
        outputs=query_output
    )
    
    save_btn.click(
        save_current_metadata,
        outputs=query_output
    )

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract metadata from research papers using LLM")
    parser.add_argument("-k", "--api_key", type=str, required=True,
                      help="Together API key for LLM access")
    args = parser.parse_args()

    # Initialize the Together client
    client = Together(api_key=args.api_key)
    
    # Launch the Gradio interface
    demo.launch()