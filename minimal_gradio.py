import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import json
import os
import datetime
import re
import textwrap
from together import Together
import argparse

# Global variables
client = None
current_metadata = None

def prompt_llm(prompt, with_linebreak=False):
    """Prompt the LLM to extract metadata from research paper content."""
    global client
    
    if not client:
        raise ValueError("Client not initialized. Please provide an API key first.")
    
    # Make the API call
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured metadata from research papers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    # Extract the response text
    response_text = response.choices[0].message.content
    
    # Optionally wrap the text
    if with_linebreak:
        response_text = textwrap.fill(response_text, width=80)
    
    return response_text

def extract_json_from_response(response):
    """Extract JSON from the LLM's response."""
    # Find JSON-like content between triple backticks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If no JSON found between backticks, try to find any JSON-like structure
    try:
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return None

def split_paper_into_sections(content):
    """Split the paper into logical sections."""
    sections = {
        'title_abstract': '',
        'methods': '',
        'results': '',
        'discussion': ''
    }
    
    # Extract title and abstract
    title_abstract_match = re.search(r'(.*?)(?=\n\s*\d+\.\s*Introduction)', content, re.DOTALL)
    if title_abstract_match:
        sections['title_abstract'] = title_abstract_match.group(1).strip()
    
    # Extract methods
    methods_match = re.search(r'\d+\.\s*Materials?\s+and\s+Methods?(.*?)(?=\d+\.\s*Results)', content, re.DOTALL)
    if methods_match:
        sections['methods'] = methods_match.group(1).strip()
    
    # Extract results
    results_match = re.search(r'\d+\.\s*Results(.*?)(?=\d+\.\s*Discussion)', content, re.DOTALL)
    if results_match:
        sections['results'] = results_match.group(1).strip()
    
    # Extract discussion
    discussion_match = re.search(r'\d+\.\s*Discussion(.*?)(?=\d+\.\s*Conclusion|\Z)', content, re.DOTALL)
    if discussion_match:
        sections['discussion'] = discussion_match.group(1).strip()
    
    return sections

def extract_metadata(content):
    """Extract metadata from research paper content using LLM."""
    # Split the paper into sections
    sections = split_paper_into_sections(content)
    
    # Extract basic information
    basic_info_prompt = f"""Extract basic information from this research paper section. Return the information in JSON format with the following structure:
{{
    "title": "paper title",
    "authors": ["author1", "author2", ...],
    "institutions": ["institution1", "institution2", ...],
    "keywords": ["keyword1", "keyword2", ...],
    "abstract": "paper abstract"
}}

Paper section:
{sections['title_abstract']}"""
    
    basic_info_response = prompt_llm(basic_info_prompt)
    basic_info = extract_json_from_response(basic_info_response)
    
    # Extract data information
    data_info_prompt = f"""Extract information about variables, data sources, and analysis methods from these paper sections. Return the information in JSON format with the following structure:
{{
    "variables": {{
        "variable1": {{
            "description": "description of variable1",
            "type": "type of variable1",
            "source": "source of variable1"
        }},
        "variable2": {{
            "description": "description of variable2",
            "type": "type of variable2",
            "source": "source of variable2"
        }}
    }},
    "data_sources": ["source1", "source2", ...],
    "analysis_methods": ["method1", "method2", ...]
}}

Methods section:
{sections['methods']}

Results section:
{sections['results']}"""
    
    data_info_response = prompt_llm(data_info_prompt)
    data_info = extract_json_from_response(data_info_response)
    
    # Combine the extracted information
    metadata = {
        "basic_info": basic_info,
        "data_info": data_info,
        "extraction_timestamp": datetime.datetime.now().isoformat()
    }
    
    return metadata

def save_metadata(metadata, output_dir='results'):
    """Save metadata to a JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metadata_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save metadata to file
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return filepath

def request_additional_variable(metadata, variable_name):
    """Request additional information about a variable from the LLM."""
    global client
    
    if not client:
        raise ValueError("Client not initialized. Please provide an API key first.")
    
    # Construct prompt for the LLM
    prompt = f"""Please provide additional information about the variable '{variable_name}' from the research paper. 
Return the information in JSON format with the following structure:
{{
    "description": "detailed description of the variable",
    "type": "data type of the variable",
    "source": "source of the variable",
    "additional_info": {{
        "key1": "value1",
        "key2": "value2"
    }}
}}"""
    
    # Get response from LLM
    response = prompt_llm(prompt)
    
    # Extract JSON from response
    additional_info = extract_json_from_response(response)
    
    if additional_info:
        # Add the new information to the metadata
        if "variables" not in metadata["data_info"]:
            metadata["data_info"]["variables"] = {}
        
        metadata["data_info"]["variables"][variable_name] = additional_info
        return additional_info
    
    return None

def remove_variable(metadata, variable_path):
    """Remove a variable from the metadata."""
    parts = variable_path.split('.')
    current = metadata
    
    # Navigate to the parent of the variable to remove
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f"Path '{variable_path}' not found in metadata")
        current = current[part]
    
    # Remove the variable
    variable_to_remove = parts[-1]
    if variable_to_remove not in current:
        raise KeyError(f"Variable '{variable_to_remove}' not found at path '{variable_path}'")
    
    current.pop(variable_to_remove)

def find_variable_in_metadata(metadata, target_var):
    """Helper function to find a variable in the metadata structure."""
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key == target_var:
                return metadata, key
            result = find_variable_in_metadata(value, target_var)
            if result:
                return result
    return None

def remove_variable_from_metadata(variable_path):
    """Remove a variable from the metadata."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    try:
        # If the path contains dots, treat it as a full path
        if '.' in variable_path:
            parts = variable_path.split('.')
            current = current_metadata
            for part in parts[:-1]:
                if part not in current:
                    return f"Path '{variable_path}' not found in metadata"
                current = current[part]
            
            variable_to_remove = parts[-1]
            if variable_to_remove not in current:
                return f"Variable '{variable_to_remove}' not found at path '{variable_path}'"
            
            removed_value = current.pop(variable_to_remove)
            return f"Removed variable '{variable_path}' with value: {removed_value}"
        else:
            # Search for the variable in the entire metadata structure
            result = find_variable_in_metadata(current_metadata, variable_path)
            if result:
                parent_dict, var_name = result
                removed_value = parent_dict.pop(var_name)
                return f"Removed variable '{variable_path}' with value: {removed_value}"
            else:
                return f"Variable '{variable_path}' not found in metadata"
    except Exception as e:
        return f"Error removing variable: {str(e)}"

def initialize_client(api_key):
    """Initialize the Together client with the provided API key."""
    global client
    client = Together(api_key=api_key)
    return client

def process_paper(paper_content):
    """Process the paper content and extract metadata."""
    global current_metadata
    
    try:
        # Extract metadata from the content
        metadata = extract_metadata(paper_content)
        
        # Update the current metadata
        current_metadata = metadata
        
        # Save the metadata
        save_metadata(metadata)
        
        return json.dumps(metadata, indent=4)
    except Exception as e:
        return f"Error processing paper: {str(e)}"

def query_metadata(query):
    """Query the current metadata."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    try:
        if not query:
            return json.dumps(current_metadata, indent=4)
        
        # Split the query into parts
        parts = query.split('.')
        
        # Navigate through the metadata
        result = current_metadata
        for part in parts:
            if part not in result:
                return f"Path '{query}' not found in metadata"
            result = result[part]
        
        return json.dumps(result, indent=4)
    except Exception as e:
        return f"Error querying metadata: {str(e)}"

def add_variable(variable_name):
    """Request additional information about a variable."""
    global current_metadata, client
    
    if not current_metadata:
        return "No metadata available. Please process a paper first."
    
    if not client:
        return "Client not initialized. Please process a paper first."
    
    try:
        additional_info = request_additional_variable(current_metadata, variable_name)
        if additional_info:
            return f"Added information for variable '{variable_name}':\n{json.dumps(additional_info, indent=4)}"
        else:
            return f"Could not extract additional information about {variable_name}"
    except Exception as e:
        return f"Error adding variable: {str(e)}"

def save_current_metadata():
    """Save the current metadata to a file."""
    global current_metadata
    
    if not current_metadata:
        return "No metadata available to save."
    
    try:
        filepath = save_metadata(current_metadata)
        return f"Metadata saved to {filepath}"
    except Exception as e:
        return f"Error saving metadata: {str(e)}"

# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract metadata from research papers using LLM")
parser.add_argument("-k", 
                    "--api_key", 
                    type=str, 
                    default="407980b3daee11d57187bc919693b335417b40bb15d2ebe504ea8d7a4edb972b",
                    help="Together API key for LLM access")
args = parser.parse_args()

# Initialize the client with the API key
initialize_client(args.api_key)

# Create the Gradio interface
with gr.Blocks(title="Research Paper Metadata Extractor") as demo:
    gr.Markdown("# Research Paper Metadata Extractor")
    
    with gr.Tab("Process Paper"):
        with gr.Row():
            with gr.Column():
                paper_content = gr.Textbox(
                    label="Paper Content",
                    placeholder="Paste your research paper content here...",
                    lines=20,
                    value="""Fisheries Research 272 (2024) 106932
Available online 14 January 2024
0165-7836/© 2024 The Authors. Published by Elsevier B.V. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-
nc-nd/4.0/).Estimating angler effort and catch from a winter recreational fishery using a
novel Bayesian methodology to integrate multiple sources of creel
survey data
Caroline M. Tucker a,*, Simone Collier a, Geoffrey Legault b, George E. Morgan a, Derrick K. de
Kerckhove a
a Ontario Ministry of Natural Resources and Forestry, Aquatic Research and Monitoring Section, 2140 East Bank Drive, Peterborough, ON K9L 1Z8, Canada
b Unaffiliated, Toronto, ON, Canada
A R T I C L E I N F O
Handled by: Russell B Millar
Keywords:
Survey design
Fisheries management
Catch rates
Fishing effort
Bayesian statistics
A B S T R A C T
The range of survey methodologies for measuring daily activity, catch and harvest (i.e. creel surveys) of recre-
ational anglers, is increasing with the advent of new technologies and improvements in remote sensing. Indi-
vidual creel survey types frequently give different insights into a fishery due to their unique sources of
methodological bias and coverage, which creates a problem for resource managers since markedly different
estimates of important fishery metrics can result. We demonstrate a joint estimation approach using a Bayesian
statistical framework that can bring together multiple survey types to derive a single estimate for important
metrics. This framework is applied to data collected from a relatively large winter fishery and integrates three
traditional creel methodologies (i.e. roving, access and aerial counts), each with very different sources of bias, to
derive a common estimate of angler effort. Models integrating two survey types are found to be have lower
uncertainty in their estimates. Further, reductions in effort for any one survey type is found to be buffered by the
joint estimation approach, such that resource managers will likely find benefits in using more than one survey
methodology in an integrated fashion to monitor a fishery."""
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

# Launch the interface
if __name__ == "__main__":
    demo.launch()