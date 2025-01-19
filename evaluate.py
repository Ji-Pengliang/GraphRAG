import os
import json
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import scipy.stats
from datetime import datetime
from tqdm import tqdm
import argparse
from nano_graphrag import GraphRAG, QueryParam
import ipdb
import traceback
import asyncio
import base64
from llm import LLMInterface
from pathlib import Path

llm = LLMInterface()
image_dir = Path('/root/code/E-RAG/data_generation/datasets/CMU_500/images/merged')

def modify_jsonl_file(file_path, query_value, idx, result):
    """
    Modifies the last line of a JSONL file to update 'query' and 'idx' fields.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    last_line = json.loads(lines[-1])
    last_line['query'] = query_value
    last_line['idx'] = idx
    last_line['response'] = result
    
    lines[-1] = json.dumps(last_line) + '\n'
    
    with open(file_path, 'w') as file:
        file.writelines(lines)

def compute_ci(values, confidence=0.95):
    """
    Compute the mean and confidence interval (t-distribution) for a list of values.
    Returns a tuple (mean, margin_of_error).
    """
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, h

def extract_entities_relationships(G):
    """
    Extracts entities, relationships, and chunks from the given graph G.
    Also creates a mapping of normalized node names to node data (name2node).
    """
    entities = []
    relationships = []
    chunks = []
    name2node = {}
    
    name_count = {}   # Tracks count of occurrences for each name
    node_name_map = {}  # Maps each node_id to a unique name
    
    # Create distinct names for each node, in case of duplicates
    for node_id, data in G.nodes(data=True):
        if 'name' not in data:
            data['name'] = 'cluster'
        original_name = data['name']
        
        if original_name not in name_count:
            name_count[original_name] = 0
        else:
            name_count[original_name] += 1
        
        distinct_name = f"{original_name}_{name_count[original_name]}"
        node_name_map[node_id] = distinct_name
        name2node[distinct_name.lower()] = data

    # Process nodes for entities, relationships, chunks
    for node_id, data in G.nodes(data=True):
        entity = {}
        chunk = {}
        
        source_id = f"Source_{node_id}"
        distinct_name = node_name_map[node_id]
        
        # Define the entity
        entity['entity_name'] = distinct_name
        entity['entity_type'] = (data['type'] 
                                 if data.get('type') == 'base' 
                                 else 'cluster')
        entity['position'] = json.dumps(data.get('position', {}))
        entity['source_id'] = source_id
        
        # Create the relationship based on level
        if not data.get('level'):
            entity['description'] = data.get('caption', '')
            entity['image_path'] = data.get('image_path', '')
            entity['timestamp'] = data.get('timestamp', '')
            content = "\n".join(f"{k}: {v}" for k, v in entity.items())
        else:
            entity['description'] = data.get('summary', '')
            
            members = data.get('members', [])
            contents = []
            for src_id in members:
                relationship = {}
                source_name = node_name_map[src_id]
                
                relationship['src_id'] = source_name
                relationship['tgt_id'] = distinct_name
                relationship['description'] = f"{source_name} is part of {distinct_name}"
                relationship['weights'] = 1.0
                relationship['keywords'] = 'belonging to'
                relationship['source_id'] = source_id
                
                relationships.append(relationship)
                contents.append(source_name)
            
            content = "\n".join(f"{k}: {v}" for k, v in entity.items())
            content += f"\n{distinct_name} contains these areas: {', '.join(contents)}"
        
        chunk['source_id'] = source_id
        chunk['content'] = content

        entities.append(entity)
        chunks.append(chunk)
    
    return entities, relationships, chunks, name2node

def extract_locations_from_result(results, name2node):
    """
    Extracts locations from the GraphRAG result output lines.
    Returns a list of retrieved location dictionaries and a list of queries.
    """
    all_rets = []
    all_queries = []
    all_responses = []
    
    for result in results:
        data = result['results']
        response = result['response']
        ret = []
        for node_name in data:
            # Clean up the node name
            node_name = node_name.strip("\"").lower()
            if node_name not in name2node:
                continue
            node = name2node[node_name]
            if node['type'] != 'base':
                continue
            d_transformed = {
                'name': node['name'],
                'description': node.get('caption', ''),
                'position': node['position'],
                'image_path': node['image_path']
            }
            ret.append(d_transformed)

        all_rets.append(ret)
        all_queries.append(result['query'])
        all_responses.append(response)

    return all_rets, all_queries, all_responses

def compute_semantic_relativity(query: str, retrieved_nodes: List[Dict]) -> Dict[str, float]:
    """Compute semantic relativity using GPT-4 to score relevance"""
    system_prompt = """You are an expert evaluator. Rate the relevance of the location given a user's query on a scale of 1-5, where:
    1 = Completely irrelevant
    2 = Somewhat irrelevant
    3 = Moderately relevant
    4 = Very relevant
    5 = Perfectly relevant
    
    Consider both the visual content and the hierarchical context in your evaluation.
    Return only the numerical score without explanation."""

    def get_score_for_node(node, attempt_num):
        try:
            # Get image path
            image_path = node.get('image_path', '')
            if not image_path:
                print(f"Attempt {attempt_num}: No image path available")
                return None

            # Use consistent path resolution
            image_name = Path(image_path).name
            absolute_image_path = image_dir / image_name  # Use image_dir directly
            
            print(f"Looking for image at: {absolute_image_path}")

            if not absolute_image_path.exists():
                print(f"Attempt {attempt_num}: Image not found at {absolute_image_path}")
                return None

            # Read and encode image
            try:
                with open(absolute_image_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode()
            except Exception as e:
                print(f"Attempt {attempt_num}: Error reading image file: {str(e)}")
                return None

            # Format location with image and hierarchical context
            location_text = (
                f"Location Name: {node.get('name', 'Unnamed')}\n"
                f"Parent Areas: {node.get('parent_areas', [])}\n"
                f"Visual Content: [Image Attached]\n"
                f"Description: {node.get('caption', 'No description')}"
            )

            prompt = f"""Query: {query}

Location Information:
{location_text}

Rate the relevance of this location on a scale of 1-5:"""

            # Get score from GPT-4 with image
            func = llm.generate_response(
                prompt, 
                system_prompt,
                image_base64=encoded_image
            )
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(func)
            try:
                score = float(response.strip())
                if 1 <= score <= 5:
                    print(f"Attempt {attempt_num}: Score = {score}")
                    return score
                else:
                    print(f"Attempt {attempt_num}: Invalid score range: {score}")
            except ValueError:
                print(f"Attempt {attempt_num}: Invalid response format: {response}")
            
        except Exception as e:
            print(f"Attempt {attempt_num}: Error: {str(e)}")
            traceback.print_exc()
        return None

    scores_per_node = []
    for i, node in enumerate(retrieved_nodes[:5]):  # Only evaluate top 5
        node_scores = []
        num_attempts = 5
        
        for attempt in range(num_attempts):
            score = get_score_for_node(node, attempt + 1)
            if score is not None:
                node_scores.append(score)
                print(f"Node {i+1}, Attempt {attempt + 1}: Score = {score}")
        
        if node_scores:
            # Average the scores for this node
            avg_score = sum(node_scores) / len(node_scores)
            scores_per_node.append(avg_score)
            print(f"Node {i+1} Average Score: {avg_score:.4f}")

    if not scores_per_node:
        return {'top1': 0.0, 'top5': 0.0}

    # Normalize scores to 0-1 range
    normalized_scores = [float(score - 1) / 4 for score in scores_per_node]
    
    return {
        'top1': normalized_scores[0] if normalized_scores else 0.0,
        'top5': sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    }

def compute_haversine_distance(loc1: Dict, loc2: Dict) -> float:
    """
    Compute haversine distance between two locations in meters
    
    Args:
        loc1, loc2: Dictionaries containing latitude/longitude coordinates
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    # Extract coordinates
    lat1 = loc1.get('y', loc1.get('latitude', 0))
    lon1 = loc1.get('x', loc1.get('longitude', 0))
    lat2 = loc2.get('latitude', 0)
    lon2 = loc2.get('longitude', 0)
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def compute_spatial_relativity(query_location: Dict, retrieved_nodes: List[Dict]) -> float:
    """
    Compute spatial relativity using exponential decay without a hard threshold
    """
    if not retrieved_nodes:
        return 0.0

    distances = []
    decay_factor = 0.005  # Controls how quickly the score decays with distance
                        # Smaller value = slower decay, larger value = faster decay

    for node in retrieved_nodes[:5]:
        try:
            # Get node location
            node_loc = {
                'latitude': node['position'].get('y', node['position'].get('latitude')),
                'longitude': node['position'].get('x', node['position'].get('longitude'))
            }
            
            # Compute distance
            distance = compute_haversine_distance(query_location, node_loc)
            
            # Use exponential decay: score = e^(-decay_factor * distance)
            score = np.exp(-decay_factor * distance)
            distances.append(score)
            
            # Debug output
            print(f"Distance: {distance:.1f}m, Score: {score:.4f}")
            
        except Exception as e:
            print(f"Error computing distance: {str(e)}")
            distances.append(0.0)

    return sum(distances) / len(distances) if distances else 0.0

def evaluate_generated_response(query: str, generated_response: str, retrieved_nodes: List[Dict]) -> Dict:
    try:
        # Parse the generated response as JSON
        try:
            response_data = json.loads(generated_response.strip('`json\n'))  # Remove markdown formatting
            generated_node = {
                'name': response_data.get('name', ''),
                'caption': response_data.get('caption', ''),
                'position': response_data.get('position', {}),
                'image_path': response_data.get('image_path', ''),
                'parent_areas': response_data.get('parent_areas', []),
                'reasons': response_data.get('reasons', '')
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Error parsing generated response: {str(e)}")
            print(f"Generated response: {generated_response}")
            return None

        print("\n=== Evaluating Generated Response ===")
        print(f"Generated Location: {generated_node['name']}")
        
        # Score the generated result using the same image directory
        image_name = Path(generated_node['image_path']).name
        generated_node['image_path'] = str(image_dir / image_name)
        # Get semantic score with multiple attempts
        semantic_scores = []
        num_attempts = 5
        for attempt in range(num_attempts):
            score = compute_semantic_relativity(query, [generated_node])
            if score['top1'] is not None:
                semantic_scores.append(score['top1'])
                print(f"Attempt {attempt + 1} Semantic Score: {score['top1']:.4f}")
        
        # Average the semantic scores
        avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        
        # Get spatial score using retrieved nodes for comparison
        spatial_score = compute_spatial_relativity(
            {'latitude': float(generated_node['position']['y']), 
                'longitude': float(generated_node['position']['x'])}, 
            retrieved_nodes  # Use retrieved nodes instead of [generated_node]
        )
        
        # Calculate combined score
        final_score = avg_semantic_score * spatial_score
        
        generation_evaluation = {
            'semantic_score': avg_semantic_score,
            'spatial_score': spatial_score,
            'combined_score': final_score
        }
        
        print("\n=== Generation Evaluation Results ===")
        print(f"Average Semantic Score ({len(semantic_scores)} attempts): {avg_semantic_score:.4f}")
        print(f"Spatial Score: {spatial_score:.4f}")
        print(f"Combined Score: {final_score:.4f}")
        
        return generation_evaluation
        
    except Exception as e:
        print(f"Error evaluating generated response: {str(e)}")
        traceback.print_exc()
        return None
    
def evaluate_query(query_item, retrieved_nodes, response):
    query = query_item['query']
    agent_location = query_item['location']
    use_history = query_item.get('use_history', False)

    # Compute retrieval metrics
    semantic_scores = compute_semantic_relativity(query, retrieved_nodes)
    spatial_score = compute_spatial_relativity(agent_location, retrieved_nodes)
    generation_scores = evaluate_generated_response(query, response, retrieved_nodes)
    
    # Calculate final semantic-spatial scores
    final_scores = {
        'top1': semantic_scores['top1'] * spatial_score,
        'top5': semantic_scores['top5'] * spatial_score
    }
    
    return {
        'query': query,
        'agent_location': agent_location,
        'use_history': use_history,
        'response': response,
        'retrieved_nodes': retrieved_nodes[:5],
        'metrics': {
            'semantic_relativity': semantic_scores,
            'spatial_relativity': spatial_score,
            'semantic_spatial_score': final_scores,
            'generation_evaluation': generation_scores
        },
        'retrieved_count': len(retrieved_nodes),
        'success': len(retrieved_nodes) > 0
    }
       
def parse_args():
    parser = argparse.ArgumentParser(description="Run GraphRAG pipeline with semantic forest.")
    parser.add_argument(
        "--initialize_kg_from_semantic_forest",
        action="store_true",
        help="If set, will build the GraphRAG knowledge graph from a semantic forest file."
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="If set, will start query from json."
    )
    return parser.parse_args()

def main():
    """
    Main function:
    - Loading the semantic forest graph
    - Transforming to GraphRAG knowledge graph format
    - Executing queries
    - Computing metrics against Google Maps ground truth
    - Saving results
    """
    args = parse_args()

    # Directories and paths (adjust as needed)
    PROJECT_ROOT = "/root/code/E-RAG/Embodied-RAG"
    graph_path = os.path.join(PROJECT_ROOT, "semantic_forests/CMU_500/semantic_forest_CMU_500.gml")
    query_path = os.path.join(PROJECT_ROOT, "query_locations.json")
    results_dir = os.path.join(PROJECT_ROOT, "evaluation_results")
    results_retrieval_path = os.path.join(results_dir, "graphrag.jsonl")

    WORKING_DIR = "./work_dir"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Initialize GraphRAG
    rag = GraphRAG(
        working_dir=WORKING_DIR,
    )

    # Load the semantic forest graph
    G = nx.read_gml(graph_path)

    # Extract information from the graph
    entities, relationships, chunks, name2node = extract_entities_relationships(G)
    custom_kg = {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }
    
    if args.initialize_kg_from_semantic_forest:
        rag.insert_custom_kg(custom_kg)
        print("Knowledge Graph for GraphRAG has been successfully initialized and inserted.")
        return

    # Load queries
    with open(query_path) as f:
        data = json.load(f)
    queries = [d['query'] for d in data['query_locations']]

    if args.query:
        for idx, query in enumerate(queries):
            result = rag.query(query, param=QueryParam(mode="local"))
            modify_jsonl_file(results_retrieval_path, query, idx, result)
        print("Query results has been successfully saved.")
        return
    
    # Read all lines from the updated JSONL file
    with open(results_retrieval_path, "r") as f:
        output = [json.loads(line) for line in f]

    # Extract retrieved results
    retrieved_results, queries, responses = extract_locations_from_result(output, name2node)

    # Default center for Google Maps search (Example: Tokyo)
    default_center = {
        'latitude': 40.4433,
        'longitude': -79.9436
    }

    time_str = datetime.now().isoformat()
    immediate_output_path = os.path.join(
        results_dir, f"results_graphrag_metadata_immediate_{time_str}.json"
    )
    detailed_output_path = os.path.join(
        results_dir, f"results_graphrag_metadata_{time_str}.json"
    )

    # Compute metrics
    results = []
    for idx, retrieved_nodes in tqdm(enumerate(retrieved_results), total=len(retrieved_results)):
        query = queries[idx]
        query_item = {
            'query': query,
            'location': default_center,
            'searchable_term': query.split()[-1]  # Example logic for the search term
        }
        result = evaluate_query(query_item, retrieved_nodes, responses[idx])
        results.append(result)

        # Save immediate results
        with open(immediate_output_path, 'a') as file:
            json_line = json.dumps(result)  # Convert result to a JSON string
            file.write(json_line + '\n')

    with open(detailed_output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Compute overall metrics
    overall_metrics = {
        'total_queries': len(results),
        'successful_queries': len([r for r in results if r['success']]),
        'metrics': {
            'semantic_relativity': {
                'top1': compute_ci([r['metrics']['semantic_relativity']['top1'] for r in results]),
                'top5': compute_ci([r['metrics']['semantic_relativity']['top5'] for r in results])
            },
            'spatial_relativity': compute_ci([r['metrics']['spatial_relativity'] for r in results]),
            'semantic_spatial_score': {
                'top1': compute_ci([r['metrics']['semantic_spatial_score']['top1'] for r in results]),
                'top5': compute_ci([r['metrics']['semantic_spatial_score']['top5'] for r in results])
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save overall metrics
    summary_output_path = os.path.join(
        results_dir, f"results_graphrag_{time_str}.json"
    )
    with open(summary_output_path, "w") as f:
        json.dump(overall_metrics, f, indent=4)

    print("Detailed results saved to:", detailed_output_path)
    print("Overall metrics saved to:", summary_output_path)

if __name__ == "__main__":
    main()

    