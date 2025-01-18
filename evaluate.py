import os
import json
import networkx as nx
import numpy as np
import scipy.stats
from datetime import datetime
from tqdm import tqdm
import googlemaps
import argparse
from nano_graphrag import GraphRAG, QueryParam

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

def get_ground_truth_locations(query_item):
    """
    Get k (5) closest ground truth locations from Google Maps based on the agent's location
    and a searchable term.
    """
    agent_location = query_item['location']
    searchable_term = query_item['searchable_term']
    
    places_result = gmaps.places_nearby(
        location=(agent_location['latitude'], agent_location['longitude']),
        keyword=searchable_term.strip('"'),  
        rank_by='distance'                   
    )
    
    ground_truth_locations = []
    for place in places_result.get('results', [])[:5]:
        location = place['geometry']['location']
        ground_truth_locations.append({
            'name': place['name'],
            'position': {
                'latitude': location['lat'],
                'longitude': location['lng']
            },
            'place_id': place['place_id']
        })
        
    return ground_truth_locations

def modify_jsonl_file(file_path, query_value, idx):
    """
    Modifies the last line of a JSONL file to update 'query' and 'idx' fields.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    last_line = json.loads(lines[-1])
    last_line['query'] = query_value
    last_line['idx'] = idx
    
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

def compute_distance(loc1, loc2) -> float:
    """
    Compute the haversine distance in meters between two latitude/longitude pairs.
    """
    R = 6371000  # Earth's radius in meters
    
    lat1 = loc1.get('y', loc1.get('latitude', 0))
    lon1 = loc1.get('x', loc1.get('longitude', 0))
    lat2 = loc2.get('latitude', 0)
    lon2 = loc2.get('longitude', 0)
    
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def compute_metrics(retrieved_locations, ground_truth_locations):
    """
    Compute NDCG-like metric (spatially weighted) and spatial precision 
    given retrieved vs. ground truth locations.
    """
    radius = 1000  # distance threshold (meters)
    if not retrieved_locations or not ground_truth_locations:
        return {
            'ndcg': 0.0,
            'spatial_precision': 0.0,
            'raw_distances': []
        }
    
    # Distance matrix
    distances = np.zeros((len(retrieved_locations), len(ground_truth_locations)))
    for i, retrieved in enumerate(retrieved_locations):
        for j, truth in enumerate(ground_truth_locations):
            distances[i, j] = compute_distance(
                retrieved['position'],
                truth['position']
            )
    
    def compute_ndcg():
        # Relevance is an exponential decay based on min distance to any ground-truth node
        relevance_scores = np.exp(-distances.min(axis=1) / (radius * 0.1))
        dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))
        ideal_dcg = np.sum(np.sort(relevance_scores)[::-1] 
                           / np.log2(np.arange(2, len(relevance_scores) + 2)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def compute_spatial_precision():
        # Spatial precision is also an exponential decay but with a different factor
        proximity_scores = np.exp(-distances.min(axis=1) / radius)
        return np.mean(proximity_scores)
    
    metrics = {
        'ndcg': compute_ndcg(),
        'spatial_precision': compute_spatial_precision(),
        'raw_distances': distances.min(axis=1).tolist()
    }
    return metrics

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
                                 if data.get('type') == 'location' 
                                 else 'cluster of locations')
        entity['position'] = json.dumps(data.get('position', {}))
        entity['source_id'] = source_id
        
        # Create the relationship based on level
        if not data.get('level'):
            entity['description'] = data.get('caption', '')
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
    
    for result in results:
        data = result['results']
        ret = []
        for node_name in data:
            # Clean up the node name
            node_name = node_name.strip("\"").lower()
            if node_name not in name2node:
                continue
            node = name2node[node_name]
            if node['type'] != 'location':
                continue
            d_transformed = {
                'name': node['name'],
                'description': node.get('caption', ''),
                'position': node['position']
            }
            ret.append(d_transformed)

        all_queries.append(result['query'])
        all_rets.append(ret)

    return all_rets, all_queries

def parse_args():
    parser = argparse.ArgumentParser(description="Run GraphRAG pipeline with semantic forest.")
    parser.add_argument(
        "--initialize_kg_from_semantic_forest",
        action="store_true",
        help="If set, will build the GraphRAG knowledge graph from a semantic forest file."
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
    graph_path = os.path.join(PROJECT_ROOT, "graph/semantic_forests/graph/semantic_forest_graph.gml")
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

    # For each query, run GraphRAG and modify the JSONL file
    for idx, query in enumerate(queries):
        result = rag.query(query, param=QueryParam(mode="local"))
        print(result)
        # Update the JSONL file with current query info
        modify_jsonl_file(results_retrieval_path, query, idx)

    # Read all lines from the updated JSONL file
    with open(results_retrieval_path, "r") as f:
        output = [json.loads(line) for line in f]

    # Extract retrieved results
    retrieved_results, queries = extract_locations_from_result(output, name2node)

    # Default center for Google Maps search (Example: Tokyo)
    default_center = {
        'latitude': 35.6762,
        'longitude': 139.6503
    }

    # Compute metrics
    results = []
    for idx, retrieved_nodes in tqdm(enumerate(retrieved_results), total=len(retrieved_results)):
        query = queries[idx]
        query_item = {
            'query': query,
            'location': default_center,
            'searchable_term': query.split()[-1]  # Example logic for the search term
        }
        # Get ground truth from Google Maps
        ground_truth_locations = get_ground_truth_locations(query_item)

        # Compute metrics
        metrics = compute_metrics(
            retrieved_nodes[:5],
            ground_truth_locations
        )
        results.append({'metrics': metrics})

    # Save detailed results
    time_str = datetime.now().isoformat()
    detailed_output_path = os.path.join(
        results_dir, f"results_graphrag_metadata_{time_str}.json"
    )
    with open(detailed_output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Compute overall metrics
    overall_metrics = {
        'total_queries': len(results),
        'metrics': {
            'ndcg': compute_ci([r['metrics']['ndcg'] for r in results]),
            'spatial_precision': compute_ci([r['metrics']['spatial_precision'] for r in results])
        },
        'timestamp': time_str
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

    