## Dependencies

1. **E-RAG**
2. **nano-GraphRAG** 
3. **Embodied RAG** 



## Directory Structure

```
E-RAG/
├── Embodied-RAG/
│   ├── graph/
│   ├── evaluation_results/
│   ├── query_locations.json
│   ├── GraphRAG/
```
i.e. Put this directory under E-RAG/Embodied-RAG/

### Key Components

- `graph/`: Knowledge Graph resources.
- `evaluation_results/`: Evaluation results.
- `query_locations.json`: Query locations for evaluation.
- `GraphRAG/`: GraphRAG implementation.



## Usage

### 1. Build the Knowledge Graph

Initialize the Knowledge Graph for GraphRAG from the semantic forest of Embodied-RAG:

```bash
python evaluate.py --initialize_kg_from_semantic_forest
```

### 2. Start Evaluation

Start evaluation from query json:

```bash
python evaluate.py
```