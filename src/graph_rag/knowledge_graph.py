"""
Knowledge Graph Implementation for RAG

Builds and queries knowledge graphs from documents.
"""

import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
import spacy
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity in knowledge graph"""
    text: str
    label: str
    doc_ids: List[int]
    
    def __hash__(self):
        return hash((self.text, self.label))


@dataclass
class Relation:
    """Relation between entities"""
    subject: str
    predicate: str
    object: str
    doc_ids: List[int]


class KnowledgeGraph:
    """NetworkX-based knowledge graph"""
    
    def __init__(self):
        """Initialize knowledge graph"""
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = []
        self.doc_metadata = {}
        
    def add_entity(self, entity: Entity):
        """Add entity to graph"""
        node_id = f"{entity.text}_{entity.label}"
        
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                text=entity.text,
                label=entity.label,
                doc_ids=entity.doc_ids
            )
        else:
            # Update doc_ids
            self.graph.nodes[node_id]['doc_ids'].extend(entity.doc_ids)
        
        self.entities[node_id] = entity
    
    def add_relation(self, relation: Relation):
        """Add relation to graph"""
        subject_id = f"{relation.subject}_ENTITY"
        object_id = f"{relation.object}_ENTITY"
        
        # Ensure nodes exist
        if subject_id not in self.graph:
            self.graph.add_node(subject_id, text=relation.subject, label="ENTITY", doc_ids=[])
        if object_id not in self.graph:
            self.graph.add_node(object_id, text=relation.object, label="ENTITY", doc_ids=[])
        
        # Add edge
        self.graph.add_edge(
            subject_id,
            object_id,
            predicate=relation.predicate,
            doc_ids=relation.doc_ids
        )
        
        self.relations.append(relation)
    
    def get_subgraph(self, entity_text: str, max_depth: int = 2) -> nx.DiGraph:
        """Get subgraph around entity"""
        # Find matching nodes
        matching_nodes = [
            node for node in self.graph.nodes()
            if entity_text.lower() in self.graph.nodes[node].get('text', '').lower()
        ]
        
        if not matching_nodes:
            return nx.DiGraph()
        
        # Get k-hop neighborhood
        subgraph_nodes = set()
        for node in matching_nodes:
            for depth in range(max_depth + 1):
                if depth == 0:
                    subgraph_nodes.add(node)
                else:
                    neighbors = set()
                    for n in subgraph_nodes:
                        neighbors.update(self.graph.predecessors(n))
                        neighbors.update(self.graph.successors(n))
                    subgraph_nodes.update(neighbors)
        
        return self.graph.subgraph(subgraph_nodes)
    
    def save(self, path: str):
        """Save knowledge graph"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'graph': nx.node_link_data(self.graph),
            'entities': {k: (v.text, v.label, v.doc_ids) for k, v in self.entities.items()},
            'relations': [(r.subject, r.predicate, r.object, r.doc_ids) for r in self.relations],
            'doc_metadata': self.doc_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load knowledge graph"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = nx.node_link_graph(data['graph'])
        self.entities = {
            k: Entity(text=v[0], label=v[1], doc_ids=v[2])
            for k, v in data['entities'].items()
        }
        self.relations = [
            Relation(subject=r[0], predicate=r[1], object=r[2], doc_ids=r[3])
            for r in data['relations']
        ]
        self.doc_metadata = data['doc_metadata']


class GraphRAG:
    """Graph-based RAG implementation"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize GraphRAG
        
        Args:
            model_name: Spacy model for NER
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Spacy model {model_name} not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        self.knowledge_graph = KnowledgeGraph()
        logger.info(f"Initialized GraphRAG with {model_name}")
    
    def extract_entities_relations(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from text
        
        Args:
            text: Input text
        
        Returns:
            Tuple of entities and relations
        """
        doc = self.nlp(text)
        
        entities = []
        relations = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                doc_ids=[]
            ))
        
        # Extract relations using dependency parsing
        for token in doc:
            # Look for subject-verb-object patterns
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = token.text
                verb = token.head.text
                
                # Find objects
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        relations.append(Relation(
                            subject=subject,
                            predicate=verb,
                            object=child.text,
                            doc_ids=[]
                        ))
        
        return entities, relations
    
    def build_knowledge_graph(self, documents: List[Dict[str, Any]]):
        """
        Build knowledge graph from documents
        
        Args:
            documents: List of documents with 'content' and optional 'id'
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents...")
        
        for doc_id, doc in enumerate(documents):
            text = doc.get('content', '')
            doc_id_val = doc.get('id', doc_id)
            
            # Extract entities and relations
            entities, relations = self.extract_entities_relations(text)
            
            # Add to knowledge graph with doc_id
            for entity in entities:
                entity.doc_ids = [doc_id_val]
                self.knowledge_graph.add_entity(entity)
            
            for relation in relations:
                relation.doc_ids = [doc_id_val]
                self.knowledge_graph.add_relation(relation)
            
            # Store document metadata
            self.knowledge_graph.doc_metadata[doc_id_val] = {
                'content': text[:500],  # Store preview
                'full_content': text,
                'entities': len(entities),
                'relations': len(relations)
            }
        
        logger.info(f"Knowledge graph built with {len(self.knowledge_graph.entities)} entities "
                   f"and {len(self.knowledge_graph.relations)} relations")
    
    def graph_retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve using graph traversal
        
        Args:
            query: Query text
            k: Number of contexts to retrieve
        
        Returns:
            Retrieved contexts
        """
        # Extract entities from query
        query_entities, _ = self.extract_entities_relations(query)
        
        if not query_entities:
            logger.warning("No entities found in query")
            return []
        
        relevant_contexts = []
        processed_docs = set()
        
        for entity in query_entities:
            # Get subgraph around entity
            subgraph = self.knowledge_graph.get_subgraph(entity.text, max_depth=2)
            
            # Extract document IDs from subgraph
            for node in subgraph.nodes():
                doc_ids = subgraph.nodes[node].get('doc_ids', [])
                for doc_id in doc_ids:
                    if doc_id not in processed_docs:
                        processed_docs.add(doc_id)
                        
                        # Get document content
                        if doc_id in self.knowledge_graph.doc_metadata:
                            content = self.knowledge_graph.doc_metadata[doc_id]['full_content']
                            relevant_contexts.append(content)
                        
                        if len(relevant_contexts) >= k:
                            break
            
            if len(relevant_contexts) >= k:
                break
        
        return relevant_contexts[:k]
    
    def explain_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Explain why documents were retrieved
        
        Args:
            query: Query text
        
        Returns:
            Explanation of retrieval
        """
        query_entities, _ = self.extract_entities_relations(query)
        
        explanation = {
            'query_entities': [{'text': e.text, 'label': e.label} for e in query_entities],
            'graph_paths': [],
            'retrieved_docs': []
        }
        
        for entity in query_entities:
            subgraph = self.knowledge_graph.get_subgraph(entity.text, max_depth=2)
            
            # Find paths in subgraph
            paths = []
            for node in subgraph.nodes():
                if entity.text.lower() in node.lower():
                    # Find connected nodes
                    for neighbor in subgraph.neighbors(node):
                        edge_data = subgraph.get_edge_data(node, neighbor)
                        if edge_data:
                            paths.append({
                                'from': node,
                                'to': neighbor,
                                'relation': edge_data.get(0, {}).get('predicate', 'related_to')
                            })
            
            explanation['graph_paths'].extend(paths)
        
        return explanation
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            'num_nodes': self.knowledge_graph.graph.number_of_nodes(),
            'num_edges': self.knowledge_graph.graph.number_of_edges(),
            'num_entities': len(self.knowledge_graph.entities),
            'num_relations': len(self.knowledge_graph.relations),
            'num_documents': len(self.knowledge_graph.doc_metadata),
            'density': nx.density(self.knowledge_graph.graph),
            'is_connected': nx.is_weakly_connected(self.knowledge_graph.graph)
        }