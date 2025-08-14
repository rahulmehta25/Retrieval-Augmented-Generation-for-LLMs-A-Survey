"""
Advanced Knowledge Graph with Entity Relationships and Graph Operations
"""

import networkx as nx
import spacy
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass, field
import json
import pickle
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
from pyvis.network import Network
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    text: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    documents: Set[str] = field(default_factory=set)
    frequency: int = 1

@dataclass
class Relation:
    """Knowledge graph relation"""
    source: str
    target: str
    type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    documents: Set[str] = field(default_factory=set)

@dataclass
class GraphQuery:
    """Query for knowledge graph"""
    entities: List[str]
    relations: List[str]
    max_hops: int = 2
    limit: int = 10
    include_attributes: bool = True

class AdvancedKnowledgeGraph:
    """
    Production-grade knowledge graph with:
    - Entity extraction and disambiguation
    - Relation extraction with types
    - Graph algorithms (PageRank, community detection, path finding)
    - Semantic search over graph
    - Graph visualization
    - Incremental updates
    - Graph persistence
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """Initialize knowledge graph"""
        
        # Core components
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.persist_path = persist_path
        
        # NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom entity ruler for domain-specific entities
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = self._get_entity_patterns()
            ruler.add_patterns(patterns)
        
        # Embeddings for semantic search
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Statistics
        self.stats = {
            "total_entities": 0,
            "total_relations": 0,
            "entity_types": Counter(),
            "relation_types": Counter(),
            "documents_processed": 0
        }
        
        # Load existing graph if path provided
        if persist_path:
            self.load()
    
    def _get_entity_patterns(self) -> List[Dict]:
        """Get custom entity patterns for domain-specific recognition"""
        patterns = [
            # Technical entities
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": ["api", "sdk", "framework", "library"]}}]},
            {"label": "ALGORITHM", "pattern": [{"LOWER": {"IN": ["algorithm", "model", "neural", "network"]}}]},
            
            # Business entities
            {"label": "METRIC", "pattern": [{"LOWER": {"IN": ["revenue", "profit", "margin", "roi"]}}]},
            {"label": "DEPARTMENT", "pattern": [{"LOWER": {"IN": ["engineering", "sales", "marketing", "hr"]}}]},
        ]
        return patterns
    
    def extract_entities_and_relations(self, text: str, doc_id: str = None) -> Tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from text"""
        
        doc = self.nlp(text)
        entities = []
        relations = []
        entity_map = {}
        
        # Extract entities
        for ent in doc.ents:
            entity_id = self._get_entity_id(ent.text, ent.label_)
            
            entity = Entity(
                id=entity_id,
                text=ent.text,
                type=ent.label_,
                attributes={
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "context": text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)]
                }
            )
            
            if doc_id:
                entity.documents.add(doc_id)
            
            entities.append(entity)
            entity_map[ent.text] = entity_id
        
        # Extract noun phrases as additional entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1 and chunk.text not in entity_map:
                entity_id = self._get_entity_id(chunk.text, "CONCEPT")
                
                entity = Entity(
                    id=entity_id,
                    text=chunk.text,
                    type="CONCEPT",
                    attributes={"root": chunk.root.text}
                )
                
                if doc_id:
                    entity.documents.add(doc_id)
                
                entities.append(entity)
                entity_map[chunk.text] = entity_id
        
        # Extract relations using dependency parsing
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subject and object
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"] and child.text in entity_map:
                        subject = entity_map[child.text]
                    elif child.dep_ in ["dobj", "pobj", "attr"] and child.text in entity_map:
                        obj = entity_map[child.text]
                
                if subject and obj:
                    relation = Relation(
                        source=subject,
                        target=obj,
                        type=token.lemma_,
                        attributes={"pos": token.pos_, "tense": token.tag_}
                    )
                    
                    if doc_id:
                        relation.documents.add(doc_id)
                    
                    relations.append(relation)
        
        # Extract relations from prepositions
        for token in doc:
            if token.pos_ == "ADP" and token.head.text in entity_map:
                for child in token.children:
                    if child.text in entity_map:
                        relation = Relation(
                            source=entity_map[token.head.text],
                            target=entity_map[child.text],
                            type=token.text,
                            weight=0.5  # Lower weight for prepositional relations
                        )
                        
                        if doc_id:
                            relation.documents.add(doc_id)
                        
                        relations.append(relation)
        
        return entities, relations
    
    def _get_entity_id(self, text: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        return f"{entity_type}:{text.lower().replace(' ', '_')}"
    
    def add_document(self, text: str, doc_id: str, metadata: Optional[Dict] = None):
        """Add document to knowledge graph"""
        
        # Extract entities and relations
        entities, relations = self.extract_entities_and_relations(text, doc_id)
        
        # Add entities to graph
        for entity in entities:
            if entity.id in self.entities:
                # Merge with existing entity
                self.entities[entity.id].frequency += 1
                self.entities[entity.id].documents.add(doc_id)
            else:
                # Add new entity
                self.entities[entity.id] = entity
                entity.embeddings = self.embedder.encode(entity.text)
                
                self.graph.add_node(
                    entity.id,
                    text=entity.text,
                    type=entity.type,
                    frequency=entity.frequency,
                    **entity.attributes
                )
        
        # Add relations to graph
        for relation in relations:
            self.relations.append(relation)
            
            self.graph.add_edge(
                relation.source,
                relation.target,
                type=relation.type,
                weight=relation.weight,
                **relation.attributes
            )
        
        # Update statistics
        self.stats["documents_processed"] += 1
        self.stats["total_entities"] = len(self.entities)
        self.stats["total_relations"] = len(self.relations)
        
        for entity in entities:
            self.stats["entity_types"][entity.type] += 1
        
        for relation in relations:
            self.stats["relation_types"][relation.type] += 1
        
        logger.info(f"Added document {doc_id}: {len(entities)} entities, {len(relations)} relations")
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between entities"""
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_subgraph(self, entity_ids: List[str], max_hops: int = 2) -> nx.MultiDiGraph:
        """Get subgraph around entities"""
        
        # Find all nodes within max_hops
        nodes = set(entity_ids)
        
        for _ in range(max_hops):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
                    new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes).copy()
        return subgraph
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Search entities using semantic similarity"""
        
        # Encode query
        query_embedding = self.embedder.encode(query)
        
        # Calculate similarities
        similarities = []
        for entity_id, entity in self.entities.items():
            if entity.embeddings is not None:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    entity.embeddings.reshape(1, -1)
                )[0][0]
                similarities.append((entity, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in similarities[:top_k]]
    
    def apply_pagerank(self, personalization: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Apply PageRank algorithm"""
        
        try:
            pagerank = nx.pagerank(
                self.graph,
                personalization=personalization,
                weight='weight'
            )
            return pagerank
        except:
            return {}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain algorithm"""
        
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Apply Louvain algorithm
        communities = community_louvain.best_partition(undirected)
        
        return communities
    
    def get_entity_importance(self) -> Dict[str, float]:
        """Calculate entity importance using multiple metrics"""
        
        importance = {}
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.graph)
        
        # PageRank
        pagerank = self.apply_pagerank()
        
        # Combine metrics
        for entity_id in self.entities:
            if entity_id in self.graph:
                importance[entity_id] = (
                    degree_centrality.get(entity_id, 0) * 0.3 +
                    betweenness.get(entity_id, 0) * 0.3 +
                    pagerank.get(entity_id, 0) * 0.4
                )
        
        return importance
    
    def query_graph(self, query: GraphQuery) -> Dict[str, Any]:
        """Query knowledge graph"""
        
        results = {
            "entities": [],
            "relations": [],
            "subgraph": None
        }
        
        # Find matching entities
        matched_entities = []
        for entity_text in query.entities:
            # Try exact match first
            entity_id = self._get_entity_id(entity_text, "UNKNOWN")
            
            if entity_id in self.entities:
                matched_entities.append(entity_id)
            else:
                # Semantic search
                similar = self.semantic_search(entity_text, top_k=1)
                if similar:
                    matched_entities.append(similar[0].id)
        
        if not matched_entities:
            return results
        
        # Get subgraph
        subgraph = self.get_subgraph(matched_entities, query.max_hops)
        
        # Extract entities from subgraph
        for node in subgraph.nodes():
            if node in self.entities:
                entity = self.entities[node]
                entity_dict = {
                    "id": entity.id,
                    "text": entity.text,
                    "type": entity.type,
                    "frequency": entity.frequency
                }
                
                if query.include_attributes:
                    entity_dict["attributes"] = entity.attributes
                
                results["entities"].append(entity_dict)
        
        # Extract relations from subgraph
        for source, target, data in subgraph.edges(data=True):
            relation_dict = {
                "source": source,
                "target": target,
                "type": data.get("type", "unknown"),
                "weight": data.get("weight", 1.0)
            }
            
            if query.include_attributes:
                relation_dict["attributes"] = {
                    k: v for k, v in data.items() 
                    if k not in ["type", "weight"]
                }
            
            results["relations"].append(relation_dict)
        
        # Limit results
        results["entities"] = results["entities"][:query.limit]
        results["relations"] = results["relations"][:query.limit * 2]
        
        return results
    
    def visualize(self, output_path: str = "knowledge_graph.html", max_nodes: int = 100):
        """Visualize knowledge graph"""
        
        # Create PyVis network
        net = Network(height="750px", width="100%", notebook=False)
        
        # Get most important nodes
        importance = self.get_entity_importance()
        top_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = set([node_id for node_id, _ in top_nodes])
        
        # Add nodes
        for node_id, _ in top_nodes:
            if node_id in self.entities:
                entity = self.entities[node_id]
                
                # Color by type
                color_map = {
                    "PERSON": "#FF6B6B",
                    "ORG": "#4ECDC4",
                    "LOC": "#45B7D1",
                    "CONCEPT": "#96CEB4",
                    "TECHNOLOGY": "#DDA0DD",
                    "default": "#FFA500"
                }
                
                color = color_map.get(entity.type, color_map["default"])
                
                net.add_node(
                    node_id,
                    label=entity.text,
                    title=f"{entity.type}: {entity.text}\nFreq: {entity.frequency}",
                    color=color,
                    size=20 + importance[node_id] * 30
                )
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            if source in top_node_ids and target in top_node_ids:
                net.add_edge(
                    source,
                    target,
                    title=data.get("type", ""),
                    weight=data.get("weight", 1.0)
                )
        
        # Set physics options
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "iterations": 100
                },
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "springConstant": 0.04,
                    "springLength": 100
                }
            }
        }
        """)
        
        # Save
        net.save_graph(output_path)
        logger.info(f"Graph visualization saved to {output_path}")
    
    def save(self, path: Optional[str] = None):
        """Save knowledge graph to disk"""
        
        save_path = path or self.persist_path
        if not save_path:
            return
        
        data = {
            "graph": nx.node_link_data(self.graph),
            "entities": self.entities,
            "relations": self.relations,
            "stats": self.stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Knowledge graph saved to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load knowledge graph from disk"""
        
        load_path = path or self.persist_path
        if not load_path:
            return
        
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            self.graph = nx.node_link_graph(data["graph"])
            self.entities = data["entities"]
            self.relations = data["relations"]
            self.stats = data["stats"]
            
            logger.info(f"Knowledge graph loaded from {load_path}")
        except FileNotFoundError:
            logger.info("No existing graph found, starting fresh")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        
        stats = dict(self.stats)
        
        if self.graph.number_of_nodes() > 0:
            stats.update({
                "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                "density": nx.density(self.graph),
                "num_components": nx.number_weakly_connected_components(self.graph),
                "largest_component": len(max(nx.weakly_connected_components(self.graph), key=len))
            })
        
        return stats