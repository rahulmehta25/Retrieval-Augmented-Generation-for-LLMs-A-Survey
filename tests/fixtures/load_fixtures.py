"""
Utility functions to load test fixtures
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FixtureLoader:
    """Helper class to load test fixtures"""
    
    def __init__(self, fixtures_dir: Optional[str] = None):
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).parent
        self.fixtures_dir = Path(fixtures_dir)
    
    def load_json_fixture(self, filename: str) -> Dict[str, Any]:
        """Load a JSON fixture file"""
        filepath = self.fixtures_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Fixture file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_sample_documents(self) -> Dict[str, Any]:
        """Load sample documents fixture"""
        return self.load_json_fixture('sample_documents.json')
    
    def load_evaluation_data(self) -> Dict[str, Any]:
        """Load evaluation data fixture"""
        return self.load_json_fixture('evaluation_data.json')
    
    def get_sample_queries(self, complexity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get sample queries, optionally filtered by complexity"""
        documents = self.load_sample_documents()
        queries = documents['queries']
        
        if complexity is None:
            return queries
        
        complexity_ranges = {
            'simple': (0.0, 0.4),
            'medium': (0.4, 0.7),
            'complex': (0.7, 1.0)
        }
        
        if complexity not in complexity_ranges:
            raise ValueError(f"Invalid complexity: {complexity}. Must be one of {list(complexity_ranges.keys())}")
        
        min_complexity, max_complexity = complexity_ranges[complexity]
        return [q for q in queries if min_complexity <= q['complexity'] < max_complexity]
    
    def get_sample_documents_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get sample documents filtered by topic"""
        documents = self.load_sample_documents()
        docs = documents['documents']
        
        topic_keywords = {
            'machine_learning': ['machine learning', 'ml'],
            'neural_networks': ['neural network', 'deep learning'],
            'nlp': ['natural language processing', 'nlp'],
            'ai_applications': ['ai applications', 'healthcare']
        }
        
        if topic not in topic_keywords:
            return docs
        
        keywords = topic_keywords[topic]
        filtered_docs = []
        
        for doc in docs:
            content_lower = doc['content'].lower()
            title_lower = doc['title'].lower()
            
            if any(keyword in content_lower or keyword in title_lower for keyword in keywords):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_conversation_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Get a specific conversation scenario"""
        documents = self.load_sample_documents()
        scenarios = documents['conversation_scenarios']
        
        for scenario in scenarios:
            if scenario['id'] == scenario_id:
                return scenario
        
        raise ValueError(f"Scenario not found: {scenario_id}")
    
    def get_evaluation_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get a specific evaluation dataset"""
        eval_data = self.load_evaluation_data()
        datasets = eval_data['evaluation_datasets']
        
        if dataset_name not in datasets:
            raise ValueError(f"Dataset not found: {dataset_name}. Available: {list(datasets.keys())}")
        
        return datasets[dataset_name]
    
    def get_retrieval_test_cases(self, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get retrieval test cases, optionally filtered by method"""
        eval_data = self.load_evaluation_data()
        test_cases = eval_data['retrieval_test_cases']
        
        if method is None:
            return test_cases
        
        return [tc for tc in test_cases if tc['retrieval_method'] == method]
    
    def get_generation_test_cases(self) -> List[Dict[str, Any]]:
        """Get generation test cases"""
        eval_data = self.load_evaluation_data()
        return eval_data['generation_test_cases']
    
    def get_edge_cases(self, case_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get edge cases, optionally filtered by type"""
        eval_data = self.load_evaluation_data()
        edge_cases = eval_data['edge_cases']
        
        if case_type is None:
            return edge_cases
        
        return [ec for ec in edge_cases if ec['case'] == case_type]
    
    def get_performance_benchmarks(self) -> Dict[str, Any]:
        """Get performance benchmarks"""
        eval_data = self.load_evaluation_data()
        return eval_data['performance_benchmarks']
    
    def create_mock_document(
        self, 
        doc_id: str, 
        title: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a mock document for testing"""
        if metadata is None:
            metadata = {
                "source": f"{doc_id}.txt",
                "page": 1,
                "section": "test",
                "author": "Test Author",
                "date": "2024-01-01"
            }
        
        return {
            "id": doc_id,
            "title": title,
            "content": content,
            "metadata": metadata
        }
    
    def create_mock_query(
        self, 
        query_id: str, 
        text: str, 
        intent: str = "question", 
        complexity: float = 0.5,
        expected_contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a mock query for testing"""
        if expected_contexts is None:
            expected_contexts = []
        
        return {
            "id": query_id,
            "text": text,
            "intent": intent,
            "complexity": complexity,
            "expected_contexts": expected_contexts,
            "ground_truth": ground_truth
        }
    
    def validate_fixture_data(self) -> Dict[str, bool]:
        """Validate all fixture data"""
        validation_results = {}
        
        try:
            # Validate sample documents
            docs = self.load_sample_documents()
            validation_results['sample_documents'] = self._validate_sample_documents(docs)
        except Exception as e:
            validation_results['sample_documents'] = False
            print(f"Sample documents validation failed: {e}")
        
        try:
            # Validate evaluation data
            eval_data = self.load_evaluation_data()
            validation_results['evaluation_data'] = self._validate_evaluation_data(eval_data)
        except Exception as e:
            validation_results['evaluation_data'] = False
            print(f"Evaluation data validation failed: {e}")
        
        return validation_results
    
    def _validate_sample_documents(self, data: Dict[str, Any]) -> bool:
        """Validate sample documents structure"""
        required_keys = ['documents', 'queries', 'conversation_scenarios']
        
        for key in required_keys:
            if key not in data:
                print(f"Missing key: {key}")
                return False
        
        # Validate documents
        for doc in data['documents']:
            required_doc_keys = ['id', 'title', 'content', 'metadata']
            if not all(key in doc for key in required_doc_keys):
                print(f"Invalid document structure: {doc.get('id', 'unknown')}")
                return False
        
        # Validate queries
        for query in data['queries']:
            required_query_keys = ['id', 'text', 'intent', 'complexity']
            if not all(key in query for key in required_query_keys):
                print(f"Invalid query structure: {query.get('id', 'unknown')}")
                return False
        
        return True
    
    def _validate_evaluation_data(self, data: Dict[str, Any]) -> bool:
        """Validate evaluation data structure"""
        required_keys = ['evaluation_datasets', 'retrieval_test_cases', 'generation_test_cases', 'edge_cases', 'performance_benchmarks']
        
        for key in required_keys:
            if key not in data:
                print(f"Missing key: {key}")
                return False
        
        return True


# Global fixture loader instance
fixture_loader = FixtureLoader()

# Convenience functions
def load_sample_documents() -> Dict[str, Any]:
    """Load sample documents fixture"""
    return fixture_loader.load_sample_documents()

def load_evaluation_data() -> Dict[str, Any]:
    """Load evaluation data fixture"""
    return fixture_loader.load_evaluation_data()

def get_simple_queries() -> List[Dict[str, Any]]:
    """Get simple queries for basic testing"""
    return fixture_loader.get_sample_queries('simple')

def get_complex_queries() -> List[Dict[str, Any]]:
    """Get complex queries for advanced testing"""
    return fixture_loader.get_sample_queries('complex')

def get_ml_documents() -> List[Dict[str, Any]]:
    """Get machine learning related documents"""
    return fixture_loader.get_sample_documents_by_topic('machine_learning')

def get_performance_targets() -> Dict[str, Any]:
    """Get performance benchmarks"""
    return fixture_loader.get_performance_benchmarks()