"""
RAG Benchmark Module

Automated benchmarking against standard datasets for comprehensive RAG evaluation.
"""

import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datasets import load_dataset
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from .ragas_metrics import RAGASEvaluator, RAGASScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    dataset_name: str
    timestamp: str
    num_examples: int
    metrics: Dict[str, float]
    latency_stats: Dict[str, float]
    token_efficiency: float
    detailed_results: Optional[List[Dict]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2)
    
    def save(self, path: str):
        """Save results to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())


class RAGBenchmark:
    """Automated benchmarking against standard datasets"""
    
    def __init__(self, rag_system, evaluator: Optional[RAGASEvaluator] = None):
        """
        Initialize benchmark
        
        Args:
            rag_system: RAG system to evaluate
            evaluator: RAGAS evaluator instance
        """
        self.rag = rag_system
        self.evaluator = evaluator or RAGASEvaluator()
        
    def run_benchmark(self, dataset: str = "squad", 
                     subset: Optional[str] = None,
                     max_examples: int = 100,
                     save_results: bool = True,
                     output_dir: str = "benchmark_results") -> BenchmarkResult:
        """
        Run comprehensive benchmark on a dataset
        
        Args:
            dataset: Dataset name (squad, ms_marco, natural_questions)
            subset: Dataset subset/configuration
            max_examples: Maximum number of examples to evaluate
            save_results: Whether to save results to file
            output_dir: Directory to save results
        
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Starting benchmark on {dataset} dataset")
        
        # Load dataset
        test_data = self._load_dataset(dataset, subset, max_examples)
        
        if not test_data:
            raise ValueError(f"No data loaded for dataset: {dataset}")
        
        results = {
            'faithfulness': [],
            'answer_relevancy': [],
            'context_relevancy': [],
            'context_precision': [],
            'latency': [],
            'token_efficiency': []
        }
        
        detailed_results = []
        
        # Process each example
        for i, item in enumerate(test_data):
            logger.info(f"Processing example {i+1}/{len(test_data)}")
            
            start_time = time.time()
            
            try:
                # Get RAG response
                response = self._get_rag_response(item['question'])
                
                # Calculate latency
                latency = time.time() - start_time
                results['latency'].append(latency)
                
                # Extract components
                answer = response.get('answer', '')
                contexts = response.get('contexts', [])
                
                # Calculate RAGAS metrics
                ragas_score = self.evaluator.evaluate(
                    question=item['question'],
                    answer=answer,
                    contexts=contexts,
                    ground_truth=item.get('ground_truth')
                )
                
                # Store scores
                results['faithfulness'].append(ragas_score.faithfulness)
                results['answer_relevancy'].append(ragas_score.answer_relevancy)
                results['context_relevancy'].append(ragas_score.context_relevancy)
                
                if ragas_score.context_precision is not None:
                    results['context_precision'].append(ragas_score.context_precision)
                
                # Calculate token efficiency
                if answer and contexts:
                    efficiency = len(answer.split()) / max(1, len(' '.join(contexts).split()))
                    results['token_efficiency'].append(efficiency)
                
                # Store detailed result
                detailed_results.append({
                    'question': item['question'],
                    'answer': answer,
                    'ground_truth': item.get('ground_truth'),
                    'ragas_scores': ragas_score.to_dict(),
                    'latency': latency
                })
                
            except Exception as e:
                logger.error(f"Error processing example {i+1}: {e}")
                continue
        
        # Aggregate results
        aggregated_metrics = {}
        for metric, values in results.items():
            if values:
                aggregated_metrics[f'{metric}_mean'] = np.mean(values)
                aggregated_metrics[f'{metric}_std'] = np.std(values)
                if metric == 'latency':
                    aggregated_metrics[f'{metric}_p50'] = np.percentile(values, 50)
                    aggregated_metrics[f'{metric}_p95'] = np.percentile(values, 95)
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            dataset_name=dataset,
            timestamp=datetime.now().isoformat(),
            num_examples=len(detailed_results),
            metrics=aggregated_metrics,
            latency_stats={
                'mean': aggregated_metrics.get('latency_mean', 0),
                'std': aggregated_metrics.get('latency_std', 0),
                'p50': aggregated_metrics.get('latency_p50', 0),
                'p95': aggregated_metrics.get('latency_p95', 0)
            },
            token_efficiency=aggregated_metrics.get('token_efficiency_mean', 0),
            detailed_results=detailed_results
        )
        
        # Save results if requested
        if save_results:
            output_path = f"{output_dir}/{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            benchmark_result.save(output_path)
            logger.info(f"Results saved to {output_path}")
        
        # Print summary
        self._print_summary(benchmark_result)
        
        return benchmark_result
    
    def _load_dataset(self, dataset_name: str, subset: Optional[str], 
                     max_examples: int) -> List[Dict[str, Any]]:
        """
        Load dataset for benchmarking
        
        Args:
            dataset_name: Name of the dataset
            subset: Dataset subset
            max_examples: Maximum examples to load
        
        Returns:
            List of examples
        """
        test_data = []
        
        try:
            if dataset_name == "squad":
                # Load SQuAD dataset
                dataset = load_dataset("squad", split="validation")
                
                for i, item in enumerate(dataset):
                    if i >= max_examples:
                        break
                    
                    test_data.append({
                        'question': item['question'],
                        'ground_truth': item['answers']['text'][0] if item['answers']['text'] else None,
                        'context': item['context']
                    })
            
            elif dataset_name == "ms_marco":
                # Load MS MARCO dataset
                dataset = load_dataset("ms_marco", "v1.1", split="test[:1000]")
                
                for i, item in enumerate(dataset):
                    if i >= max_examples:
                        break
                    
                    test_data.append({
                        'question': item['query'],
                        'ground_truth': item.get('answers', [None])[0],
                        'passages': item.get('passages', [])
                    })
            
            elif dataset_name == "natural_questions":
                # Load Natural Questions dataset
                dataset = load_dataset("natural_questions", split="validation[:1000]")
                
                for i, item in enumerate(dataset):
                    if i >= max_examples:
                        break
                    
                    test_data.append({
                        'question': item['question']['text'],
                        'ground_truth': item['annotations'][0]['short_answers'][0]['text'] 
                                      if item['annotations'] and item['annotations'][0]['short_answers'] 
                                      else None
                    })
            
            elif dataset_name == "custom":
                # Load custom dataset from JSON file
                if subset:
                    with open(subset, 'r') as f:
                        custom_data = json.load(f)
                    
                    for i, item in enumerate(custom_data):
                        if i >= max_examples:
                            break
                        test_data.append(item)
                else:
                    # Create sample data for testing
                    test_data = [
                        {
                            'question': 'What is the capital of France?',
                            'ground_truth': 'Paris'
                        },
                        {
                            'question': 'Who wrote Romeo and Juliet?',
                            'ground_truth': 'William Shakespeare'
                        },
                        {
                            'question': 'What is the largest planet in our solar system?',
                            'ground_truth': 'Jupiter'
                        }
                    ][:max_examples]
            
            else:
                logger.warning(f"Unknown dataset: {dataset_name}. Using sample data.")
                # Default sample data
                test_data = [
                    {
                        'question': 'What is machine learning?',
                        'ground_truth': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.'
                    }
                ]
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            # Fallback to sample data
            test_data = [
                {
                    'question': 'What is RAG?',
                    'ground_truth': 'Retrieval-Augmented Generation'
                }
            ]
        
        logger.info(f"Loaded {len(test_data)} examples from {dataset_name}")
        return test_data
    
    def _get_rag_response(self, question: str) -> Dict[str, Any]:
        """
        Get response from RAG system
        
        Args:
            question: Question to ask
        
        Returns:
            Response dictionary with answer and contexts
        """
        try:
            # Try to get both answer and contexts
            if hasattr(self.rag, 'query_with_contexts'):
                return self.rag.query_with_contexts(question)
            elif hasattr(self.rag, 'query'):
                # Get answer
                answer = self.rag.query(question)
                
                # Try to get contexts separately
                contexts = []
                if hasattr(self.rag, 'retrieve'):
                    contexts = self.rag.retrieve(question)
                    # Extract content if contexts are dict objects
                    if contexts and isinstance(contexts[0], dict):
                        contexts = [ctx.get('content', str(ctx)) for ctx in contexts]
                
                return {
                    'answer': answer,
                    'contexts': contexts
                }
            else:
                raise AttributeError("RAG system must have either query_with_contexts or query method")
        
        except Exception as e:
            logger.error(f"Error getting RAG response: {e}")
            return {
                'answer': '',
                'contexts': []
            }
    
    def _print_summary(self, result: BenchmarkResult):
        """
        Print benchmark summary
        
        Args:
            result: Benchmark result
        """
        print("\n" + "="*50)
        print(f"Benchmark Results for {result.dataset_name}")
        print("="*50)
        print(f"Number of examples: {result.num_examples}")
        print(f"Timestamp: {result.timestamp}")
        print("\nMetrics:")
        print("-"*30)
        
        for metric, value in result.metrics.items():
            if '_mean' in metric:
                metric_name = metric.replace('_mean', '')
                print(f"  {metric_name}: {value:.3f}")
        
        print("\nLatency Statistics:")
        print("-"*30)
        for stat, value in result.latency_stats.items():
            print(f"  {stat}: {value:.3f}s")
        
        print(f"\nToken Efficiency: {result.token_efficiency:.3f}")
        print("="*50)
    
    def compare_benchmarks(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Compare multiple benchmark results
        
        Args:
            results: List of benchmark results to compare
        
        Returns:
            Comparison summary
        """
        comparison = {
            'datasets': [r.dataset_name for r in results],
            'metrics_comparison': {}
        }
        
        # Extract common metrics
        all_metrics = set()
        for r in results:
            all_metrics.update([k.replace('_mean', '') for k in r.metrics.keys() if '_mean' in k])
        
        # Compare each metric
        for metric in all_metrics:
            comparison['metrics_comparison'][metric] = {}
            for r in results:
                metric_key = f'{metric}_mean'
                if metric_key in r.metrics:
                    comparison['metrics_comparison'][metric][r.dataset_name] = r.metrics[metric_key]
        
        # Find best performing dataset for each metric
        comparison['best_performers'] = {}
        for metric, values in comparison['metrics_comparison'].items():
            if values:
                if metric == 'latency':  # Lower is better for latency
                    best = min(values.items(), key=lambda x: x[1])
                else:  # Higher is better for other metrics
                    best = max(values.items(), key=lambda x: x[1])
                comparison['best_performers'][metric] = best[0]
        
        return comparison