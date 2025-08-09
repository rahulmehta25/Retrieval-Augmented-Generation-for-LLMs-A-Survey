#!/usr/bin/env python3
"""
RAGAS Evaluation Script for RAG System

This script provides comprehensive evaluation of the RAG system using RAGAS metrics.
It can be run standalone or integrated into CI/CD pipelines.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag.naive_rag import NaiveRAG
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.evaluation.benchmark import RAGBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_sample_test_data() -> List[Dict[str, Any]]:
    """Create sample test data for evaluation"""
    return [
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France."
        },
        {
            "question": "Explain machine learning in simple terms.",
            "ground_truth": "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed."
        },
        {
            "question": "What are the main components of a RAG system?",
            "ground_truth": "The main components of a RAG system are: retriever (for finding relevant documents), generator (for producing answers), and a vector store (for storing embeddings)."
        },
        {
            "question": "How does retrieval-augmented generation work?",
            "ground_truth": "RAG works by first retrieving relevant documents from a knowledge base based on a query, then using those documents as context to generate an accurate answer."
        },
        {
            "question": "What is the purpose of embeddings in RAG?",
            "ground_truth": "Embeddings in RAG convert text into numerical vectors that capture semantic meaning, enabling efficient similarity search for relevant documents."
        }
    ]


def evaluate_single_query(rag_system: NaiveRAG, evaluator: RAGASEvaluator,
                         question: str, ground_truth: str = None) -> Dict[str, Any]:
    """Evaluate a single query"""
    
    # Get answer with contexts
    result = rag_system.query_with_contexts(question)
    answer = result['answer']
    contexts = result['contexts']
    
    # Evaluate with RAGAS
    if contexts:
        ragas_score = evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        return {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'ragas_scores': ragas_score.to_dict()
        }
    else:
        return {
            'question': question,
            'answer': answer,
            'contexts': [],
            'ground_truth': ground_truth,
            'ragas_scores': None,
            'error': 'No contexts retrieved'
        }


def run_evaluation(config_path: str = 'config.yaml',
                  test_data_path: str = None,
                  output_path: str = None,
                  dataset: str = "custom",
                  max_examples: int = 10) -> Dict[str, Any]:
    """
    Run comprehensive evaluation
    
    Args:
        config_path: Path to RAG configuration
        test_data_path: Path to test data JSON file
        output_path: Path to save results
        dataset: Dataset name for benchmarking
        max_examples: Maximum examples to evaluate
    
    Returns:
        Evaluation results
    """
    
    logger.info("Initializing RAG system...")
    rag_system = NaiveRAG(config_path=config_path, enable_evaluation=True)
    
    logger.info("Initializing RAGAS evaluator...")
    evaluator = RAGASEvaluator(llm_generator=rag_system.generator)
    
    # Load or create test data
    if test_data_path and Path(test_data_path).exists():
        logger.info(f"Loading test data from {test_data_path}")
        test_data = load_test_data(test_data_path)
    else:
        logger.info("Using sample test data")
        test_data = create_sample_test_data()
    
    # Limit examples if specified
    test_data = test_data[:max_examples]
    
    logger.info(f"Evaluating {len(test_data)} examples...")
    
    # Evaluate each example
    results = []
    for i, example in enumerate(test_data, 1):
        logger.info(f"Evaluating example {i}/{len(test_data)}: {example['question'][:50]}...")
        
        result = evaluate_single_query(
            rag_system,
            evaluator,
            example['question'],
            example.get('ground_truth')
        )
        results.append(result)
    
    # Calculate aggregate metrics
    aggregated = {
        'total_examples': len(results),
        'successful_evaluations': sum(1 for r in results if r.get('ragas_scores')),
        'failed_evaluations': sum(1 for r in results if not r.get('ragas_scores')),
        'average_scores': {}
    }
    
    # Calculate average scores
    valid_scores = [r['ragas_scores'] for r in results if r.get('ragas_scores')]
    if valid_scores:
        metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy', 'context_precision', 'overall']
        for metric in metrics:
            scores = [s.get(metric) for s in valid_scores if s.get(metric) is not None]
            if scores:
                aggregated['average_scores'][metric] = sum(scores) / len(scores)
    
    # Prepare final results
    final_results = {
        'config_path': config_path,
        'dataset': dataset,
        'timestamp': Path(__file__).stat().st_mtime,
        'aggregated_metrics': aggregated,
        'individual_results': results
    }
    
    # Save results if output path specified
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("RAGAS Evaluation Summary")
    print("="*60)
    print(f"Total Examples: {aggregated['total_examples']}")
    print(f"Successful: {aggregated['successful_evaluations']}")
    print(f"Failed: {aggregated['failed_evaluations']}")
    
    if aggregated['average_scores']:
        print("\nAverage RAGAS Scores:")
        print("-"*30)
        for metric, score in aggregated['average_scores'].items():
            print(f"  {metric}: {score:.3f}")
    
    print("="*60)
    
    return final_results


def run_benchmark_evaluation(config_path: str = 'config.yaml',
                            dataset: str = "squad",
                            max_examples: int = 100) -> Dict[str, Any]:
    """Run benchmark evaluation using standard datasets"""
    
    logger.info("Initializing RAG system for benchmarking...")
    rag_system = NaiveRAG(config_path=config_path, enable_evaluation=True)
    
    logger.info("Initializing evaluator...")
    evaluator = RAGASEvaluator(llm_generator=rag_system.generator)
    
    logger.info("Initializing benchmark...")
    benchmark = RAGBenchmark(rag_system, evaluator)
    
    logger.info(f"Running benchmark on {dataset} dataset...")
    result = benchmark.run_benchmark(
        dataset=dataset,
        max_examples=max_examples,
        save_results=True
    )
    
    return result.to_json()


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system with RAGAS metrics')
    parser.add_argument('--config', default='config.yaml', help='Path to RAG configuration')
    parser.add_argument('--test-data', help='Path to test data JSON file')
    parser.add_argument('--output', help='Path to save evaluation results')
    parser.add_argument('--dataset', default='custom', help='Dataset for benchmarking (squad, ms_marco, custom)')
    parser.add_argument('--max-examples', type=int, default=10, help='Maximum examples to evaluate')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark evaluation')
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            # Run benchmark evaluation
            results = run_benchmark_evaluation(
                config_path=args.config,
                dataset=args.dataset,
                max_examples=args.max_examples
            )
        else:
            # Run custom evaluation
            results = run_evaluation(
                config_path=args.config,
                test_data_path=args.test_data,
                output_path=args.output,
                dataset=args.dataset,
                max_examples=args.max_examples
            )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()