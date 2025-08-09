# RAGAS Evaluation Framework Documentation

## Overview

This document describes the RAGAS (Retrieval Augmented Generation Assessment) evaluation framework integrated into our RAG system. RAGAS provides automated metrics to evaluate the quality of retrieval-augmented generation without requiring extensive human evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Metrics](#core-metrics)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)
6. [Evaluation Scripts](#evaluation-scripts)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

RAGAS is a framework for evaluating RAG pipelines using a suite of metrics that measure different aspects of the system's performance:

- **Generation Quality**: How well the system generates answers
- **Retrieval Quality**: How well the system retrieves relevant documents
- **End-to-End Performance**: Overall system effectiveness

## Core Metrics

### 1. Faithfulness Score (0-1)

Measures how grounded the generated answer is in the retrieved contexts.

**How it works:**
- Decomposes the answer into atomic statements
- Verifies each statement against the retrieved contexts
- Returns the ratio of supported statements

**Interpretation:**
- High score (>0.8): Answer is well-grounded in retrieved documents
- Low score (<0.5): Answer contains hallucinations or unsupported claims

### 2. Answer Relevancy Score (0-1)

Measures how relevant the answer is to the asked question.

**How it works:**
- Generates potential questions from the answer
- Compares generated questions with the original question
- Calculates semantic similarity

**Interpretation:**
- High score (>0.8): Answer directly addresses the question
- Low score (<0.5): Answer is off-topic or incomplete

### 3. Context Relevancy Score (0-1)

Measures how relevant the retrieved contexts are to the question.

**How it works:**
- Analyzes each sentence in retrieved contexts
- Scores relevance to the question
- Averages scores across all contexts

**Interpretation:**
- High score (>0.7): Retrieved documents are highly relevant
- Low score (<0.4): Retrieval needs improvement

### 4. Context Precision Score (0-1)

Measures the ranking quality of retrieved contexts when ground truth is available.

**How it works:**
- Ranks contexts by relevance
- Checks which contexts contain ground truth information
- Calculates precision at different positions

**Interpretation:**
- High score (>0.8): Most relevant documents are ranked highly
- Low score (<0.5): Relevant documents are not prioritized

## Installation

### Requirements

```bash
# Install in a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- `ragas>=0.1.0` - Core RAGAS library
- `sentence-transformers>=2.2.0` - For embeddings
- `nltk>=3.8.0` - For text processing
- `datasets>=2.14.0` - For benchmark datasets

## Usage

### 1. Basic Evaluation

```python
from src.evaluation.ragas_metrics import RAGASEvaluator
from src.rag.naive_rag import NaiveRAG

# Initialize RAG system with evaluation
rag = NaiveRAG(config_path='config.yaml', enable_evaluation=True)

# Query with automatic evaluation
answer, ragas_scores = rag.query_with_evaluation(
    question="What is RAG?",
    ground_truth="Retrieval-Augmented Generation"  # Optional
)

print(f"Answer: {answer}")
print(f"RAGAS Scores: {ragas_scores.to_dict()}")
```

### 2. Standalone Evaluation

```python
from src.evaluation.ragas_metrics import RAGASEvaluator

# Initialize evaluator
evaluator = RAGASEvaluator()

# Evaluate existing outputs
score = evaluator.evaluate(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    contexts=["Paris is the capital city of France.", "France is in Europe."],
    ground_truth="Paris"  # Optional
)

print(f"Faithfulness: {score.faithfulness:.3f}")
print(f"Answer Relevancy: {score.answer_relevancy:.3f}")
print(f"Context Relevancy: {score.context_relevancy:.3f}")
print(f"Overall: {score.overall:.3f}")
```

### 3. Batch Evaluation

```python
# Evaluate multiple examples
examples = [
    {
        'question': 'What is ML?',
        'answer': 'Machine Learning is...',
        'contexts': ['ML is a subset of AI...'],
        'ground_truth': 'Machine Learning'
    },
    # More examples...
]

aggregated = evaluator.batch_evaluate(examples)
print(f"Average Faithfulness: {aggregated['faithfulness']:.3f}")
```

## API Endpoints

### 1. Single Query Evaluation

**Endpoint:** `POST /api/evaluate/query`

**Request:**
```json
{
    "question": "What is RAG?",
    "ground_truth": "Retrieval-Augmented Generation"  // Optional
}
```

**Response:**
```json
{
    "answer": "RAG is a technique that...",
    "contexts": ["Context 1", "Context 2"],
    "ragas_scores": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.92,
        "context_relevancy": 0.88,
        "context_precision": 0.90,
        "overall": 0.89
    }
}
```

### 2. Batch Evaluation

**Endpoint:** `POST /api/evaluate/batch`

**Request:**
```json
{
    "examples": [
        {
            "question": "Question 1",
            "ground_truth": "Answer 1"
        },
        {
            "question": "Question 2",
            "ground_truth": "Answer 2"
        }
    ]
}
```

**Response:**
```json
{
    "individual_results": [...],
    "aggregated_metrics": {
        "faithfulness": 0.87,
        "answer_relevancy": 0.91,
        "context_relevancy": 0.85,
        "overall": 0.88,
        "num_examples": 2
    }
}
```

### 3. Run Benchmark

**Endpoint:** `POST /api/benchmark/run`

**Request:**
```json
{
    "dataset": "squad",  // or "ms_marco", "custom"
    "max_examples": 100,
    "save_results": true
}
```

**Response:**
```json
{
    "dataset_name": "squad",
    "num_examples": 100,
    "metrics": {
        "faithfulness_mean": 0.82,
        "answer_relevancy_mean": 0.89,
        "context_relevancy_mean": 0.86
    },
    "latency_stats": {
        "mean": 1.2,
        "p50": 1.1,
        "p95": 2.3
    },
    "token_efficiency": 0.15
}
```

### 4. Human Evaluation Interface

**Endpoint:** `GET /`

Access the human evaluation interface at `http://localhost:8090/` for manual evaluation with a web UI.

## Evaluation Scripts

### 1. Command-Line Evaluation

```bash
# Evaluate with custom test data
python evaluate_rag_system.py \
    --test-data test_data/evaluation_examples.json \
    --output results/evaluation_results.json \
    --max-examples 20

# Run benchmark on SQuAD dataset
python evaluate_rag_system.py \
    --benchmark \
    --dataset squad \
    --max-examples 100
```

### 2. Test Script

```bash
# Quick test of RAGAS implementation
python test_ragas.py
```

Output:
```
Testing RAGAS Metrics Implementation
==================================================
✓ Faithfulness: 0.850
✓ Answer Relevancy: 0.923
✓ Context Relevancy: 0.876
✓ Context Precision: 0.900
✓ Overall Score: 0.887

✅ All RAGAS tests passed successfully!
```

## Best Practices

### 1. Preparing Test Data

Create high-quality evaluation datasets:

```json
[
    {
        "question": "Clear, specific question",
        "ground_truth": "Accurate, concise answer"
    }
]
```

### 2. Interpreting Scores

- **Overall Score > 0.8**: Excellent performance
- **Overall Score 0.6-0.8**: Good performance, some improvements needed
- **Overall Score < 0.6**: Significant improvements required

### 3. Improving Scores

**Low Faithfulness:**
- Improve prompt engineering to ground answers in context
- Add citation requirements to prompts
- Filter out low-quality retrieved documents

**Low Answer Relevancy:**
- Improve query understanding
- Add query clarification steps
- Fine-tune the generator on domain-specific data

**Low Context Relevancy:**
- Improve embedding model
- Optimize chunking strategy
- Add query expansion or rewriting

**Low Context Precision:**
- Implement reranking
- Use hybrid search
- Fine-tune retriever

### 4. Regular Evaluation

Set up automated evaluation:

```python
# Run daily benchmark
from src.evaluation.benchmark import RAGBenchmark

benchmark = RAGBenchmark(rag_system, evaluator)
results = benchmark.run_benchmark(
    dataset="custom",
    max_examples=50,
    save_results=True
)

# Monitor trends
if results.metrics['overall_mean'] < 0.7:
    send_alert("RAG performance degraded")
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: sentence_transformers**
   ```bash
   pip install sentence-transformers
   ```

2. **LLM not available for evaluation**
   - RAGAS can work without LLM but with reduced accuracy
   - Configure Ollama or OpenAI for best results

3. **Slow evaluation**
   - Reduce batch size
   - Use GPU for embeddings: `device='cuda'`
   - Cache embeddings for repeated evaluations

4. **Low scores on specific metrics**
   - Check data quality
   - Verify retrieval is working
   - Review prompt templates

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run evaluation to see detailed logs
evaluator = RAGASEvaluator()
score = evaluator.evaluate(...)
```

## Advanced Features

### 1. Custom Metrics

Extend the evaluator with custom metrics:

```python
class CustomRAGASEvaluator(RAGASEvaluator):
    def custom_metric(self, question, answer, contexts):
        # Implement custom logic
        return score
```

### 2. Weighted Scoring

Adjust metric weights for domain-specific needs:

```python
def calculate_weighted_score(ragas_score):
    weights = {
        'faithfulness': 0.4,
        'answer_relevancy': 0.3,
        'context_relevancy': 0.2,
        'context_precision': 0.1
    }
    
    weighted_sum = sum(
        getattr(ragas_score, metric) * weight 
        for metric, weight in weights.items()
    )
    return weighted_sum
```

### 3. A/B Testing

Compare different RAG configurations:

```python
from src.evaluation.benchmark import RAGBenchmark

# Test configuration A
rag_a = NaiveRAG(config_path='config_a.yaml')
benchmark_a = RAGBenchmark(rag_a)
results_a = benchmark_a.run_benchmark()

# Test configuration B
rag_b = NaiveRAG(config_path='config_b.yaml')
benchmark_b = RAGBenchmark(rag_b)
results_b = benchmark_b.run_benchmark()

# Compare
comparison = benchmark_a.compare_benchmarks([results_a, results_b])
print(f"Best configuration: {comparison['best_performers']}")
```

## Conclusion

RAGAS provides a comprehensive framework for evaluating RAG systems. By regularly monitoring these metrics, you can:

1. Identify performance issues early
2. Track improvements over time
3. Compare different configurations objectively
4. Ensure consistent quality for users

For more information, see:
- [RAGAS GitHub Repository](https://github.com/explodinggradients/ragas)
- [Original RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)