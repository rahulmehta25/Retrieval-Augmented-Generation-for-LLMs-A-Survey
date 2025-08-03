"""
Performance Optimization Module

This module implements various performance optimization techniques for RAG systems
including embedding quantization, GPU acceleration, and dynamic optimization.
"""

import time
import psutil
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

class PerformanceMonitor:
    """
    Monitors system performance and resource usage.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """
        Start timing an operation.
        """
        self.metrics[operation] = {'start_time': time.time()}
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and return duration.
        """
        if operation in self.metrics:
            duration = time.time() - self.metrics[operation]['start_time']
            self.metrics[operation]['duration'] = duration
            self.metrics[operation]['end_time'] = time.time()
            return duration
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        """
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        """
        return psutil.cpu_percent(interval=1)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        """
        return {
            'timings': {k: v.get('duration', 0) for k, v in self.metrics.items() if 'duration' in v},
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage()
        }

class EmbeddingQuantizer:
    """
    Implements embedding quantization to reduce memory usage and improve performance.
    """
    
    def __init__(self, quantization_type: str = 'scalar', bits: int = 8):
        """
        Initialize quantizer.
        
        Args:
            quantization_type: Type of quantization ('scalar', 'product', 'binary')
            bits: Number of bits for quantization
        """
        self.quantization_type = quantization_type
        self.bits = bits
        self.scale_factors = None
        self.offset_factors = None
    
    def quantize(self, embeddings: List[List[float]]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Quantize embeddings to reduce memory usage.
        """
        if not embeddings:
            return [], {}
        
        embeddings_array = np.array(embeddings)
        
        if self.quantization_type == 'scalar':
            return self._scalar_quantize(embeddings_array)
        elif self.quantization_type == 'product':
            return self._product_quantize(embeddings_array)
        elif self.quantization_type == 'binary':
            return self._binary_quantize(embeddings_array)
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")
    
    def _scalar_quantize(self, embeddings: np.ndarray) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Scalar quantization - quantize each dimension independently.
        """
        min_vals = np.min(embeddings, axis=0)
        max_vals = np.max(embeddings, axis=0)
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        
        # Scale to [0, 2^bits - 1]
        max_quantized = (1 << self.bits) - 1
        quantized = ((embeddings - min_vals) / ranges * max_quantized).astype(np.int32)
        
        metadata = {
            'min_vals': min_vals.tolist(),
            'max_vals': max_vals.tolist(),
            'quantization_type': 'scalar',
            'bits': self.bits
        }
        
        return quantized.tolist(), metadata
    
    def _product_quantize(self, embeddings: np.ndarray) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Product quantization - divide vectors into sub-vectors and quantize each.
        """
        # Simple implementation: divide into 4 sub-vectors
        sub_vectors = 4
        sub_size = embeddings.shape[1] // sub_vectors
        
        quantized_parts = []
        metadata_parts = []
        
        for i in range(sub_vectors):
            start_idx = i * sub_size
            end_idx = start_idx + sub_size if i < sub_vectors - 1 else embeddings.shape[1]
            
            sub_embeddings = embeddings[:, start_idx:end_idx]
            sub_quantized, sub_metadata = self._scalar_quantize(sub_embeddings)
            
            quantized_parts.append(sub_quantized)
            metadata_parts.append(sub_metadata)
        
        metadata = {
            'sub_vectors': sub_vectors,
            'sub_metadata': metadata_parts,
            'quantization_type': 'product',
            'bits': self.bits
        }
        
        return quantized_parts, metadata
    
    def _binary_quantize(self, embeddings: np.ndarray) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Binary quantization - represent each dimension as a single bit.
        """
        # Convert to binary: positive values become 1, negative become 0
        binary_embeddings = (embeddings > 0).astype(np.int8)
        
        metadata = {
            'quantization_type': 'binary',
            'bits': 1
        }
        
        return binary_embeddings.tolist(), metadata
    
    def dequantize(self, quantized_embeddings: List[List[int]], metadata: Dict[str, Any]) -> List[List[float]]:
        """
        Dequantize embeddings back to original scale.
        """
        if metadata['quantization_type'] == 'scalar':
            return self._scalar_dequantize(quantized_embeddings, metadata)
        elif metadata['quantization_type'] == 'product':
            return self._product_dequantize(quantized_embeddings, metadata)
        elif metadata['quantization_type'] == 'binary':
            return self._binary_dequantize(quantized_embeddings, metadata)
        else:
            raise ValueError(f"Unknown quantization type: {metadata['quantization_type']}")
    
    def _scalar_dequantize(self, quantized: List[List[int]], metadata: Dict[str, Any]) -> List[List[float]]:
        """
        Dequantize scalar-quantized embeddings.
        """
        quantized_array = np.array(quantized)
        min_vals = np.array(metadata['min_vals'])
        max_vals = np.array(metadata['max_vals'])
        
        ranges = max_vals - min_vals
        max_quantized = (1 << metadata['bits']) - 1
        
        dequantized = (quantized_array / max_quantized) * ranges + min_vals
        return dequantized.tolist()
    
    def _product_dequantize(self, quantized_parts: List[List[List[int]]], metadata: Dict[str, Any]) -> List[List[float]]:
        """
        Dequantize product-quantized embeddings.
        """
        dequantized_parts = []
        for i, part in enumerate(quantized_parts):
            sub_metadata = metadata['sub_metadata'][i]
            dequantized_part = self._scalar_dequantize(part, sub_metadata)
            dequantized_parts.append(dequantized_part)
        
        # Concatenate parts
        return np.concatenate([np.array(part) for part in dequantized_parts], axis=1).tolist()
    
    def _binary_dequantize(self, binary_embeddings: List[List[int]], metadata: Dict[str, Any]) -> List[List[float]]:
        """
        Dequantize binary embeddings (simple conversion back to -1/1).
        """
        binary_array = np.array(binary_embeddings)
        # Convert 0/1 back to -1/1
        return (2 * binary_array - 1).tolist()

class DynamicChunkOptimizer:
    """
    Optimizes chunk size dynamically based on content and performance metrics.
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000, 
                 target_chunk_size: int = 500):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.performance_history = []
    
    def optimize_chunk_size(self, text: str, current_performance: Dict[str, Any]) -> int:
        """
        Optimize chunk size based on text characteristics and performance.
        """
        # Analyze text characteristics
        text_stats = self._analyze_text(text)
        
        # Consider performance metrics
        performance_score = self._calculate_performance_score(current_performance)
        
        # Calculate optimal chunk size
        optimal_size = self._calculate_optimal_size(text_stats, performance_score)
        
        # Update performance history
        self.performance_history.append({
            'chunk_size': optimal_size,
            'performance_score': performance_score,
            'text_stats': text_stats
        })
        
        return optimal_size
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text characteristics to determine optimal chunking.
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            'total_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'complexity': self._calculate_complexity(text)
        }
    
    def _calculate_complexity(self, text: str) -> float:
        """
        Calculate text complexity score.
        """
        # Simple complexity metric based on word length and punctuation
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        punctuation_count = sum(1 for char in text if char in '.,;:!?')
        
        return (avg_word_length * 0.7) + (punctuation_count / len(words) * 0.3)
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """
        Calculate performance score from metrics.
        """
        # Consider retrieval time, memory usage, and accuracy
        retrieval_time = performance.get('retrieval_time', 1.0)
        memory_usage = performance.get('memory_usage', 0.5)
        
        # Normalize and combine metrics
        time_score = max(0, 1 - retrieval_time / 10.0)  # Prefer faster retrieval
        memory_score = max(0, 1 - memory_usage)  # Prefer lower memory usage
        
        return (time_score * 0.6) + (memory_score * 0.4)
    
    def _calculate_optimal_size(self, text_stats: Dict[str, Any], performance_score: float) -> int:
        """
        Calculate optimal chunk size based on text stats and performance.
        """
        base_size = self.target_chunk_size
        
        # Adjust based on text characteristics
        if text_stats['complexity'] > 0.7:
            # Complex text: smaller chunks
            base_size *= 0.8
        elif text_stats['complexity'] < 0.3:
            # Simple text: larger chunks
            base_size *= 1.2
        
        # Adjust based on performance
        if performance_score < 0.5:
            # Poor performance: try smaller chunks
            base_size *= 0.9
        
        # Ensure within bounds
        return max(self.min_chunk_size, min(self.max_chunk_size, int(base_size)))

class GPUAccelerator:
    """
    Manages GPU acceleration for RAG components.
    """
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.device = 'cuda' if self.gpu_available else 'cpu'
    
    def _check_gpu_availability(self) -> bool:
        """
        Check if GPU is available for acceleration.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_device(self) -> str:
        """
        Get the appropriate device for computation.
        """
        return self.device
    
    def optimize_for_gpu(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model configuration for GPU usage.
        """
        if not self.gpu_available:
            return model_config
        
        # Add GPU-specific optimizations
        optimized_config = model_config.copy()
        optimized_config['device'] = 'cuda'
        optimized_config['use_fp16'] = True  # Use half precision for memory efficiency
        
        return optimized_config
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information and capabilities.
        """
        if not self.gpu_available:
            return {'available': False}
        
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info.append({
                    'id': i,
                    'name': gpu_name,
                    'memory_gb': gpu_memory
                })
            
            return {
                'available': True,
                'count': gpu_count,
                'devices': gpu_info
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

class PerformanceOptimizer:
    """
    Main performance optimization orchestrator.
    """
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.quantizer = EmbeddingQuantizer()
        self.chunk_optimizer = DynamicChunkOptimizer()
        self.gpu_accelerator = GPUAccelerator()
    
    def optimize_rag_system(self, rag_system: Any) -> Dict[str, Any]:
        """
        Apply comprehensive performance optimizations to RAG system.
        """
        optimizations = {}
        
        # Monitor current performance
        self.monitor.start_timer('optimization')
        
        # GPU optimization
        if self.gpu_accelerator.gpu_available:
            optimizations['gpu'] = self.gpu_accelerator.get_gpu_info()
            logging.info("GPU acceleration available")
        
        # Memory optimization
        memory_usage = self.monitor.get_memory_usage()
        if memory_usage['percent_used'] > 80:
            logging.warning("High memory usage detected, consider quantization")
            optimizations['memory_warning'] = memory_usage
        
        # Chunk size optimization
        optimizations['chunk_optimization'] = {
            'current_size': 500,  # Default
            'recommended_size': self.chunk_optimizer.target_chunk_size
        }
        
        self.monitor.end_timer('optimization')
        
        return optimizations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        """
        return {
            'performance_metrics': self.monitor.get_performance_summary(),
            'system_resources': {
                'memory': self.monitor.get_memory_usage(),
                'cpu': self.monitor.get_cpu_usage(),
                'gpu': self.gpu_accelerator.get_gpu_info()
            },
            'optimization_history': self.chunk_optimizer.performance_history
        } 