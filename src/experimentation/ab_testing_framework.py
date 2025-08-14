"""
Production A/B Testing Framework for RAG Systems
"""

import hashlib
import json
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class AllocationStrategy(Enum):
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"  # Multi-armed bandit

@dataclass
class Variant:
    """Experiment variant configuration"""
    name: str
    config: Dict[str, Any]
    allocation: float  # Percentage of traffic
    description: str = ""
    is_control: bool = False

@dataclass
class Metric:
    """Experiment metric definition"""
    name: str
    type: str  # "binary", "continuous", "count"
    higher_is_better: bool = True
    minimum_sample_size: int = 100
    significance_level: float = 0.05

@dataclass
class ExperimentResult:
    """Results for a single variant"""
    variant_name: str
    samples: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Experiment:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[Metric]
    start_date: datetime
    end_date: Optional[datetime]
    status: ExperimentStatus
    allocation_strategy: AllocationStrategy
    target_sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: List[ExperimentResult] = field(default_factory=list)

class ABTestingFramework:
    """
    Production A/B testing framework with:
    - Multiple allocation strategies
    - Statistical significance testing
    - Multi-armed bandit optimization
    - Experiment management
    - Real-time monitoring
    - Result analysis and reporting
    """
    
    def __init__(self, persist_path: Optional[str] = "./experiments"):
        """Initialize A/B testing framework"""
        
        self.persist_path = persist_path
        
        # Active experiments
        self.experiments: Dict[str, Experiment] = {}
        
        # Experiment data
        self.experiment_data: Dict[str, List[Dict]] = defaultdict(list)
        
        # User assignments (for deterministic allocation)
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Multi-armed bandit state
        self.bandit_state: Dict[str, Dict] = {}
        
        # Load existing experiments
        if persist_path:
            self.load_experiments()
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        metrics: List[Metric],
        duration_days: int = 14,
        allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM,
        target_sample_size: int = 1000
    ) -> Experiment:
        """Create a new A/B test experiment"""
        
        # Validate variants
        total_allocation = sum(v.allocation for v in variants)
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Variant allocations must sum to 1.0, got {total_allocation}")
        
        # Ensure one control variant
        control_variants = [v for v in variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one control variant is required")
        
        # Create experiment
        experiment = Experiment(
            experiment_id=self._generate_experiment_id(name),
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            status=ExperimentStatus.DRAFT,
            allocation_strategy=allocation_strategy,
            target_sample_size=target_sample_size
        )
        
        # Initialize bandit state if using adaptive allocation
        if allocation_strategy == AllocationStrategy.ADAPTIVE:
            self._initialize_bandit(experiment)
        
        self.experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Created experiment {experiment.experiment_id}: {name}")
        
        return experiment
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{name_hash}"
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Can only start experiments in DRAFT status")
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.now()
        
        logger.info(f"Started experiment {experiment_id}")
        
        # Save state
        if self.persist_path:
            self.save_experiments()
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            raise ValueError(f"Can only stop active experiments")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now()
        
        # Calculate final results
        self.analyze_experiment(experiment_id)
        
        logger.info(f"Stopped experiment {experiment_id}")
        
        # Save state
        if self.persist_path:
            self.save_experiments()
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict] = None
    ) -> Optional[Variant]:
        """Get variant assignment for user"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is active
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Check if experiment has expired
        if experiment.end_date and datetime.now() > experiment.end_date:
            self.stop_experiment(experiment_id)
            return None
        
        # Get or assign variant
        if experiment.allocation_strategy == AllocationStrategy.DETERMINISTIC:
            variant = self._get_deterministic_variant(experiment, user_id)
        elif experiment.allocation_strategy == AllocationStrategy.WEIGHTED:
            variant = self._get_weighted_variant(experiment)
        elif experiment.allocation_strategy == AllocationStrategy.ADAPTIVE:
            variant = self._get_adaptive_variant(experiment, user_id)
        else:  # RANDOM
            variant = self._get_random_variant(experiment)
        
        # Record assignment
        self.user_assignments[experiment_id][user_id] = variant.name
        
        return variant
    
    def _get_deterministic_variant(
        self,
        experiment: Experiment,
        user_id: str
    ) -> Variant:
        """Deterministic variant assignment based on user ID"""
        
        # Check existing assignment
        if user_id in self.user_assignments[experiment.experiment_id]:
            variant_name = self.user_assignments[experiment.experiment_id][user_id]
            return next(v for v in experiment.variants if v.name == variant_name)
        
        # Hash user ID to get consistent assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (user_hash % 100) / 100.0
        
        # Assign based on allocation ranges
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.allocation
            if bucket < cumulative:
                return variant
        
        return experiment.variants[-1]
    
    def _get_weighted_variant(self, experiment: Experiment) -> Variant:
        """Weighted random variant assignment"""
        
        weights = [v.allocation for v in experiment.variants]
        return np.random.choice(experiment.variants, p=weights)
    
    def _get_random_variant(self, experiment: Experiment) -> Variant:
        """Pure random variant assignment"""
        return random.choice(experiment.variants)
    
    def _get_adaptive_variant(
        self,
        experiment: Experiment,
        user_id: str
    ) -> Variant:
        """Adaptive variant assignment using Thompson Sampling"""
        
        exp_id = experiment.experiment_id
        
        if exp_id not in self.bandit_state:
            self._initialize_bandit(experiment)
        
        # Thompson Sampling
        samples = []
        for variant in experiment.variants:
            state = self.bandit_state[exp_id][variant.name]
            # Sample from Beta distribution
            sample = np.random.beta(state['successes'] + 1, state['failures'] + 1)
            samples.append(sample)
        
        # Select variant with highest sample
        best_idx = np.argmax(samples)
        return experiment.variants[best_idx]
    
    def _initialize_bandit(self, experiment: Experiment):
        """Initialize multi-armed bandit state"""
        
        self.bandit_state[experiment.experiment_id] = {}
        
        for variant in experiment.variants:
            self.bandit_state[experiment.experiment_id][variant.name] = {
                'successes': 0,
                'failures': 0,
                'total': 0
            }
    
    def record_event(
        self,
        experiment_id: str,
        user_id: str,
        variant_name: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict] = None
    ):
        """Record experiment event"""
        
        if experiment_id not in self.experiments:
            return
        
        event = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'variant': variant_name,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.experiment_data[experiment_id].append(event)
        
        # Update bandit state if using adaptive allocation
        experiment = self.experiments[experiment_id]
        if experiment.allocation_strategy == AllocationStrategy.ADAPTIVE:
            self._update_bandit(experiment_id, variant_name, metrics)
    
    def _update_bandit(
        self,
        experiment_id: str,
        variant_name: str,
        metrics: Dict[str, Any]
    ):
        """Update bandit state based on observed reward"""
        
        if experiment_id not in self.bandit_state:
            return
        
        # Use primary metric for bandit update
        experiment = self.experiments[experiment_id]
        primary_metric = experiment.metrics[0]
        
        if primary_metric.name in metrics:
            value = metrics[primary_metric.name]
            state = self.bandit_state[experiment_id][variant_name]
            
            state['total'] += 1
            
            # Binary reward (can be customized)
            if primary_metric.type == "binary":
                if value:
                    state['successes'] += 1
                else:
                    state['failures'] += 1
            else:
                # Convert continuous metric to binary
                threshold = 0.5  # Can be configured
                if value > threshold:
                    state['successes'] += 1
                else:
                    state['failures'] += 1
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        data = self.experiment_data[experiment_id]
        
        if not data:
            return {"error": "No data collected"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        results = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'total_samples': len(df),
            'variants': {}
        }
        
        # Analyze each variant
        for variant in experiment.variants:
            variant_data = df[df['variant'] == variant.name]
            
            variant_results = {
                'samples': len(variant_data),
                'metrics': {}
            }
            
            # Calculate metrics
            for metric in experiment.metrics:
                if metric.name in variant_data['metrics'].iloc[0] if len(variant_data) > 0 else False:
                    metric_values = [m[metric.name] for m in variant_data['metrics']]
                    
                    if metric.type == "binary":
                        mean = np.mean(metric_values)
                        ci = self._calculate_binomial_ci(metric_values)
                    elif metric.type == "continuous":
                        mean = np.mean(metric_values)
                        ci = self._calculate_normal_ci(metric_values)
                    else:  # count
                        mean = np.mean(metric_values)
                        ci = self._calculate_poisson_ci(metric_values)
                    
                    variant_results['metrics'][metric.name] = {
                        'mean': mean,
                        'confidence_interval': ci,
                        'std': np.std(metric_values) if len(metric_values) > 1 else 0
                    }
            
            results['variants'][variant.name] = variant_results
        
        # Statistical significance testing
        results['comparisons'] = self._perform_significance_tests(experiment, df)
        
        # Winner determination
        results['winner'] = self._determine_winner(results)
        
        # Update experiment results
        experiment.results = [
            ExperimentResult(
                variant_name=name,
                samples=data['samples'],
                metrics={m: data['metrics'][m]['mean'] for m in data['metrics']},
                confidence_intervals={m: data['metrics'][m]['confidence_interval'] 
                                     for m in data['metrics']}
            )
            for name, data in results['variants'].items()
        ]
        
        return results
    
    def _calculate_binomial_ci(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for binary metric"""
        
        if not values:
            return (0, 0)
        
        successes = sum(values)
        n = len(values)
        
        if n == 0:
            return (0, 0)
        
        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = successes / n
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def _calculate_normal_ci(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for continuous metric"""
        
        if not values or len(values) < 2:
            return (0, 0)
        
        mean = np.mean(values)
        std_error = stats.sem(values)
        ci = stats.t.interval(confidence, len(values) - 1, 
                             loc=mean, scale=std_error)
        
        return ci
    
    def _calculate_poisson_ci(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for count metric"""
        
        if not values:
            return (0, 0)
        
        mean = np.mean(values)
        
        # Use normal approximation for large counts
        if mean > 20:
            std_error = np.sqrt(mean / len(values))
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            return (mean - z * std_error, mean + z * std_error)
        
        # Use exact Poisson for small counts
        total = sum(values)
        lower = stats.chi2.ppf((1 - confidence) / 2, 2 * total) / (2 * len(values))
        upper = stats.chi2.ppf(1 - (1 - confidence) / 2, 2 * (total + 1)) / (2 * len(values))
        
        return (lower, upper)
    
    def _perform_significance_tests(
        self,
        experiment: Experiment,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        comparisons = {}
        
        # Find control variant
        control = next(v for v in experiment.variants if v.is_control)
        control_data = df[df['variant'] == control.name]
        
        if len(control_data) == 0:
            return comparisons
        
        # Compare each treatment to control
        for variant in experiment.variants:
            if variant.is_control:
                continue
            
            variant_data = df[df['variant'] == variant.name]
            
            if len(variant_data) == 0:
                continue
            
            comparison_key = f"{control.name}_vs_{variant.name}"
            comparisons[comparison_key] = {}
            
            for metric in experiment.metrics:
                if metric.name not in control_data['metrics'].iloc[0]:
                    continue
                
                control_values = [m[metric.name] for m in control_data['metrics']]
                variant_values = [m[metric.name] for m in variant_data['metrics']]
                
                # Perform appropriate test
                if metric.type == "binary":
                    # Chi-square test
                    control_success = sum(control_values)
                    variant_success = sum(variant_values)
                    
                    contingency_table = [
                        [control_success, len(control_values) - control_success],
                        [variant_success, len(variant_values) - variant_success]
                    ]
                    
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    
                elif metric.type == "continuous":
                    # T-test
                    _, p_value = stats.ttest_ind(control_values, variant_values)
                    
                else:  # count
                    # Poisson test (approximated with normal)
                    control_mean = np.mean(control_values)
                    variant_mean = np.mean(variant_values)
                    
                    pooled_mean = (sum(control_values) + sum(variant_values)) / \
                                 (len(control_values) + len(variant_values))
                    
                    se = np.sqrt(pooled_mean * (1/len(control_values) + 1/len(variant_values)))
                    
                    if se > 0:
                        z_stat = (variant_mean - control_mean) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    else:
                        p_value = 1.0
                
                # Calculate lift
                control_mean = np.mean(control_values)
                variant_mean = np.mean(variant_values)
                
                if control_mean > 0:
                    lift = (variant_mean - control_mean) / control_mean
                else:
                    lift = 0
                
                comparisons[comparison_key][metric.name] = {
                    'p_value': p_value,
                    'significant': p_value < metric.significance_level,
                    'lift': lift,
                    'control_mean': control_mean,
                    'variant_mean': variant_mean
                }
        
        return comparisons
    
    def _determine_winner(self, results: Dict[str, Any]) -> Optional[str]:
        """Determine experiment winner"""
        
        if 'comparisons' not in results or not results['comparisons']:
            return None
        
        # Count significant wins for each variant
        wins = defaultdict(int)
        
        for comparison, metrics in results['comparisons'].items():
            for metric_name, metric_results in metrics.items():
                if metric_results['significant']:
                    # Determine which variant won
                    if metric_results['lift'] > 0:
                        # Treatment variant won
                        variant_name = comparison.split('_vs_')[1]
                        wins[variant_name] += 1
                    else:
                        # Control won
                        control_name = comparison.split('_vs_')[0]
                        wins[control_name] += 1
        
        if not wins:
            return None
        
        # Return variant with most wins
        return max(wins, key=wins.get)
    
    def get_experiment_report(self, experiment_id: str) -> str:
        """Generate human-readable experiment report"""
        
        results = self.analyze_experiment(experiment_id)
        experiment = self.experiments[experiment_id]
        
        report = []
        report.append(f"# Experiment Report: {experiment.name}")
        report.append(f"**ID**: {experiment_id}")
        report.append(f"**Status**: {experiment.status.value}")
        report.append(f"**Duration**: {experiment.start_date} to {experiment.end_date or 'ongoing'}")
        report.append(f"**Total Samples**: {results['total_samples']}")
        report.append("")
        
        # Variant results
        report.append("## Variant Performance")
        for variant_name, variant_data in results['variants'].items():
            variant = next(v for v in experiment.variants if v.name == variant_name)
            report.append(f"\n### {variant_name} {'(Control)' if variant.is_control else ''}")
            report.append(f"- Samples: {variant_data['samples']}")
            
            for metric_name, metric_data in variant_data['metrics'].items():
                report.append(f"- {metric_name}: {metric_data['mean']:.4f} "
                            f"CI: [{metric_data['confidence_interval'][0]:.4f}, "
                            f"{metric_data['confidence_interval'][1]:.4f}]")
        
        # Statistical comparisons
        if results['comparisons']:
            report.append("\n## Statistical Comparisons")
            for comparison, metrics in results['comparisons'].items():
                report.append(f"\n### {comparison}")
                for metric_name, metric_results in metrics.items():
                    report.append(f"- {metric_name}:")
                    report.append(f"  - p-value: {metric_results['p_value']:.4f}")
                    report.append(f"  - Significant: {'Yes' if metric_results['significant'] else 'No'}")
                    report.append(f"  - Lift: {metric_results['lift']:.2%}")
        
        # Winner
        if results['winner']:
            report.append(f"\n## Winner: {results['winner']}")
        else:
            report.append("\n## Winner: No clear winner yet")
        
        return "\n".join(report)
    
    def save_experiments(self, path: Optional[str] = None):
        """Save experiments to disk"""
        
        save_path = path or self.persist_path
        if not save_path:
            return
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save experiments
        exp_file = f"{save_path}/experiments.pkl"
        with open(exp_file, 'wb') as f:
            pickle.dump(self.experiments, f)
        
        # Save data
        data_file = f"{save_path}/experiment_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(dict(self.experiment_data), f)
        
        # Save assignments
        assign_file = f"{save_path}/user_assignments.pkl"
        with open(assign_file, 'wb') as f:
            pickle.dump(dict(self.user_assignments), f)
        
        # Save bandit state
        bandit_file = f"{save_path}/bandit_state.pkl"
        with open(bandit_file, 'wb') as f:
            pickle.dump(self.bandit_state, f)
        
        logger.info(f"Experiments saved to {save_path}")
    
    def load_experiments(self, path: Optional[str] = None):
        """Load experiments from disk"""
        
        load_path = path or self.persist_path
        if not load_path:
            return
        
        import os
        
        # Load experiments
        exp_file = f"{load_path}/experiments.pkl"
        if os.path.exists(exp_file):
            with open(exp_file, 'rb') as f:
                self.experiments = pickle.load(f)
        
        # Load data
        data_file = f"{load_path}/experiment_data.pkl"
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                self.experiment_data = defaultdict(list, pickle.load(f))
        
        # Load assignments
        assign_file = f"{load_path}/user_assignments.pkl"
        if os.path.exists(assign_file):
            with open(assign_file, 'rb') as f:
                self.user_assignments = defaultdict(dict, pickle.load(f))
        
        # Load bandit state
        bandit_file = f"{load_path}/bandit_state.pkl"
        if os.path.exists(bandit_file):
            with open(bandit_file, 'rb') as f:
                self.bandit_state = pickle.load(f)
        
        logger.info(f"Loaded {len(self.experiments)} experiments from {load_path}")