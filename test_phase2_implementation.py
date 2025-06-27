#!/usr/bin/env python3
"""
Phase 2 Implementation Test Suite
Tests for RiskNICLP-5G Phase 2 functionality and backward compatibility
"""

import sys
import os
import torch
import dgl
import numpy as np
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestPhase2RiskBiasAttention(unittest.TestCase):
    """Test Risk-Bias Attention Module"""
    
    def setUp(self):
        self.hidden_dim = 64
        self.num_nodes = 10
        self.device = 'cpu'
        
    def test_simplified_risk_bias_gat_import(self):
        """Test SimplifiedRiskBiasGAT can be imported"""
        try:
            from src.models.risk_bias_attention import SimplifiedRiskBiasGAT
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import SimplifiedRiskBiasGAT: {e}")
            
    def test_simplified_risk_bias_gat_creation(self):
        """Test SimplifiedRiskBiasGAT can be created"""
        from src.models.risk_bias_attention import SimplifiedRiskBiasGAT
        
        layer = SimplifiedRiskBiasGAT(
            in_feats=self.hidden_dim,
            out_feats=self.hidden_dim,
            num_heads=4
        )
        self.assertIsNotNone(layer)
        self.assertEqual(layer.in_feats, self.hidden_dim)
        
    def test_simplified_risk_bias_gat_forward(self):
        """Test SimplifiedRiskBiasGAT forward pass"""
        from src.models.risk_bias_attention import SimplifiedRiskBiasGAT
        
        # Create test graph
        graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
        features = torch.randn(3, self.hidden_dim)
        risk_scores = torch.tensor([0.1, 0.5, 0.9])
        
        layer = SimplifiedRiskBiasGAT(
            in_feats=self.hidden_dim,
            out_feats=self.hidden_dim
        )
        
        # Test without risk scores
        output1 = layer(graph, features)
        self.assertEqual(output1.shape, (3, self.hidden_dim))
        
        # Test with risk scores
        output2 = layer(graph, features, risk_scores)
        self.assertEqual(output2.shape, (3, self.hidden_dim))
        
        # Risk-enhanced output should be different
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_risk_bias_attention_wrapper(self):
        """Test RiskBiasAttentionWrapper"""
        from src.models.risk_bias_attention import RiskBiasAttentionWrapper
        
        wrapper = RiskBiasAttentionWrapper(
            in_feats=self.hidden_dim,
            out_feats=self.hidden_dim,
            method='simplified'
        )
        self.assertIsNotNone(wrapper)


class TestPhase2NegativeSampling(unittest.TestCase):
    """Test Negative Sampling Module"""
    
    def setUp(self):
        self.num_nodes = 20
        self.device = 'cpu'
        
    def test_baseline_negative_sampler_import(self):
        """Test BaselineNegativeSampler can be imported"""
        try:
            from src.data.negative_sampler import BaselineNegativeSampler
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import BaselineNegativeSampler: {e}")
            
    def test_baseline_negative_sampler_creation(self):
        """Test BaselineNegativeSampler can be created"""
        from src.data.negative_sampler import BaselineNegativeSampler
        
        sampler = BaselineNegativeSampler(neg_ratio=3, seed=42)
        self.assertEqual(sampler.neg_ratio, 3)
        self.assertEqual(sampler.seed, 42)
        
    def test_baseline_negative_sampling(self):
        """Test baseline negative sampling functionality"""
        from src.data.negative_sampler import BaselineNegativeSampler
        
        # Create test graph and positive edges
        graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), num_nodes=self.num_nodes)
        positive_edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
        
        sampler = BaselineNegativeSampler(neg_ratio=2, seed=42)
        negative_edges = sampler.sample_negatives(graph, positive_edges)
        
        # Should generate 2x positive edges as negative
        expected_neg_count = len(positive_edges) * 2
        self.assertEqual(len(negative_edges), expected_neg_count)
        
        # Negative edges should not overlap with positive edges
        positive_set = set([tuple(edge.tolist()) for edge in positive_edges])
        negative_set = set([tuple(edge.tolist()) for edge in negative_edges])
        overlap = positive_set.intersection(negative_set)
        self.assertEqual(len(overlap), 0)
        
    def test_risk_aware_negative_sampler(self):
        """Test RiskAwareNegativeSampler"""
        from src.data.negative_sampler import RiskAwareNegativeSampler
        
        sampler = RiskAwareNegativeSampler(
            strategy='stratified',
            neg_ratio=2,
            seed=42
        )
        
        # Create test data
        graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=10)
        positive_edges = torch.tensor([[0, 1], [1, 2]])
        risk_scores = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
        
        negative_edges = sampler.sample_negatives(graph, positive_edges, risk_scores)
        self.assertGreater(len(negative_edges), 0)
        
    def test_create_negative_sampler_factory(self):
        """Test negative sampler factory function"""
        from src.data.negative_sampler import create_negative_sampler
        
        # Test baseline sampler creation
        baseline_config = {'type': 'baseline', 'neg_ratio': 3}
        baseline_sampler = create_negative_sampler(baseline_config)
        self.assertIsNotNone(baseline_sampler)
        
        # Test risk-aware sampler creation
        risk_config = {'type': 'risk_aware', 'strategy': 'stratified'}
        risk_sampler = create_negative_sampler(risk_config)
        self.assertIsNotNone(risk_sampler)


class TestPhase2TemporalPrediction(unittest.TestCase):
    """Test Temporal Prediction Module"""
    
    def setUp(self):
        self.hidden_dim = 64
        self.device = 'cpu'
        
    def test_graph_rnn_cell_import(self):
        """Test GraphRNNCell can be imported"""
        try:
            from src.models.temporal_predictor import GraphRNNCell
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import GraphRNNCell: {e}")
            
    def test_graph_rnn_cell_creation(self):
        """Test GraphRNNCell can be created"""
        from src.models.temporal_predictor import GraphRNNCell
        
        cell = GraphRNNCell(hidden_dim=self.hidden_dim)
        self.assertEqual(cell.hidden_dim, self.hidden_dim)
        
    def test_graph_rnn_cell_forward(self):
        """Test GraphRNNCell forward pass"""
        from src.models.temporal_predictor import GraphRNNCell
        
        cell = GraphRNNCell(hidden_dim=self.hidden_dim)
        
        # Create test graph
        graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
        features = torch.randn(3, self.hidden_dim)
        
        # Test forward pass
        hidden = cell(graph, features)
        self.assertEqual(hidden.shape, (3, self.hidden_dim))
        
        # Test with previous hidden state
        hidden2 = cell(graph, features, hidden)
        self.assertEqual(hidden2.shape, (3, self.hidden_dim))
        
    def test_graph_temporal_predictor(self):
        """Test GraphTemporalPredictor"""
        from src.models.temporal_predictor import GraphTemporalPredictor
        
        predictor = GraphTemporalPredictor(
            hidden_dim=self.hidden_dim,
            future_steps=3,
            method='graph_rnn'
        )
        
        self.assertEqual(predictor.method, 'graph_rnn')
        self.assertEqual(predictor.future_steps, 3)
        
    def test_temporal_data_loader(self):
        """Test TemporalDataLoader"""
        from src.models.temporal_predictor import TemporalDataLoader
        
        # Create test graphs
        graphs = []
        for i in range(10):
            graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
            graphs.append(graph)
            
        loader = TemporalDataLoader(graphs, window_size=3, stride=1)
        sequences = loader.get_temporal_sequences()
        
        self.assertGreater(len(sequences), 0)
        self.assertEqual(len(sequences[0]['historical_graphs']), 2)  # window_size - 1


class TestPhase2TemporalMetrics(unittest.TestCase):
    """Test Temporal Evaluation Metrics"""
    
    def test_temporal_evaluator_import(self):
        """Test TemporalEvaluator can be imported"""
        try:
            from src.evaluation.temporal_metrics import TemporalEvaluator
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Cannot import TemporalEvaluator: {e}")
            
    def test_temporal_evaluator_creation(self):
        """Test TemporalEvaluator can be created"""
        from src.evaluation.temporal_metrics import TemporalEvaluator
        
        evaluator = TemporalEvaluator(lead_time_thresholds=[5, 10, 15])
        self.assertEqual(evaluator.lead_time_thresholds, [5, 10, 15])
        
    def test_lead_time_evaluation(self):
        """Test lead-time evaluation"""
        from src.evaluation.temporal_metrics import TemporalEvaluator
        
        evaluator = TemporalEvaluator()
        
        # Create test data
        predictions = torch.tensor([[0.8, 0.6, 0.3], [0.2, 0.9, 0.7]])  # [num_samples, future_steps]
        true_events = torch.tensor([10.0, 15.0])  # Event times
        timestamps = torch.tensor([5.0, 8.0])     # Prediction times
        
        results = evaluator.evaluate_lead_time(predictions, true_events, timestamps)
        
        self.assertIn('mean_lead_time', results)
        self.assertIn('successful_predictions', results)
        
    def test_risk_weighted_metrics(self):
        """Test risk-weighted metrics"""
        from src.evaluation.temporal_metrics import compute_risk_weighted_metrics
        
        predictions = torch.tensor([0.8, 0.6, 0.3, 0.9])
        targets = torch.tensor([1, 1, 0, 1])
        risk_scores = torch.tensor([0.9, 0.5, 0.1, 0.8])
        
        metrics = compute_risk_weighted_metrics(predictions, targets, risk_scores)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('risk_weighted_precision', metrics)
        self.assertIn('risk_weighted_recall', metrics)


class TestPhase2TrainingSupport(unittest.TestCase):
    """Test Training Module Temporal Support"""
    
    @patch('src.models.training.TemporalEvaluator')
    @patch('src.models.training.TemporalDataLoader')
    def test_progressive_trainer_temporal_support(self, mock_temporal_loader, mock_temporal_evaluator):
        """Test ProgressiveTrainer with temporal support"""
        try:
            from src.models.training import ProgressiveTrainer
            from src.common.logger import get_logger
            
            # Mock config
            config = Mock()
            config.training.learning_rate = 1e-3
            config.training.weight_decay = 1e-4
            config.training.max_epochs = 5
            config.training.mode = 'temporal'
            
            # Mock model and temporal predictor
            model = Mock()
            temporal_predictor = Mock()
            logger = get_logger()
            
            trainer = ProgressiveTrainer(
                model=model,
                config=config,
                logger=logger,
                device='cpu',
                temporal_predictor=temporal_predictor
            )
            
            self.assertEqual(trainer.training_mode, 'temporal')
            self.assertIsNotNone(trainer.temporal_predictor)
            
        except ImportError as e:
            self.fail(f"Cannot test temporal training support: {e}")


class TestPhase2Configuration(unittest.TestCase):
    """Test Phase 2 Configuration Support"""
    
    def test_phase2_config_file_exists(self):
        """Test that Phase 2 configuration is present"""
        config_file = Path("config/baseline_config.yaml")
        self.assertTrue(config_file.exists(), "Configuration file not found")
        
        # Read and check for Phase 2 sections
        with open(config_file, 'r') as f:
            content = f.read()
            
        self.assertIn('phase2:', content, "Phase 2 configuration section not found")
        self.assertIn('risk_bias_attention:', content, "Risk-bias attention config not found")
        self.assertIn('negative_sampling:', content, "Negative sampling config not found")
        self.assertIn('temporal_prediction:', content, "Temporal prediction config not found")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with Phase 1"""
    
    def test_existing_model_imports(self):
        """Test that existing Phase 1 imports still work"""
        try:
            from src.models.riskniclp_model import RiskNICLPModel
            from src.models.encoders import SharedEncoder
            from src.data.graph_dataset import RiskNICLPDataset
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Backward compatibility broken: {e}")
            
    def test_existing_training_interface(self):
        """Test that existing training interface still works"""
        try:
            from src.models.training import ProgressiveTrainer
            from src.common.logger import get_logger
            
            # Test original constructor (without temporal_predictor)
            config = Mock()
            config.training.learning_rate = 1e-3
            config.training.weight_decay = 1e-4
            config.training.max_epochs = 5
            config.training.mode = 'standard'
            
            model = Mock()
            logger = get_logger()
            
            trainer = ProgressiveTrainer(
                model=model,
                config=config,
                logger=logger,
                device='cpu'
            )
            
            # Should work without temporal_predictor
            self.assertIsNone(trainer.temporal_predictor)
            self.assertEqual(trainer.training_mode, 'standard')
            
        except Exception as e:
            self.fail(f"Backward compatibility broken in training: {e}")


def run_phase2_tests():
    """Run all Phase 2 tests"""
    print("🧪 Running Phase 2 Implementation Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPhase2RiskBiasAttention,
        TestPhase2NegativeSampling,
        TestPhase2TemporalPrediction,
        TestPhase2TemporalMetrics,
        TestPhase2TrainingSupport,
        TestPhase2Configuration,
        TestBackwardCompatibility
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_tests - failures - errors}")
    print(f"   Failed: {failures}")
    print(f"   Errors: {errors}")
    
    if failures > 0:
        print(f"\n❌ Failed Tests:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n🚫 Error Tests:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = failures == 0 and errors == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    success = run_phase2_tests()
    sys.exit(0 if success else 1) 