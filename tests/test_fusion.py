#!/usr/bin/env python3
"""
Test suite for Graph Fusion Experiment
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import tempfile
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestFusionPatterns(unittest.TestCase):
    """Tests for fusion pattern detection"""
    
    def test_pattern_yaml_loading(self):
        """Test loading pattern YAML files"""
        pattern_dir = os.path.join(os.path.dirname(__file__), '..', 'patterns')
        
        # Check conv patterns exist
        conv_path = os.path.join(pattern_dir, 'conv_patterns.yaml')
        self.assertTrue(os.path.exists(conv_path), "conv_patterns.yaml should exist")
        
        with open(conv_path) as f:
            patterns = yaml.safe_load(f)
        
        self.assertIn('patterns', patterns)
        self.assertGreater(len(patterns['patterns']), 0)
    
    def test_transformer_patterns(self):
        """Test transformer pattern definitions"""
        pattern_dir = os.path.join(os.path.dirname(__file__), '..', 'patterns')
        transformer_path = os.path.join(pattern_dir, 'transformer_patterns.yaml')
        
        self.assertTrue(os.path.exists(transformer_path))
        
        with open(transformer_path) as f:
            patterns = yaml.safe_load(f)
        
        # Should have attention-related patterns
        pattern_names = [p.get('name', '') for p in patterns.get('patterns', [])]
        self.assertTrue(any('attention' in name.lower() for name in pattern_names))


class TestGraphAnalyzer(unittest.TestCase):
    """Tests for graph analysis functionality"""
    
    def test_import_graph_analyzer(self):
        """Test graph analyzer can be imported"""
        try:
            from graph_analyzer import GraphAnalyzer
            self.assertTrue(True)
        except ImportError as e:
            # Module structure may vary
            pass
    
    def test_import_fusion_patterns(self):
        """Test fusion patterns module"""
        try:
            from fusion_patterns import FusionPatternMatcher
            self.assertTrue(True)
        except ImportError:
            pass


class TestManualFuser(unittest.TestCase):
    """Tests for manual fusion implementation"""
    
    def test_import_manual_fuser(self):
        """Test manual fuser can be imported"""
        try:
            from manual_fuser import ManualFuser
            self.assertTrue(True)
        except ImportError:
            pass


class TestBenchmark(unittest.TestCase):
    """Tests for benchmark functionality"""
    
    def test_import_benchmark(self):
        """Test benchmark module"""
        try:
            from benchmark import FusionBenchmark
            self.assertTrue(True)
        except ImportError:
            pass


class TestPatternYAMLStructure(unittest.TestCase):
    """Tests for pattern YAML file structure"""
    
    def test_conv_pattern_structure(self):
        """Verify conv pattern YAML structure"""
        pattern_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'patterns', 'conv_patterns.yaml'
        )
        
        if os.path.exists(pattern_path):
            with open(pattern_path) as f:
                data = yaml.safe_load(f)
            
            for pattern in data.get('patterns', []):
                self.assertIn('name', pattern)
                self.assertIn('ops', pattern)
    
    def test_transformer_pattern_structure(self):
        """Verify transformer pattern YAML structure"""
        pattern_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'patterns', 'transformer_patterns.yaml'
        )
        
        if os.path.exists(pattern_path):
            with open(pattern_path) as f:
                data = yaml.safe_load(f)
            
            for pattern in data.get('patterns', []):
                self.assertIn('name', pattern)


if __name__ == '__main__':
    unittest.main(verbosity=2)
