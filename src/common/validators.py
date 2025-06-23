"""
Data Validation Framework for RiskNICLP-5G

Provides comprehensive validation for:
- Data quality and schema validation
- Graph structure integrity
- Model input validation
- Configuration validation
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    warnings.warn("DGL not available. Graph validation disabled.")

try:
    import great_expectations as ge
    HAS_GE = True
except ImportError:
    HAS_GE = False
    warnings.warn("Great Expectations not available. Advanced validation disabled.")


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidator:
    """Data quality validator for tabular and graph data"""
    
    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, raise exceptions on validation failures.
                        If False, only log warnings.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
        
    def validate_csv_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate CSV data schema
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            required_columns: List of required columns (subset of expected)
        
        Returns:
            Validation results dictionary
        """
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_columns': [],
            'extra_columns': [],
            'null_counts': {},
            'data_types': {},
            'is_valid': True
        }
        
        # Check column presence
        df_columns = set(df.columns)
        expected_set = set(expected_columns)
        required_set = set(required_columns or [])
        
        results['missing_columns'] = list(expected_set - df_columns)
        results['extra_columns'] = list(df_columns - expected_set)
        
        # Check required columns
        missing_required = list(required_set - df_columns)
        if missing_required:
            results['is_valid'] = False
            error_msg = f"Missing required columns: {missing_required}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                warnings.warn(error_msg)
        
        # Check null values and data types
        for col in df.columns:
            results['null_counts'][col] = df[col].isnull().sum()
            results['data_types'][col] = str(df[col].dtype)
        
        # Check for completely empty columns
        empty_columns = [col for col, null_count in results['null_counts'].items() 
                        if null_count == len(df)]
        if empty_columns:
            results['is_valid'] = False
            error_msg = f"Completely empty columns found: {empty_columns}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                warnings.warn(error_msg)
        
        self.validation_results['csv_schema'] = results
        return results
    
    def validate_feature_distribution(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        check_outliers: bool = True,
        outlier_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Validate feature distributions and detect anomalies
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            check_outliers: Whether to check for outliers
            outlier_threshold: Z-score threshold for outlier detection
        
        Returns:
            Feature distribution validation results
        """
        results = {
            'feature_stats': {},
            'outliers': {},
            'skewness': {},
            'infinite_values': {},
            'is_valid': True
        }
        
        for col in feature_columns:
            if col not in df.columns:
                continue
                
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Basic statistics
            results['feature_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
            
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            results['infinite_values'][col] = int(inf_count)
            if inf_count > 0:
                results['is_valid'] = False
                warning_msg = f"Infinite values found in {col}: {inf_count}"
                if self.strict_mode:
                    raise ValidationError(warning_msg)
                else:
                    warnings.warn(warning_msg)
            
            # Outlier detection
            if check_outliers and df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > outlier_threshold).sum()
                results['outliers'][col] = int(outliers)
                
                # Excessive outliers warning
                outlier_ratio = outliers / len(df)
                if outlier_ratio > 0.1:  # More than 10% outliers
                    warning_msg = f"High outlier ratio in {col}: {outlier_ratio:.2%}"
                    warnings.warn(warning_msg)
            
            # Skewness check
            skewness = df[col].skew()
            results['skewness'][col] = float(skewness)
            
            # Extreme skewness warning
            if abs(skewness) > 2:
                warning_msg = f"High skewness in {col}: {skewness:.2f}"
                warnings.warn(warning_msg)
        
        self.validation_results['feature_distribution'] = results
        return results
    
    def validate_attack_labels(
        self,
        df: pd.DataFrame,
        label_column: str = 'Label',
        attack_type_column: str = 'Attack Type'
    ) -> Dict[str, Any]:
        """
        Validate attack labels and their consistency
        
        Args:
            df: DataFrame with attack labels
            label_column: Name of binary label column
            attack_type_column: Name of attack type column
        
        Returns:
            Label validation results
        """
        results = {
            'label_distribution': {},
            'attack_type_distribution': {},
            'label_consistency': True,
            'is_valid': True
        }
        
        # Check label distribution
        if label_column in df.columns:
            label_counts = df[label_column].value_counts()
            results['label_distribution'] = label_counts.to_dict()
            
            # Check for extreme class imbalance
            if len(label_counts) == 2:
                minority_ratio = min(label_counts) / len(df)
                if minority_ratio < 0.01:  # Less than 1%
                    warning_msg = f"Extreme class imbalance: {minority_ratio:.2%} minority class"
                    warnings.warn(warning_msg)
        
        # Check attack type distribution
        if attack_type_column in df.columns:
            attack_counts = df[attack_type_column].value_counts()
            results['attack_type_distribution'] = attack_counts.to_dict()
        
        # Check label consistency
        if label_column in df.columns and attack_type_column in df.columns:
            # Benign traffic should have label 0
            benign_mask = df[attack_type_column].str.lower().str.contains('benign|normal', na=False)
            benign_labels = df.loc[benign_mask, label_column]
            if len(benign_labels) > 0 and (benign_labels != 0).any():
                results['label_consistency'] = False
                results['is_valid'] = False
                error_msg = "Inconsistent labels: benign traffic with non-zero labels"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                else:
                    warnings.warn(error_msg)
            
            # Attack traffic should have label 1
            attack_mask = ~benign_mask
            attack_labels = df.loc[attack_mask, label_column]
            if len(attack_labels) > 0 and (attack_labels != 1).any():
                results['label_consistency'] = False
                results['is_valid'] = False
                error_msg = "Inconsistent labels: attack traffic with zero labels"
                if self.strict_mode:
                    raise ValidationError(error_msg)
                else:
                    warnings.warn(error_msg)
        
        self.validation_results['attack_labels'] = results
        return results
    
    def validate_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        check_monotonic: bool = True,
        check_gaps: bool = True
    ) -> Dict[str, Any]:
        """
        Validate time series data
        
        Args:
            df: DataFrame with time series
            time_column: Name of time column
            check_monotonic: Whether to check for monotonic ordering
            check_gaps: Whether to check for time gaps
        
        Returns:
            Time series validation results
        """
        results = {
            'time_range': {},
            'is_monotonic': True,
            'time_gaps': [],
            'is_valid': True
        }
        
        if time_column not in df.columns:
            error_msg = f"Time column '{time_column}' not found"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                warnings.warn(error_msg)
                return results
        
        # Convert to datetime if string
        time_series = pd.to_datetime(df[time_column], errors='coerce')
        
        # Check for invalid timestamps
        invalid_count = time_series.isnull().sum()
        if invalid_count > 0:
            results['is_valid'] = False
            error_msg = f"Invalid timestamps found: {invalid_count}"
            if self.strict_mode:
                raise ValidationError(error_msg)
            else:
                warnings.warn(error_msg)
        
        # Time range
        results['time_range'] = {
            'start': str(time_series.min()),
            'end': str(time_series.max()),
            'duration': str(time_series.max() - time_series.min())
        }
        
        # Check monotonic ordering
        if check_monotonic:
            is_monotonic = time_series.is_monotonic_increasing
            results['is_monotonic'] = is_monotonic
            if not is_monotonic:
                warning_msg = "Time series is not monotonic"
                warnings.warn(warning_msg)
        
        # Check for time gaps (simplified)
        if check_gaps and len(time_series) > 1:
            time_diffs = time_series.diff().dropna()
            median_diff = time_diffs.median()
            large_gaps = time_diffs[time_diffs > 10 * median_diff]
            results['time_gaps'] = len(large_gaps)
            
            if len(large_gaps) > 0:
                warning_msg = f"Large time gaps detected: {len(large_gaps)}"
                warnings.warn(warning_msg)
        
        self.validation_results['time_series'] = results
        return results
    
    def validate_temporal_features(
        self,
        df: pd.DataFrame,
        temporal_feature_pattern: str = "temporal_",
        check_range: bool = True,
        expected_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Dict[str, Any]:
        """
        Validate temporal encoded features
        
        Args:
            df: DataFrame with temporal features
            temporal_feature_pattern: Pattern to identify temporal features
            check_range: Whether to check value ranges
            expected_range: Expected value range for temporal features
        
        Returns:
            Temporal feature validation results
        """
        results = {
            'temporal_feature_count': 0,
            'feature_stats': {},
            'range_violations': {},
            'encoding_type': None,
            'is_valid': True
        }
        
        # Find temporal features
        temporal_features = [col for col in df.columns if temporal_feature_pattern in col]
        results['temporal_feature_count'] = len(temporal_features)
        
        if not temporal_features:
            warning_msg = "No temporal features found"
            warnings.warn(warning_msg)
            return results
        
        # Detect encoding type
        if any('logbin' in col for col in temporal_features):
            results['encoding_type'] = 'log_bin'
        elif any('fourier' in col for col in temporal_features):
            results['encoding_type'] = 'random_fourier'
        else:
            results['encoding_type'] = 'unknown'
        
        # Validate each temporal feature
        for col in temporal_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Basic statistics
            results['feature_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isnull().sum())
            }
            
            # Range validation
            if check_range:
                min_val, max_val = expected_range
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                results['range_violations'][col] = int(out_of_range)
                
                if out_of_range > 0:
                    violation_ratio = out_of_range / len(df)
                    if violation_ratio > 0.01:  # More than 1% out of range
                        results['is_valid'] = False
                        error_msg = f"Range violations in {col}: {violation_ratio:.2%}"
                        if self.strict_mode:
                            raise ValidationError(error_msg)
                        else:
                            warnings.warn(error_msg)
        
        # Encoding-specific validation
        if results['encoding_type'] == 'log_bin':
            # For log-bin encoding, features should be mostly binary/sparse
            for col in temporal_features:
                unique_values = df[col].nunique()
                if unique_values > 10:  # Too many unique values for binary encoding
                    warning_msg = f"Log-bin feature {col} has {unique_values} unique values (expected ≤10)"
                    warnings.warn(warning_msg)
        
        elif results['encoding_type'] == 'random_fourier':
            # For random fourier, features should be continuous and normalized
            for col in temporal_features:
                feature_std = df[col].std()
                if feature_std < 0.1 or feature_std > 2.0:  # Unexpected standard deviation
                    warning_msg = f"Random Fourier feature {col} has unusual std: {feature_std:.3f}"
                    warnings.warn(warning_msg)
        
        self.validation_results['temporal_features'] = results
        return results
    
    def save_validation_report(self, output_path: Union[str, Path]):
        """Save validation results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"📊 Validation report saved to {output_path}")


class GraphValidator:
    """Graph structure validator for DGL graphs"""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_results = {}
        
        if not HAS_DGL:
            warnings.warn("DGL not available. Graph validation disabled.")
    
    def validate_heterograph_structure(
        self,
        graph,
        expected_node_types: List[str],
        expected_edge_types: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Validate heterogeneous graph structure
        
        Args:
            graph: DGL heterograph
            expected_node_types: List of expected node type names
            expected_edge_types: List of (src_type, edge_type, dst_type) tuples
        
        Returns:
            Graph structure validation results
        """
        if not HAS_DGL:
            return {'error': 'DGL not available'}
        
        results = {
            'node_types': list(graph.ntypes),
            'edge_types': list(graph.canonical_etypes),
            'node_counts': {},
            'edge_counts': {},
            'is_valid': True
        }
        
        # Node and edge counts
        for ntype in graph.ntypes:
            results['node_counts'][ntype] = graph.num_nodes(ntype)
        
        for etype in graph.canonical_etypes:
            results['edge_counts'][str(etype)] = graph.num_edges(etype)
        
        self.validation_results['graph_structure'] = results
        return results
    
    def validate_node_features(
        self,
        graph,
        expected_feature_dims: Dict[str, int],
        check_nan: bool = True,
        check_infinite: bool = True
    ) -> Dict[str, Any]:
        """
        Validate node features in the graph
        
        Args:
            graph: DGL graph
            expected_feature_dims: Dict mapping node type to expected feature dimension
            check_nan: Whether to check for NaN values
            check_infinite: Whether to check for infinite values
        
        Returns:
            Node feature validation results
        """
        if not HAS_DGL:
            return {'error': 'DGL not available'}
        
        results = {
            'feature_dims': {},
            'feature_stats': {},
            'nan_counts': {},
            'infinite_counts': {},
            'is_valid': True
        }
        
        for ntype in graph.ntypes:
            if 'h' in graph.nodes[ntype].data:
                features = graph.nodes[ntype].data['h']
                actual_dim = features.shape[1] if len(features.shape) > 1 else 1
                results['feature_dims'][ntype] = actual_dim
                
                # Check expected dimension
                if ntype in expected_feature_dims:
                    expected_dim = expected_feature_dims[ntype]
                    if actual_dim != expected_dim:
                        results['is_valid'] = False
                        error_msg = f"Feature dimension mismatch for {ntype}: expected {expected_dim}, got {actual_dim}"
                        if self.strict_mode:
                            raise ValidationError(error_msg)
                        else:
                            warnings.warn(error_msg)
                
                # Feature statistics
                results['feature_stats'][ntype] = {
                    'mean': float(features.mean()),
                    'std': float(features.std()),
                    'min': float(features.min()),
                    'max': float(features.max())
                }
                
                # Check for NaN values
                if check_nan:
                    nan_count = torch.isnan(features).sum().item()
                    results['nan_counts'][ntype] = nan_count
                    if nan_count > 0:
                        results['is_valid'] = False
                        error_msg = f"NaN values found in {ntype} features: {nan_count}"
                        if self.strict_mode:
                            raise ValidationError(error_msg)
                        else:
                            warnings.warn(error_msg)
                
                # Check for infinite values
                if check_infinite:
                    inf_count = torch.isinf(features).sum().item()
                    results['infinite_counts'][ntype] = inf_count
                    if inf_count > 0:
                        results['is_valid'] = False
                        error_msg = f"Infinite values found in {ntype} features: {inf_count}"
                        if self.strict_mode:
                            raise ValidationError(error_msg)
                        else:
                            warnings.warn(error_msg)
        
        self.validation_results['node_features'] = results
        return results
    
    def validate_graph_connectivity(
        self,
        graph,
        min_nodes_per_type: Optional[Dict[str, int]] = None,
        min_edges_per_type: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Validate graph connectivity properties
        
        Args:
            graph: DGL graph
            min_nodes_per_type: Minimum required nodes per type
            min_edges_per_type: Minimum required edges per type
        
        Returns:
            Connectivity validation results
        """
        if not HAS_DGL:
            return {'error': 'DGL not available'}
        
        results = {
            'connected_components': {},
            'node_degrees': {},
            'connectivity_warnings': [],
            'is_valid': True
        }
        
        # Check minimum node requirements
        if min_nodes_per_type:
            for ntype, min_count in min_nodes_per_type.items():
                if ntype in graph.ntypes:
                    actual_count = graph.num_nodes(ntype)
                    if actual_count < min_count:
                        results['is_valid'] = False
                        error_msg = f"Insufficient nodes in {ntype}: {actual_count} < {min_count}"
                        if self.strict_mode:
                            raise ValidationError(error_msg)
                        else:
                            warnings.warn(error_msg)
        
        # Check minimum edge requirements
        if min_edges_per_type:
            for etype_str, min_count in min_edges_per_type.items():
                # Convert string back to tuple if needed
                for etype in graph.canonical_etypes:
                    if str(etype) == etype_str:
                        actual_count = graph.num_edges(etype)
                        if actual_count < min_count:
                            results['is_valid'] = False
                            error_msg = f"Insufficient edges in {etype}: {actual_count} < {min_count}"
                            if self.strict_mode:
                                raise ValidationError(error_msg)
                            else:
                                warnings.warn(error_msg)
        
        # Basic degree statistics
        for ntype in graph.ntypes:
            in_degree_sum = 0
            out_degree_sum = 0
            
            for etype in graph.canonical_etypes:
                if etype[2] == ntype:  # incoming edges
                    in_degree_sum += graph.num_edges(etype)
                if etype[0] == ntype:  # outgoing edges
                    out_degree_sum += graph.num_edges(etype)
            
            num_nodes = graph.num_nodes(ntype)
            avg_in_degree = in_degree_sum / num_nodes if num_nodes > 0 else 0
            avg_out_degree = out_degree_sum / num_nodes if num_nodes > 0 else 0
            
            results['node_degrees'][ntype] = {
                'avg_in_degree': avg_in_degree,
                'avg_out_degree': avg_out_degree
            }
        
        self.validation_results['graph_connectivity'] = results
        return results
    
    def save_validation_report(self, output_path: Union[str, Path]):
        """Save graph validation results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"📊 Graph validation report saved to {output_path}")


def validate_model_inputs(
    batch: Dict[str, torch.Tensor],
    expected_keys: List[str],
    expected_shapes: Dict[str, Tuple[int, ...]]
) -> bool:
    """
    Validate model input batch format
    
    Args:
        batch: Input batch dictionary
        expected_keys: Expected keys in batch
        expected_shapes: Expected tensor shapes (excluding batch dimension)
    
    Returns:
        True if valid, raises ValidationError if invalid
    """
    
    # Check required keys
    missing_keys = set(expected_keys) - set(batch.keys())
    if missing_keys:
        raise ValidationError(f"Missing required keys in batch: {missing_keys}")
    
    # Check tensor shapes
    for key, expected_shape in expected_shapes.items():
        if key in batch:
            actual_shape = batch[key].shape[1:]  # Exclude batch dimension
            if actual_shape != expected_shape:
                raise ValidationError(
                    f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}"
                )
    
    # Check for NaN/Inf values
    for key, tensor in batch.items():
        if torch.isnan(tensor).any():
            raise ValidationError(f"NaN values found in batch['{key}']")
        if torch.isinf(tensor).any():
            raise ValidationError(f"Infinite values found in batch['{key}']")
    
    return True 