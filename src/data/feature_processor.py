"""
Feature Processing and Risk Vectorization for RiskNICLP-5G

Implements:
- 112→45 dimensional feature selection (variance + correlation + domain knowledge)
- Risk score vectorization with 8-bin discretization
- Z-score + sigmoid normalization
- Adaptive/fixed binning strategies
- Temporal encoding (Log-bin and Random Fourier methods)
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
import json
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings

from ..common.logger import get_logger, monitor_performance


class FeatureProcessor:
    """Feature selection and preprocessing pipeline"""
    
    def __init__(
        self,
        method: str = "hybrid",
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.9,
        target_dim: int = 45,
        save_cache: bool = True,
        cache_dir: Optional[str] = None,
        temporal_config: Optional[Dict] = None
    ):
        """
        Args:
            method: Feature selection method ('variance', 'correlation', 'hybrid', 'domain')
            variance_threshold: Minimum variance threshold
            correlation_threshold: Maximum correlation threshold  
            target_dim: Target feature dimension
            save_cache: Whether to save/load feature selection cache
            cache_dir: Cache directory path
            temporal_config: Temporal encoding configuration
        """
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.target_dim = target_dim
        self.save_cache = save_cache
        self.cache_dir = Path(cache_dir or "data/feature_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporal encoding configuration
        self.temporal_config = temporal_config or {}
        self.temporal_enabled = self.temporal_config.get('enable', False)
        self.temporal_method = self.temporal_config.get('method', 'log_bin')
        self.log_bins = self.temporal_config.get('log_bins', 16)
        self.fourier_dim = self.temporal_config.get('fourier_dim', 32)
        self.fourier_sigma = self.temporal_config.get('fourier_sigma', 1.0)
        self.time_features = self.temporal_config.get('time_features', ['StartTime', 'LastTime', 'Dur'])
        self.min_delta = self.temporal_config.get('min_delta', 0.001)
        self.max_delta = self.temporal_config.get('max_delta', 3600.0)
        
        # Temporal encoding components
        self.fourier_weights = None  # For random fourier encoding
        self.log_bin_edges = None    # For log bin encoding
        
        # Feature selection components
        self.variance_selector = None
        self.correlation_mask = None
        self.kbest_selector = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        
        # Domain knowledge features (5G network security relevant)
        self.domain_features = [
            # Protocol and QoS related
            'Proto', 'sTos', 'dTos', 'sTtl', 'dTtl',
            # Traffic volume and patterns  
            'TotPkts', 'SrcPkts', 'DstPkts', 'TotBytes', 'SrcBytes', 'DstBytes',
            # Timing features
            'Dur', 'RunTime', 'Mean', 'Stddev', 'Sum', 'Min', 'Max',
            # Flow characteristics
            'SrcAddr', 'DstAddr', 'Sport', 'Dport',
            # Advanced flow metrics
            'Rate', 'SrcRate', 'DstRate', 'Loss', 'pLoss', 'QLoss',
            # Window and state features
            'State', 'sWin', 'dWin', 'sMaxPktSz', 'dMaxPktSz',
            # Inter-arrival time features  
            'SrcJitter', 'DstJitter', 'ActiveMin', 'ActiveMax',
            # Connection patterns
            'IdleMin', 'IdleMax', 'ActiveMean', 'IdleMean'
        ]
        
        self.logger = get_logger()
        
    @monitor_performance
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[np.ndarray] = None
    ) -> 'FeatureProcessor':
        """
        Fit feature selection pipeline
        
        Args:
            X: Feature matrix (samples × features)
            y: Target labels (optional, for supervised selection)
        
        Returns:
            Self for method chaining
        """
        self.logger.logger.info(f"Fitting feature processor with method '{self.method}'")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        original_features = list(X.columns)
        self.logger.logger.info(f"Original feature count: {len(original_features)}")
        
        # Check cache (include temporal config in cache key)
        temporal_suffix = ""
        if self.temporal_enabled:
            temporal_suffix = f"_temporal_{self.temporal_method}_{self.log_bins}_{self.fourier_dim}"
        cache_path = self.cache_dir / f"feature_selection_{self.method}_{self.target_dim}{temporal_suffix}.pkl"
        if self.save_cache and cache_path.exists():
            self.logger.logger.info("Loading cached feature selection")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.selected_features = cache_data['selected_features']
                self.feature_importance = cache_data.get('feature_importance', {})
                self.scaler = cache_data.get('scaler', StandardScaler())
                # Restore temporal encoding components
                self.log_bin_edges = cache_data.get('log_bin_edges', None)
                self.fourier_weights = cache_data.get('fourier_weights', None)
            return self
        
        # Apply temporal encoding first (before feature selection)
        X_processed = self.encode_temporal_features(X.copy())
        
        # Update feature tracking
        if self.temporal_enabled:
            self.logger.logger.info(f"Feature count after temporal encoding: {len(X_processed.columns)}")
        
        # Apply feature selection based on method
        feature_mask = np.ones(len(X_processed.columns), dtype=bool)
        
        if self.method in ['variance', 'hybrid']:
            X_processed, feature_mask = self._apply_variance_selection(X_processed, feature_mask)
            
        if self.method in ['correlation', 'hybrid']:
            X_processed, feature_mask = self._apply_correlation_selection(X_processed, feature_mask)
            
        if self.method in ['domain', 'hybrid']:
            X_processed, feature_mask = self._apply_domain_selection(X_processed, feature_mask)
            
        # Final K-best selection if needed
        if len(X_processed.columns) > self.target_dim and y is not None:
            self.logger.logger.info(f"Applying K-best selection: {len(X_processed.columns)} → {self.target_dim}")
            self.kbest_selector = SelectKBest(f_classif, k=self.target_dim)
            X_processed = pd.DataFrame(
                self.kbest_selector.fit_transform(X_processed, y),
                columns=X_processed.columns[self.kbest_selector.get_support()]
            )
        
        # Store selected features
        self.selected_features = list(X_processed.columns)
        self.logger.logger.info(f"Final feature count: {len(self.selected_features)}")
        
        # Fit scaler
        self.scaler.fit(X_processed)
        
        # Calculate feature importance
        self._calculate_feature_importance(X_processed, y)
        
        # Save cache
        if self.save_cache:
            cache_data = {
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
                'scaler': self.scaler,
                'method': self.method,
                'target_dim': self.target_dim,
                'temporal_config': self.temporal_config,
                'log_bin_edges': self.log_bin_edges,
                'fourier_weights': self.fourier_weights
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            self.logger.logger.info(f"Feature selection cached to {cache_path}")
        
        return self
    
    def _apply_variance_selection(
        self, 
        X: pd.DataFrame, 
        feature_mask: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply variance-based feature selection"""
        self.logger.logger.info(f"Applying variance threshold: {self.variance_threshold}")
        
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_selected = pd.DataFrame(
            self.variance_selector.fit_transform(X),
            columns=X.columns[self.variance_selector.get_support()]
        )
        
        # Update feature mask
        support_mask = self.variance_selector.get_support()
        feature_mask = feature_mask & support_mask
        
        removed_count = len(X.columns) - len(X_selected.columns)
        self.logger.logger.info(f"Variance selection removed {removed_count} features")
        
        return X_selected, feature_mask
    
    def _apply_correlation_selection(
        self, 
        X: pd.DataFrame, 
        feature_mask: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply correlation-based feature selection"""
        self.logger.logger.info(f"Applying correlation threshold: {self.correlation_threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to remove
        to_remove = set()
        for col in upper_tri.columns:
            correlated_features = upper_tri.index[upper_tri[col] > self.correlation_threshold].tolist()
            to_remove.update(correlated_features)
        
        # Keep features not in removal set
        features_to_keep = [col for col in X.columns if col not in to_remove]
        X_selected = X[features_to_keep]
        
        self.logger.logger.info(f"Correlation selection removed {len(to_remove)} features")
        
        return X_selected, feature_mask
    
    def _apply_domain_selection(
        self, 
        X: pd.DataFrame, 
        feature_mask: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply domain knowledge-based feature selection (5G security whitelist)"""
        self.logger.logger.info("Applying 5G security domain knowledge selection")
        
        # Get 5G security feature whitelist
        security_whitelist = self._get_security_whitelist()
        
        # Prioritize security-relevant features
        available_security_features = [f for f in security_whitelist if f in X.columns]
        other_features = [f for f in X.columns if f not in security_whitelist]
        
        self.logger.logger.info(f"Found {len(available_security_features)} security features in data")
        
        # Select top security features and fill remaining with others
        if len(available_security_features) >= self.target_dim:
            selected_features = available_security_features[:self.target_dim]
        else:
            # Prioritize security features, then add high-variance others
            remaining_count = self.target_dim - len(available_security_features)
            # Sort other features by variance (descending)
            other_variances = [(f, X[f].var()) for f in other_features]
            other_variances.sort(key=lambda x: x[1], reverse=True)
            top_other_features = [f for f, _ in other_variances[:remaining_count]]
            
            selected_features = available_security_features + top_other_features
        
        X_selected = X[selected_features]
        
        self.logger.logger.info(f"Domain selection kept {len(available_security_features)} security features, {len(selected_features) - len(available_security_features)} additional features")
        
        return X_selected, feature_mask
    
    def _get_security_whitelist(self) -> List[str]:
        """Get 5G network security feature whitelist"""
        return [
            # Core flow metrics
            'TotPkts', 'TotBytes', 'SrcPkts', 'DstPkts', 
            # Protocol and service indicators
            'Proto', 'Sport', 'Dport', 'sTos', 'dTos',
            # Temporal patterns (critical for attack detection)
            'Dur', 'StartTime', 'LastTime',
            # Statistical aggregations (anomaly detection)
            'Mean', 'Sum', 'Min', 'Max', 'Std',
            # Rate and throughput metrics
            'Rate', 'SrcRate', 'DstRate', 
            # Packet size patterns
            'SrcBytes', 'DstBytes', 'sMaxPktSz', 'dMaxPktSz',
            # Connection state and window features
            'State', 'sWin', 'dWin',
            # Jitter and timing features (DDoS indicators)
            'SrcJitter', 'DstJitter',
            # Activity patterns (APT detection)
            'ActiveMin', 'ActiveMax', 'ActiveMean',
            'IdleMin', 'IdleMax', 'IdleMean',
            # Loss indicators (QoS attacks)
            'Loss', 'pLoss', 'QLoss'
        ]
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: Optional[np.ndarray]):
        """Calculate feature importance scores"""
        if y is None:
            # Use variance as importance
            self.feature_importance = {
                col: float(X[col].var()) for col in X.columns
            }
        else:
            # Use F-score as importance
            from sklearn.feature_selection import f_classif
            f_scores, _ = f_classif(X, y)
            self.feature_importance = {
                col: float(score) for col, score in zip(X.columns, f_scores)
            }
    
    def encode_temporal_features(
        self, 
        data: pd.DataFrame,
        time_cols: Optional[List[str]] = None,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Encode temporal features using log-bin or random fourier methods
        
        Args:
            data: Input DataFrame with time columns
            time_cols: List of time column names to encode
            method: Encoding method ('log_bin' or 'random_fourier')
            
        Returns:
            DataFrame with additional temporal features
        """
        if not self.temporal_enabled:
            return data
        
        method = method or self.temporal_method
        time_cols = time_cols or self.time_features
        
        self.logger.logger.info(f"Encoding temporal features using method '{method}'")
        
        # Validate time columns exist
        available_cols = [col for col in time_cols if col in data.columns]
        if not available_cols:
            self.logger.logger.warning(f"No temporal columns found in data: {time_cols}")
            return data
        
        data_encoded = data.copy()
        
        # Calculate time deltas
        time_deltas = self._calculate_time_deltas(data_encoded, available_cols)
        
        # Apply encoding method
        if method == "log_bin":
            encoded_features = self._log_bin_encode(time_deltas)
            feature_names = [f"temporal_logbin_{i}" for i in range(encoded_features.shape[1])]
        elif method == "random_fourier":
            encoded_features = self._random_fourier_encode(time_deltas)
            feature_names = [f"temporal_fourier_{i}" for i in range(encoded_features.shape[1])]
        else:
            raise ValueError(f"Unknown temporal encoding method: {method}")
        
        # Add encoded features to DataFrame
        encoded_df = pd.DataFrame(encoded_features, index=data_encoded.index, columns=feature_names)
        data_encoded = pd.concat([data_encoded, encoded_df], axis=1)
        
        self.logger.logger.info(f"Added {len(feature_names)} temporal features")
        
        return data_encoded
    
    def _calculate_time_deltas(self, data: pd.DataFrame, time_cols: List[str]) -> np.ndarray:
        """Calculate time deltas from time columns"""
        deltas = []
        
        # Duration delta (if available)
        if 'Dur' in time_cols and 'Dur' in data.columns:
            dur_values = pd.to_numeric(data['Dur'], errors='coerce').fillna(0)
            deltas.append(dur_values.values)
        
        # Start-End time delta
        if 'StartTime' in time_cols and 'LastTime' in time_cols:
            if 'StartTime' in data.columns and 'LastTime' in data.columns:
                start_vals = pd.to_numeric(data['StartTime'], errors='coerce').fillna(0)
                end_vals = pd.to_numeric(data['LastTime'], errors='coerce').fillna(0)
                time_diff = end_vals - start_vals
                deltas.append(time_diff.values)
        
        # Inter-arrival time delta (consecutive StartTime differences)
        if 'StartTime' in time_cols and 'StartTime' in data.columns:
            start_vals = pd.to_numeric(data['StartTime'], errors='coerce').fillna(0)
            inter_arrival = start_vals.diff().fillna(0)
            deltas.append(inter_arrival.values)
        
        if not deltas:
            # Fallback: use zeros
            self.logger.logger.warning("No valid time deltas calculated, using zeros")
            return np.zeros((len(data), 1))
        
        # Stack deltas into matrix
        time_deltas = np.column_stack(deltas)
        
        # Handle invalid values
        time_deltas = np.nan_to_num(time_deltas, nan=0.0, posinf=self.max_delta, neginf=0.0)
        time_deltas = np.abs(time_deltas)  # Ensure non-negative
        
        return time_deltas
    
    def _log_bin_encode(self, time_deltas: np.ndarray) -> np.ndarray:
        """
        Encode time deltas using logarithmic binning
        
        Args:
            time_deltas: Time delta matrix (samples × delta_types)
            
        Returns:
            Log-bin encoded features (samples × log_bins)
        """
        # Clip to valid range
        clipped_deltas = np.clip(time_deltas, self.min_delta, self.max_delta)
        
        # Log transform
        log_deltas = np.log(clipped_deltas + 1e-8)
        
        # Create log bin edges if not fitted
        if self.log_bin_edges is None:
            log_min = np.log(self.min_delta + 1e-8)
            log_max = np.log(self.max_delta + 1e-8)
            self.log_bin_edges = np.linspace(log_min, log_max, self.log_bins + 1)
        
        # Bin each delta type and aggregate
        encoded_features = []
        for delta_idx in range(time_deltas.shape[1]):
            # Digitize into bins
            bin_indices = np.digitize(log_deltas[:, delta_idx], self.log_bin_edges)
            bin_indices = np.clip(bin_indices - 1, 0, self.log_bins - 1)  # Convert to 0-based
            
            # One-hot encode
            one_hot = np.eye(self.log_bins)[bin_indices]
            encoded_features.append(one_hot)
        
        # Concatenate all delta types
        if len(encoded_features) > 1:
            result = np.concatenate(encoded_features, axis=1)
        else:
            result = encoded_features[0]
        
        return result
    
    def _random_fourier_encode(self, time_deltas: np.ndarray) -> np.ndarray:
        """
        Encode time deltas using random Fourier features
        
        Args:
            time_deltas: Time delta matrix (samples × delta_types)
            
        Returns:
            Random Fourier encoded features (samples × fourier_dim)
        """
        # Generate random weights if not fitted
        if self.fourier_weights is None:
            # Random frequencies ~ N(0, 1/sigma^2)
            self.fourier_weights = np.random.normal(
                0, 1.0 / self.fourier_sigma, 
                size=(time_deltas.shape[1], self.fourier_dim // 2)
            )
        
        # Project time deltas onto random frequencies
        projections = time_deltas @ self.fourier_weights
        
        # Apply trigonometric functions
        cos_features = np.cos(projections)
        sin_features = np.sin(projections)
        
        # Concatenate cos and sin features
        fourier_features = np.concatenate([cos_features, sin_features], axis=1)
        
        # Normalize by sqrt(feature_dim)
        fourier_features = fourier_features / np.sqrt(self.fourier_dim)
        
        return fourier_features
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform features using fitted selection and scaling
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features is None:
            raise ValueError("FeatureProcessor not fitted. Call fit() first.")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Apply temporal encoding if enabled
        X_processed = self.encode_temporal_features(X)
        
        # Select features (temporal features may have been added)
        available_features = [f for f in self.selected_features if f in X_processed.columns]
        if len(available_features) != len(self.selected_features):
            self.logger.logger.warning(
                f"Some selected features not found in data: "
                f"{set(self.selected_features) - set(available_features)}"
            )
        
        X_selected = X_processed[available_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get selected feature names"""
        if self.selected_features is None:
            raise ValueError("FeatureProcessor not fitted")
        return self.selected_features.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("FeatureProcessor not fitted")
        return self.feature_importance.copy()
    
    def save_feature_report(self, output_path: Union[str, Path]):
        """Save feature selection report"""
        if self.selected_features is None:
            raise ValueError("FeatureProcessor not fitted")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'method': self.method,
            'original_dim': len(self.domain_features) if hasattr(self, 'domain_features') else 'unknown',
            'target_dim': self.target_dim,
            'final_dim': len(self.selected_features),
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'parameters': {
                'variance_threshold': self.variance_threshold,
                'correlation_threshold': self.correlation_threshold
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.logger.info(f"Feature selection report saved to {output_path}")


class RiskVectorizer:
    """Risk score vectorization and normalization"""
    
    def __init__(
        self,
        num_bins: int = 8,
        embed_dim: int = 16,
        normalization: str = "zscore_sigmoid",
        adaptive_binning: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            num_bins: Number of risk discretization bins
            embed_dim: Risk embedding dimension
            normalization: Normalization method ('zscore_sigmoid', 'minmax', 'quantile')
            adaptive_binning: Whether to use adaptive quantile-based binning
            cache_dir: Cache directory path
        """
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        self.normalization = normalization
        self.adaptive_binning = adaptive_binning
        self.cache_dir = Path(cache_dir or "data/risk_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fitted parameters
        self.risk_mean = None
        self.risk_std = None
        self.risk_min = None
        self.risk_max = None
        self.quantiles = None
        self.bin_edges = None
        
        # Embedding layer
        self.risk_embedding = torch.nn.Embedding(num_bins, embed_dim)
        torch.nn.init.xavier_uniform_(self.risk_embedding.weight)
        
        self.logger = get_logger()
    
    @monitor_performance  
    def fit(self, risk_scores: np.ndarray) -> 'RiskVectorizer':
        """
        Fit risk normalization and 8-bin discretization parameters
        
        Args:
            risk_scores: Raw Risk-PPR scores array
            
        Returns:
            Self for method chaining
        """
        self.logger.logger.info(f"Fitting risk vectorizer with {len(risk_scores)} samples using {self.normalization} normalization")
        
        # Validate and clean input
        if not isinstance(risk_scores, np.ndarray):
            risk_scores = np.array(risk_scores)
        
        # Remove invalid values (NaN, inf)
        valid_mask = np.isfinite(risk_scores) & (risk_scores >= 0)
        valid_scores = risk_scores[valid_mask]
        
        if len(valid_scores) == 0:
            raise ValueError("No valid risk scores found")
        
        if len(valid_scores) < self.num_bins:
            self.logger.logger.warning(f"Only {len(valid_scores)} valid scores for {self.num_bins} bins")
        
        # Calculate normalization parameters based on method
        self.risk_mean = float(np.mean(valid_scores))
        self.risk_std = float(np.std(valid_scores)) 
        self.risk_min = float(np.min(valid_scores))
        self.risk_max = float(np.max(valid_scores))
        
        # Ensure non-zero std for z-score normalization
        if self.risk_std < 1e-6:
            self.risk_std = 1.0
            self.logger.logger.warning("Risk scores have near-zero variance, using std=1.0")
        
        self.logger.logger.info(f"Risk statistics: mean={self.risk_mean:.4f}, std={self.risk_std:.4f}, min={self.risk_min:.4f}, max={self.risk_max:.4f}")
        
        # Calculate binning parameters
        if self.adaptive_binning:
            # Quantile-based adaptive binning for better distribution
            self.quantiles = np.quantile(valid_scores, np.linspace(0, 1, self.num_bins + 1))
            self.bin_edges = self.quantiles
            self.logger.logger.info("Using adaptive quantile-based binning")
        else:
            # Fixed binning on normalized range [0,1]
            normalized_scores = self._normalize_scores(valid_scores)
            self.bin_edges = np.linspace(0, 1, self.num_bins + 1)
            self.logger.logger.info("Using fixed binning on normalized range")
        
        # Ensure bin edges are strictly increasing
        if not np.all(np.diff(self.bin_edges) > 0):
            # Handle edge case where bin edges are not strictly increasing
            self.bin_edges = np.linspace(np.min(self.bin_edges), np.max(self.bin_edges), self.num_bins + 1)
            self.logger.logger.warning("Adjusted bin edges to ensure strict monotonicity")
        
        self.logger.logger.info(f"Risk bin edges ({self.num_bins} bins): {self.bin_edges}")
        
        # Validate bin distribution
        normalized_scores = self._normalize_scores(valid_scores)
        bin_counts = np.histogram(normalized_scores, bins=self.bin_edges)[0]
        self.logger.logger.info(f"Bin distribution: {bin_counts}")
        
        # Warn if bins are severely imbalanced
        min_count, max_count = np.min(bin_counts), np.max(bin_counts)
        if max_count > 0 and min_count / max_count < 0.1:
            self.logger.logger.warning(f"Imbalanced bin distribution: min={min_count}, max={max_count}")
        
        return self
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply normalization to risk scores"""
        if self.normalization == "zscore_sigmoid":
            # Z-score normalization followed by sigmoid
            z_scores = (scores - self.risk_mean) / (self.risk_std + 1e-8)
            normalized = 1 / (1 + np.exp(-z_scores))
        elif self.normalization == "minmax":
            # Min-max normalization to [0, 1]
            normalized = (scores - self.risk_min) / (self.risk_max - self.risk_min + 1e-8)
        elif self.normalization == "quantile":
            # Quantile normalization
            normalized = np.searchsorted(self.quantiles, scores) / len(self.quantiles)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        return np.clip(normalized, 0, 1)
    
    def _discretize_scores(self, normalized_scores: np.ndarray) -> np.ndarray:
        """Discretize normalized scores into bins"""
        # Use searchsorted for efficient binning
        bin_indices = np.searchsorted(self.bin_edges[1:], normalized_scores)
        
        # Clip to valid range
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        return bin_indices
    
    def transform(self, risk_scores: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Transform risk scores to normalized values and embeddings
        
        Args:
            risk_scores: Raw risk scores
            
        Returns:
            Tuple of (normalized_scores, risk_embeddings)
        """
        if self.risk_mean is None:
            raise ValueError("RiskVectorizer not fitted. Call fit() first.")
        
        # Handle invalid values
        valid_mask = np.isfinite(risk_scores)
        normalized_scores = np.zeros_like(risk_scores)
        
        # Normalize valid scores
        if np.any(valid_mask):
            normalized_scores[valid_mask] = self._normalize_scores(risk_scores[valid_mask])
        
        # Discretize for embeddings
        bin_indices = self._discretize_scores(normalized_scores)
        
        # Convert to embeddings
        bin_tensor = torch.from_numpy(bin_indices).long()
        risk_embeddings = self.risk_embedding(bin_tensor)
        
        return normalized_scores, risk_embeddings
    
    def fit_transform(self, risk_scores: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Fit and transform in one step"""
        return self.fit(risk_scores).transform(risk_scores)
    
    def get_bin_edges(self) -> np.ndarray:
        """Get risk binning edges"""
        if self.bin_edges is None:
            raise ValueError("RiskVectorizer not fitted")
        return self.bin_edges.copy()
    
    def save_risk_report(self, output_path: Union[str, Path]):
        """Save risk vectorization report"""
        if self.risk_mean is None:
            raise ValueError("RiskVectorizer not fitted")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'normalization': self.normalization,
            'num_bins': self.num_bins,
            'embed_dim': self.embed_dim,
            'adaptive_binning': self.adaptive_binning,
            'statistics': {
                'risk_mean': self.risk_mean,
                'risk_std': self.risk_std,
                'risk_min': self.risk_min,
                'risk_max': self.risk_max
            },
            'bin_edges': self.bin_edges.tolist() if self.bin_edges is not None else None,
            'quantiles': self.quantiles.tolist() if self.quantiles is not None else None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.logger.info(f"Risk vectorization report saved to {output_path}") 