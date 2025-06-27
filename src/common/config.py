"""
Multi-layer Configuration Management for RiskNICLP-5G

Supports configuration from:
1. YAML files (base configuration)
2. Command line arguments (override YAML)
3. Environment variables (highest priority)

Configuration hierarchy: ENV > CLI > YAML > Defaults
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, fields
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # 异质图节点特征维度配置 (方案B)
    node_feature_dims: Dict[str, int] = field(default_factory=lambda: {
        'Alert': 9,   # Alert节点固定9维 (修复后)
        'Flow': 17,   # Flow节点17维 (原始维度)  
        'UE': 16      # UE节点16维 (原始维度)
    })
    
    # 传统配置字段 (向后兼容)
    input_dim: int = 112
    feature_dim: int = 45
    hidden_dim: int = 64
    risk_embed_dim: int = 16
    num_risk_bins: int = 8
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    temperature: float = 0.1
    gravity_epsilon: float = 1e-3


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    accumulate_steps: int = 4
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    direction_penalty_weight: float = 2.0
    
    # Progressive training phases
    basic_epochs: int = 5
    risk_aware_epochs: int = 10
    basic_auc_threshold: float = 0.80
    risk_recall_threshold: float = 0.75
    
    # Optimization
    enable_bf16: bool = True
    enable_gradient_checkpointing: bool = True
    gradient_clip_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = "cosine"
    warmup_epochs: int = 2
    eta_min_ratio: float = 0.01


@dataclass
class DataConfig:
    """Data processing configuration"""
    graph_dir: str = "data/processed/graphs"
    cache_dir: str = "data/csr_cache"
    metadata_dir: str = "data/processed/metadata"
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Feature processing
    feature_selection_method: str = "hybrid"
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.9
    
    # Risk vectorization
    risk_norm_method: str = "zscore_sigmoid"
    adaptive_binning: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: [
        'auc', 'aupr', 'directional_hit_100', 'mttd_30s',
        'risk_weighted_precision', 'risk_weighted_recall'
    ])
    save_predictions: bool = True
    save_attention_weights: bool = True
    visualization: bool = True
    confidence_interval: bool = True
    statistical_test: str = "bootstrap"  # or "t_test"
    n_bootstrap: int = 1000


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # Experiment tracking
    enable_wandb: bool = True
    enable_tensorboard: bool = True
    wandb_project: str = "riskniclp-5g"
    wandb_entity: Optional[str] = None
    
    # Monitoring
    log_system_metrics: bool = True
    system_metrics_interval: int = 100  # steps
    save_checkpoints: bool = True
    checkpoint_interval: int = 500  # steps


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    enable_data_anonymization: bool = True
    hash_sensitive_fields: bool = True
    log_data_access: bool = True
    max_log_retention_days: int = 30


@dataclass
class Phase2Config:
    """Phase 2 enhancement configuration"""
    # Risk-Bias Attention
    risk_bias_attention: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'simplified',
        'beta_1': 0.5,
        'beta_2': 0.5,
        'temperature': 0.1
    })
    
    # Negative Sampling
    negative_sampling: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'type': 'baseline',
        'strategy': 'stratified',
        'neg_ratio': 3,
        'risk_thresholds': [0.8, 0.5, 0.2],
        'hard_ratio': 0.3,
        'seed': 42
    })
    
    # Temporal Prediction
    temporal_prediction: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'hetero_lstm',
        'window_size': 5,
        'stride': 1,
        'future_steps': 3,
        'lead_time_targets': [5, 10, 15],
        'hetero_lstm': {
            'spatial_encoder_reuse': True,
            'graph_aggregation': 'mean_pool',
            'lstm_layers': 1,
            'bidirectional': False
        },
        'tgat_risk': {
            'edge_level_modeling': True,
            'risk_bias_temperature': 0.1,
            'num_heads': 4,
            'time_encoding_dim': 16
        },
        'graph_rnn': {
            'node_level_generation': True,
            'rnn_type': 'GRU',
            'message_passing_steps': 3
        }
    })
    
    # Cross-Scale Context
    cross_scale_context: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'fusion_method': 'gate'
    })
    
    # Training Enhancements
    training_enhancements: Dict[str, Any] = field(default_factory=lambda: {
        'progressive': True,
        'base_epochs': 20,
        'risk_epochs': 30,
        'scheduler': 'cosine',
        'early_stopping': {
            'enabled': True,
            'patience': 10,
            'min_delta': 0.001
        }
    })


@dataclass
class RiskNICLPConfig:
    """Complete RiskNICLP-5G configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    
    # Runtime
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 8
    
    # Experiment
    experiment_name: Optional[str] = None
    experiment_tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Validate data split ratios
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
            
        # Validate directories
        if not Path(self.data.graph_dir).exists():
            raise ValueError(f"Graph directory does not exist: {self.data.graph_dir}")
            
        # Create cache directory if it doesn't exist
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="RiskNICLP-5G Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/baseline_config.yaml",
        help="Path to configuration file"
    )
    
    # Override common parameters
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, help="Maximum epochs")
    
    # Experiment settings
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--notes", type=str, help="Experiment notes")
    parser.add_argument("--tags", nargs="*", help="Experiment tags")
    
    # Monitoring
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Data
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    
    return parser


def load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables"""
    env_overrides = {}
    
    # Define environment variable mappings
    env_mappings = {
        'RISKNICLP_SEED': 'seed',
        'RISKNICLP_DEVICE': 'device',
        'RISKNICLP_BATCH_SIZE': ('training', 'batch_size'),
        'RISKNICLP_LEARNING_RATE': ('training', 'learning_rate'),
        'RISKNICLP_MAX_EPOCHS': ('training', 'max_epochs'),
        'RISKNICLP_DATA_DIR': ('data', 'graph_dir'),
        'RISKNICLP_CACHE_DIR': ('data', 'cache_dir'),
        'RISKNICLP_LOG_LEVEL': ('logging', 'log_level'),
        'RISKNICLP_WANDB_PROJECT': ('logging', 'wandb_project'),
        'RISKNICLP_WANDB_ENTITY': ('logging', 'wandb_entity'),
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if isinstance(config_path, tuple):
                # Nested configuration
                section, key = config_path
                if section not in env_overrides:
                    env_overrides[section] = {}
                env_overrides[section][key] = _convert_env_value(value)
            else:
                # Top-level configuration
                env_overrides[config_path] = _convert_env_value(value)
                
    return env_overrides


def _convert_env_value(value: str) -> Union[str, int, float, bool]:
    """Convert environment variable string to appropriate type"""
    # Boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String
    return value


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    merged = {}
    
    for config in configs:
        if config:
            merged = OmegaConf.merge(merged, config)
    
    return OmegaConf.to_container(merged, resolve=True)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    cli_args: Optional[List[str]] = None,
    include_env: bool = True
) -> RiskNICLPConfig:
    """
    Load configuration with multi-layer override support
    
    Priority order (highest to lowest):
    1. Environment variables
    2. Command line arguments  
    3. YAML configuration file
    4. Default values
    """
    
    # 1. Start with defaults (handled by dataclass defaults)
    config_dict = {}
    
    # 2. Load YAML configuration
    if config_path:
        yaml_config = load_yaml_config(config_path)
        config_dict = merge_configs(config_dict, yaml_config)
    
    # 3. Parse CLI arguments
    parser = create_cli_parser()
    if cli_args is not None:
        args = parser.parse_args(cli_args)
    else:
        args = parser.parse_args()
    
    # Convert CLI args to config dict
    cli_config = {}
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            # Convert kebab-case to snake_case and nested structure
            key = key.replace('-', '_')
            if key in ['batch_size', 'learning_rate', 'max_epochs']:
                cli_config.setdefault('training', {})[key] = value
            elif key in ['data_dir']:
                cli_config.setdefault('data', {})['graph_dir'] = value
            elif key in ['cache_dir']:
                cli_config.setdefault('data', {})[key] = value
            elif key in ['log_level', 'no_wandb', 'no_tensorboard']:
                logging_config = cli_config.setdefault('logging', {})
                if key == 'no_wandb':
                    logging_config['enable_wandb'] = False
                elif key == 'no_tensorboard':
                    logging_config['enable_tensorboard'] = False
                else:
                    logging_config[key] = value
            elif key == 'tags':
                cli_config['experiment_tags'] = value or []
            else:
                cli_config[key] = value
    
    config_dict = merge_configs(config_dict, cli_config)
    
    # 4. Apply environment variable overrides
    if include_env:
        env_config = load_env_overrides()
        config_dict = merge_configs(config_dict, env_config)
    
    # 5. Create final configuration object
    omega_config = OmegaConf.create(config_dict)
    
    # Convert to our dataclass structure
    try:
        config = OmegaConf.to_object(omega_config)
        if not isinstance(config, RiskNICLPConfig):
            # Manual conversion if OmegaConf doesn't handle dataclass conversion
            config = RiskNICLPConfig(**config_dict)
    except Exception:
        # Fallback to manual creation
        config = RiskNICLPConfig()
        
        # Apply nested configurations
        for section_name in ['model', 'training', 'data', 'evaluation', 'logging', 'security']:
            if section_name in config_dict:
                section_config = config_dict[section_name]
                section_obj = getattr(config, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        # Apply top-level configurations
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'evaluation', 'logging', 'security']:
                if hasattr(config, key):
                    setattr(config, key, value)
    
    return config


def save_config(config: RiskNICLPConfig, output_path: Union[str, Path]):
    """Save configuration to YAML file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = OmegaConf.structured(config)
    config_dict = OmegaConf.to_container(config_dict, resolve=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def print_config(config: RiskNICLPConfig):
    """Pretty print configuration"""
    config_dict = OmegaConf.structured(config)
    print("=" * 50)
    print("RiskNICLP-5G Configuration")
    print("=" * 50)
    print(OmegaConf.to_yaml(config_dict))
    print("=" * 50) 