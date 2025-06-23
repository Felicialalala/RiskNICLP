"""
Unified Logging and Monitoring System for RiskNICLP-5G

Supports:
- Local file logging with rotation
- WandB integration for experiment tracking
- TensorBoard integration for visualization
- Prometheus metrics export
- Runtime performance monitoring
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
import warnings

# Optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("WandB not installed. Experiment tracking disabled.")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    warnings.warn("TensorBoard not installed. TensorBoard logging disabled.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not installed. System monitoring disabled.")


class RiskNICLPLogger:
    """Enhanced logger with monitoring capabilities"""
    
    def __init__(
        self,
        name: str = "RiskNICLP-5G",
        log_dir: Union[str, Path] = "logs",
        level: int = logging.INFO,
        enable_wandb: bool = False,
        enable_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup basic logger
        self.logger = self._setup_file_logger(level)
        
        # Setup experiment tracking
        self.wandb_run = None
        self.tb_writer = None
        
        if enable_wandb and HAS_WANDB:
            self._setup_wandb(wandb_project, wandb_config)
            
        if enable_tensorboard and HAS_TENSORBOARD:
            self._setup_tensorboard()
            
        # Performance monitoring
        self.start_time = time.time()
        self.step_count = 0
        
    def _setup_file_logger(self, level: int) -> logging.Logger:
        """Setup file logger with rotation"""
        logger = logging.getLogger(self.name)
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _setup_wandb(self, project: Optional[str], config: Optional[Dict]):
        """Setup WandB experiment tracking"""
        try:
            self.wandb_run = wandb.init(
                project=project or "riskniclp-5g",
                config=config or {},
                dir=str(self.log_dir)
            )
            self.logger.info("WandB initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            
    def _setup_tensorboard(self):
        """Setup TensorBoard logging"""
        try:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            self.logger.info("TensorBoard initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all enabled backends"""
        if step is None:
            step = self.step_count
            self.step_count += 1
            
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
        
        # Log to WandB
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")
                
        # Log to TensorBoard
        if self.tb_writer:
            try:
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(key, value, step)
                self.tb_writer.flush()
            except Exception as e:
                self.logger.warning(f"Failed to log to TensorBoard: {e}")
                
    def log_system_metrics(self, step: Optional[int] = None):
        """Log system performance metrics"""
        if not HAS_PSUTIL:
            return
            
        try:
            system_metrics = {
                'system/cpu_percent': psutil.cpu_percent(),
                'system/memory_percent': psutil.virtual_memory().percent,
                'system/disk_usage': psutil.disk_usage('/').percent,
                'system/runtime_hours': (time.time() - self.start_time) / 3600
            }
            
            # Add GPU metrics if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    system_metrics.update({
                        'system/gpu_memory_percent': gpu.memoryUtil * 100,
                        'system/gpu_utilization': gpu.load * 100
                    })
            except ImportError:
                pass
                
            self.log_metrics(system_metrics, step)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
            
    def log_model_info(self, model, input_shape: Optional[tuple] = None):
        """Log model architecture information"""
        try:
            import torch
            
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info = {
                    'model/total_parameters': total_params,
                    'model/trainable_parameters': trainable_params,
                    'model/non_trainable_parameters': total_params - trainable_params
                }
                
                self.log_metrics(model_info)
                self.logger.info(f"Model info: {total_params:,} total parameters, {trainable_params:,} trainable")
                
                # Log model summary to WandB
                if self.wandb_run and input_shape:
                    try:
                        import torchinfo
                        summary = torchinfo.summary(model, input_shape, verbose=0)
                        self.wandb_run.log({"model_summary": str(summary)})
                    except ImportError:
                        self.logger.warning("torchinfo not available for model summary")
                        
        except Exception as e:
            self.logger.warning(f"Failed to log model info: {e}")
            
    def finish(self):
        """Clean up resources"""
        if self.wandb_run:
            self.wandb_run.finish()
            
        if self.tb_writer:
            self.tb_writer.close()
            
        self.logger.info("Logger cleanup completed")


# Global logger instance
_global_logger: Optional[RiskNICLPLogger] = None


def get_logger(
    name: str = "RiskNICLP-5G",
    **kwargs
) -> RiskNICLPLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = RiskNICLPLogger(name=name, **kwargs)
        
    return _global_logger


def setup_monitoring(
    enable_wandb: bool = True,
    enable_tensorboard: bool = True,
    wandb_project: str = "riskniclp-5g",
    wandb_config: Optional[Dict] = None,
    log_level: int = logging.INFO
) -> RiskNICLPLogger:
    """Setup comprehensive monitoring"""
    logger = get_logger(
        enable_wandb=enable_wandb,
        enable_tensorboard=enable_tensorboard,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        level=log_level
    )
    
    logger.logger.info("Monitoring setup completed")
    return logger


def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.log_metrics({
                f'performance/{func.__name__}_duration': duration
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
            
    return wrapper 