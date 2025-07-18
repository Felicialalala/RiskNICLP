"""
内存监控工具
用于监控训练过程中的内存使用情况，防止OOM
"""

import psutil
import torch
import gc
import logging
from typing import Dict, Optional
import time


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 memory_threshold: float = 0.85):
        """
        初始化内存监控器
        
        Args:
            logger: 日志记录器
            memory_threshold: 内存使用阈值（0-1），超过此阈值会触发警告
        """
        self.logger = logger or logging.getLogger(__name__)
        self.memory_threshold = memory_threshold
        self.process = psutil.Process()
        
    def get_memory_info(self) -> Dict[str, float]:
        """获取当前内存使用信息"""
        # 系统内存
        system_memory = psutil.virtual_memory()
        
        # 进程内存
        process_memory = self.process.memory_info()
        
        # PyTorch内存（如果使用CUDA）
        torch_memory = {}
        if torch.cuda.is_available():
            torch_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        
        return {
            'system_total': system_memory.total / 1024**3,      # GB
            'system_used': system_memory.used / 1024**3,       # GB
            'system_available': system_memory.available / 1024**3,  # GB
            'system_percent': system_memory.percent,           # %
            'process_rss': process_memory.rss / 1024**3,       # GB
            'process_vms': process_memory.vms / 1024**3,       # GB
            **torch_memory
        }
    
    def check_memory_usage(self, context: str = "") -> bool:
        """
        检查内存使用情况
        
        Args:
            context: 上下文信息
            
        Returns:
            bool: True表示内存使用正常，False表示内存使用过高
        """
        memory_info = self.get_memory_info()
        
        # 检查系统内存使用率
        if memory_info['system_percent'] > self.memory_threshold * 100:
            # 兼容不同类型的logger
            if hasattr(self.logger, 'logger'):
                self.logger.logger.warning(
                    f"⚠️ 内存使用过高 [{context}]: "
                    f"系统内存使用率 {memory_info['system_percent']:.1f}%, "
                    f"进程内存 {memory_info['process_rss']:.2f}GB"
                )
            else:
                self.logger.warning(
                    f"⚠️ 内存使用过高 [{context}]: "
                    f"系统内存使用率 {memory_info['system_percent']:.1f}%, "
                    f"进程内存 {memory_info['process_rss']:.2f}GB"
                )
            return False
        
        # 记录正常的内存使用情况
        if context:
            # 兼容不同类型的logger
            if hasattr(self.logger, 'logger'):
                self.logger.logger.info(
                    f"💾 内存使用 [{context}]: "
                    f"系统 {memory_info['system_percent']:.1f}%, "
                    f"进程 {memory_info['process_rss']:.2f}GB"
                )
            else:
                self.logger.info(
                    f"💾 内存使用 [{context}]: "
                    f"系统 {memory_info['system_percent']:.1f}%, "
                    f"进程 {memory_info['process_rss']:.2f}GB"
                )
        
        return True
    
    def force_cleanup(self):
        """强制清理内存"""
        # 兼容不同类型的logger
        if hasattr(self.logger, 'logger'):
            self.logger.logger.info("🧹 开始强制内存清理...")
        else:
            self.logger.info("🧹 开始强制内存清理...")

        # Python垃圾回收
        collected = gc.collect()

        # PyTorch缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 兼容不同类型的logger
        if hasattr(self.logger, 'logger'):
            self.logger.logger.info(f"🧹 内存清理完成: 回收了 {collected} 个对象")
        else:
            self.logger.info(f"🧹 内存清理完成: 回收了 {collected} 个对象")
    
    def log_memory_summary(self):
        """记录内存使用摘要"""
        memory_info = self.get_memory_info()
        
        self.logger.info("📊 内存使用摘要:")
        self.logger.info(f"   系统内存: {memory_info['system_used']:.2f}GB / {memory_info['system_total']:.2f}GB ({memory_info['system_percent']:.1f}%)")
        self.logger.info(f"   进程内存: RSS={memory_info['process_rss']:.2f}GB, VMS={memory_info['process_vms']:.2f}GB")
        
        if torch.cuda.is_available():
            self.logger.info(f"   GPU内存: 已分配={memory_info.get('allocated', 0):.2f}GB, 已保留={memory_info.get('reserved', 0):.2f}GB")


class MemoryEfficientTrainingMixin:
    """内存高效训练混入类"""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self._last_memory_check = 0
        self._memory_check_interval = 10  # 每10个batch检查一次内存
    
    def check_memory_before_batch(self, batch_idx: int, context: str = ""):
        """在处理batch前检查内存"""
        if batch_idx % self._memory_check_interval == 0:
            if not self.memory_monitor.check_memory_usage(f"Batch {batch_idx} {context}"):
                # 内存使用过高，强制清理
                self.memory_monitor.force_cleanup()
                
                # 再次检查，如果仍然过高则抛出异常
                if not self.memory_monitor.check_memory_usage(f"After cleanup - Batch {batch_idx}"):
                    raise RuntimeError(f"内存使用过高，无法继续训练 (Batch {batch_idx})")
    
    def cleanup_after_epoch(self, epoch: int):
        """在epoch结束后清理内存"""
        self.memory_monitor.force_cleanup()
        self.memory_monitor.log_memory_summary()
        
        # 记录epoch内存使用情况
        memory_info = self.memory_monitor.get_memory_info()
        if hasattr(self, 'logger') and hasattr(self.logger, 'logger'):
            self.logger.logger.info(
                f"📊 Epoch {epoch} 内存使用: "
                f"系统 {memory_info['system_percent']:.1f}%, "
                f"进程 {memory_info['process_rss']:.2f}GB"
            )


def memory_efficient_decorator(func):
    """内存高效装饰器"""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        # 执行前检查内存
        monitor.check_memory_usage(f"Before {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 执行后清理内存
            monitor.force_cleanup()
            monitor.check_memory_usage(f"After {func.__name__}")
    
    return wrapper


# 全局内存监控实例
global_memory_monitor = MemoryMonitor()


def get_memory_usage_mb() -> float:
    """获取当前进程内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(logger: logging.Logger, context: str = ""):
    """记录内存使用情况"""
    memory_mb = get_memory_usage_mb()
    system_memory = psutil.virtual_memory()
    
    logger.info(
        f"💾 内存使用 [{context}]: "
        f"进程 {memory_mb:.1f}MB, "
        f"系统 {system_memory.percent:.1f}%"
    )


def check_memory_limit(limit_mb: float = 8000) -> bool:
    """检查内存使用是否超过限制"""
    current_mb = get_memory_usage_mb()
    return current_mb < limit_mb
