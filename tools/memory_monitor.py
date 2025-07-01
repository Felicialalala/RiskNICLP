#!/usr/bin/env python3
"""
RiskNICLP-5G 内存监控工具

实时监控训练过程中的内存使用情况，预防OOM问题
"""

import psutil
import time
import torch
import threading
import argparse
from pathlib import Path
import json
from datetime import datetime


class MemoryMonitor:
    """实时内存监控器"""
    
    def __init__(self, 
                 log_file: str = "memory_usage.log",
                 warning_threshold: float = 0.85,
                 critical_threshold: float = 0.95,
                 monitor_interval: float = 5.0):
        """
        初始化内存监控器
        
        Args:
            log_file: 内存使用日志文件
            warning_threshold: 内存使用警告阈值 (85%)
            critical_threshold: 内存使用危险阈值 (95%)
            monitor_interval: 监控间隔（秒）
        """
        self.log_file = log_file
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitor_interval = monitor_interval
        
        self.monitoring = False
        self.monitor_thread = None
        
        # 统计信息
        self.max_memory_usage = 0.0
        self.max_swap_usage = 0.0
        self.warning_count = 0
        self.critical_count = 0
        
    def start_monitoring(self):
        """开始内存监控"""
        if self.monitoring:
            return
            
        print(f"🔍 开始内存监控，间隔：{self.monitor_interval}秒")
        print(f"📊 警告阈值：{self.warning_threshold*100:.0f}%，危险阈值：{self.critical_threshold*100:.0f}%")
        print(f"📝 日志文件：{self.log_file}")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止内存监控"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        self._print_summary()
        
    def _monitor_loop(self):
        """监控主循环"""
        with open(self.log_file, 'w') as f:
            # 写入CSV头部
            f.write("timestamp,memory_percent,memory_used_gb,memory_total_gb,swap_percent,swap_used_gb,gpu_memory_gb,process_memory_gb,status\n")
            
            while self.monitoring:
                try:
                    memory_info = self._get_memory_info()
                    self._log_memory_info(f, memory_info)
                    self._check_thresholds(memory_info)
                    
                    time.sleep(self.monitor_interval)
                    
                except Exception as e:
                    print(f"⚠️ 内存监控错误: {e}")
                    time.sleep(self.monitor_interval)
                    
    def _get_memory_info(self):
        """获取系统内存信息"""
        # 系统内存
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 当前进程内存
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        # GPU内存（如果可用）
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            except:
                pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'swap_percent': swap.percent,
            'swap_used_gb': swap.used / (1024**3),
            'swap_total_gb': swap.total / (1024**3),
            'gpu_memory_gb': gpu_memory_gb,
            'process_memory_gb': process_memory_gb
        }
        
    def _log_memory_info(self, f, info):
        """记录内存信息到文件"""
        status = "normal"
        if info['memory_percent'] / 100.0 >= self.critical_threshold:
            status = "critical"
        elif info['memory_percent'] / 100.0 >= self.warning_threshold:
            status = "warning"
            
        # 更新统计
        self.max_memory_usage = max(self.max_memory_usage, info['memory_percent'])
        self.max_swap_usage = max(self.max_swap_usage, info['swap_percent'])
        
        # 写入CSV
        f.write(f"{info['timestamp']},{info['memory_percent']:.1f},{info['memory_used_gb']:.2f},"
                f"{info['memory_total_gb']:.2f},{info['swap_percent']:.1f},{info['swap_used_gb']:.2f},"
                f"{info['gpu_memory_gb']:.2f},{info['process_memory_gb']:.2f},{status}\n")
        f.flush()
        
    def _check_thresholds(self, info):
        """检查内存使用阈值"""
        memory_ratio = info['memory_percent'] / 100.0
        
        if memory_ratio >= self.critical_threshold:
            self.critical_count += 1
            print(f"🚨 内存使用危险: {info['memory_percent']:.1f}% "
                  f"(进程: {info['process_memory_gb']:.2f}GB, "
                  f"Swap: {info['swap_percent']:.1f}%)")
            
            # 建议清理操作
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        elif memory_ratio >= self.warning_threshold:
            self.warning_count += 1
            if self.warning_count % 10 == 1:  # 每10次警告打印一次
                print(f"⚠️ 内存使用警告: {info['memory_percent']:.1f}% "
                      f"(进程: {info['process_memory_gb']:.2f}GB)")
        
    def _print_summary(self):
        """打印监控摘要"""
        print("\n📊 内存监控摘要:")
        print(f"   最大内存使用: {self.max_memory_usage:.1f}%")
        print(f"   最大Swap使用: {self.max_swap_usage:.1f}%") 
        print(f"   警告次数: {self.warning_count}")
        print(f"   危险次数: {self.critical_count}")
        
        if self.critical_count > 0:
            print("🚨 检测到内存危险情况，建议优化训练配置")
        elif self.warning_count > 50:
            print("⚠️ 频繁内存警告，建议关注内存使用")
        else:
            print("✅ 内存使用正常")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RiskNICLP-5G 内存监控工具")
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='memory_usage.log',
        help='内存使用日志文件路径'
    )
    
    parser.add_argument(
        '--warning-threshold',
        type=float,
        default=0.85,
        help='内存使用警告阈值 (0.0-1.0)'
    )
    
    parser.add_argument(
        '--critical-threshold', 
        type=float,
        default=0.95,
        help='内存使用危险阈值 (0.0-1.0)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=5.0,
        help='监控间隔（秒）'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='监控持续时间（秒），不指定则持续监控'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建监控器
    monitor = MemoryMonitor(
        log_file=args.log_file,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold,
        monitor_interval=args.interval
    )
    
    try:
        # 开始监控
        monitor.start_monitoring()
        
        if args.duration:
            print(f"⏱️ 监控 {args.duration} 秒...")
            time.sleep(args.duration)
        else:
            print("🔄 持续监控中，按 Ctrl+C 停止...")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断监控")
    except Exception as e:
        print(f"❌ 监控失败: {e}")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main() 