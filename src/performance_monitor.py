"""
性能监控模块
监控数据加载时间、内存使用效率和图构建质量指标
"""

import time
import psutil
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self):
        self.metrics = {
            'data_loading_times': [],
            'memory_usage': [],
            'graph_construction_times': [],
            'success_rates': [],
            'error_counts': 0,
            'total_processed': 0
        }
        self.start_time = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        logger.info("性能监控开始")
    
    def record_data_loading(self, loading_time: float, file_size: int):
        """记录数据加载性能"""
        self.metrics['data_loading_times'].append({
            'time': loading_time,
            'file_size': file_size,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_memory_usage(self):
        """记录内存使用情况"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        self.metrics['memory_usage'].append({
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'cpu_percent': cpu_percent,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_graph_construction(self, construction_time: float, 
                                num_nodes: int, num_edges: int, success: bool):
        """记录图构建性能"""
        self.metrics['graph_construction_times'].append({
            'time': construction_time,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            self.metrics['success_rates'].append(1)
        else:
            self.metrics['success_rates'].append(0)
            self.metrics['error_counts'] += 1
        
        self.metrics['total_processed'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'total_processing_time': total_time,
            'total_files_processed': self.metrics['total_processed'],
            'success_rate': (sum(self.metrics['success_rates']) / 
                           len(self.metrics['success_rates']) * 100) 
                           if self.metrics['success_rates'] else 0,
            'error_count': self.metrics['error_counts'],
            'avg_data_loading_time': (
                sum([m['time'] for m in self.metrics['data_loading_times']]) /
                len(self.metrics['data_loading_times'])
            ) if self.metrics['data_loading_times'] else 0,
            'avg_graph_construction_time': (
                sum([m['time'] for m in self.metrics['graph_construction_times']]) /
                len(self.metrics['graph_construction_times'])
            ) if self.metrics['graph_construction_times'] else 0,
            'peak_memory_usage': max([m['rss'] for m in self.metrics['memory_usage']], 
                                   default=0),
            'avg_cpu_usage': (
                sum([m['cpu_percent'] for m in self.metrics['memory_usage']]) /
                len(self.metrics['memory_usage'])
            ) if self.metrics['memory_usage'] else 0
        }
        
        return summary
    
    def save_metrics(self, filepath: str):
        """保存性能指标到文件"""
        metrics_data = {
            'summary': self.get_performance_summary(),
            'detailed_metrics': self.metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能指标已保存到: {filepath}")

def compare_performance_before_after(before_metrics: Dict, after_metrics: Dict) -> Dict:
    """比较优化前后的性能"""
    comparison = {
        'data_loading_improvement': {
            'before_avg': before_metrics.get('avg_data_loading_time', 0),
            'after_avg': after_metrics.get('avg_data_loading_time', 0),
            'improvement_pct': 0
        },
        'memory_usage_improvement': {
            'before_peak': before_metrics.get('peak_memory_usage', 0),
            'after_peak': after_metrics.get('peak_memory_usage', 0),
            'improvement_pct': 0
        },
        'success_rate_improvement': {
            'before_rate': before_metrics.get('success_rate', 0),
            'after_rate': after_metrics.get('success_rate', 0),
            'improvement_pct': 0
        }
    }
    
    # 计算改进百分比
    for metric in comparison:
        before_val = comparison[metric]['before_avg'] or comparison[metric].get('before_peak', 0) or comparison[metric].get('before_rate', 0)
        after_val = comparison[metric]['after_avg'] or comparison[metric].get('after_peak', 0) or comparison[metric].get('after_rate', 0)
        
        if before_val > 0:
            improvement = ((before_val - after_val) / before_val) * 100
            comparison[metric]['improvement_pct'] = improvement
    
    return comparison 