#!/usr/bin/env python3
"""
RiskNICLP-5G 内存优化训练脚本
专门处理大规模攻击预测数据的内存问题
"""

import os
import sys
import gc
import torch
import psutil
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_memory_optimization():
    """设置内存优化"""
    # 设置PyTorch内存管理
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # 启用内存映射
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 设置环境变量以优化内存使用
    os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
    os.environ['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数
    
    # 强制垃圾回收
    gc.collect()

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    print(f"内存使用: RSS={memory_info.rss / 1024**3:.2f}GB, "
          f"VMS={memory_info.vms / 1024**3:.2f}GB, "
          f"百分比={memory_percent:.1f}%")
    
    return memory_info.rss / 1024**3  # 返回RSS内存使用(GB)

def force_memory_cleanup():
    """强制内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_memory_limit(max_memory_gb=8.0):
    """检查内存使用是否超过限制"""
    current_memory = monitor_memory()
    if current_memory > max_memory_gb:
        print(f"⚠️ 内存使用超过限制 ({current_memory:.2f}GB > {max_memory_gb}GB)")
        force_memory_cleanup()
        return False
    return True

def run_memory_optimized_training():
    """运行内存优化的训练"""
    print("🚀 开始内存优化的攻击链路预测训练...")
    
    # 设置内存优化
    setup_memory_optimization()
    
    # 监控初始内存
    print("📊 初始内存状态:")
    monitor_memory()
    
    try:
        # 导入训练模块
        from src.models.training import RiskNICLPTrainer
        from src.config.config_loader import load_config
        from src.utils.logger import setup_logger
        
        print("✅ 模块导入成功")
        monitor_memory()
        
        # 加载配置
        config = load_config()
        print("✅ 配置加载成功")
        monitor_memory()
        
        # 设置日志
        logger = setup_logger()
        print("✅ 日志设置成功")
        monitor_memory()
        
        # 检查内存限制
        if not check_memory_limit(6.0):  # 6GB限制
            print("❌ 内存使用过高，训练终止")
            return False
        
        # 创建训练器
        print("🔧 创建训练器...")
        trainer = RiskNICLPTrainer(config, logger)
        print("✅ 训练器创建成功")
        monitor_memory()
        
        # 检查内存限制
        if not check_memory_limit(8.0):  # 8GB限制
            print("❌ 训练器创建后内存使用过高")
            return False
        
        # 开始训练
        print("🎯 开始训练...")
        model = trainer.train()
        
        print("✅ 训练完成")
        monitor_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 错误时强制清理内存
        force_memory_cleanup()
        return False

def run_lightweight_training():
    """运行轻量级训练（进一步减少内存使用）"""
    print("🚀 开始轻量级攻击链路预测训练...")
    
    # 更激进的内存优化
    setup_memory_optimization()
    
    # 设置更小的批次大小和样本数量
    os.environ['RISKNICLP_BATCH_SIZE'] = '2'  # 极小批次
    os.environ['RISKNICLP_MAX_SAMPLES'] = '100000'  # 限制样本数量
    os.environ['RISKNICLP_MEMORY_LIMIT'] = '4'  # 4GB内存限制
    
    try:
        # 直接运行训练脚本，但使用环境变量控制
        import subprocess
        
        result = subprocess.run([
            sys.executable, 'train.py'
        ], capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            print("✅ 轻量级训练完成")
            print("训练输出:")
            print(result.stdout[-1000:])  # 显示最后1000字符
            return True
        else:
            print("❌ 轻量级训练失败")
            print("错误输出:")
            print(result.stderr[-1000:])  # 显示最后1000字符错误
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 训练超时")
        return False
    except Exception as e:
        print(f"❌ 轻量级训练异常: {e}")
        return False

def main():
    """主函数"""
    print("🔍 RiskNICLP-5G 内存优化训练")
    
    # 检查系统内存
    total_memory = psutil.virtual_memory().total / 1024**3
    available_memory = psutil.virtual_memory().available / 1024**3
    
    print(f"系统内存: 总计={total_memory:.1f}GB, 可用={available_memory:.1f}GB")
    
    if available_memory < 8.0:
        print("⚠️ 可用内存不足8GB，将使用轻量级训练模式")
        success = run_lightweight_training()
    else:
        print("✅ 内存充足，尝试标准内存优化训练")
        success = run_memory_optimized_training()
        
        if not success:
            print("⚠️ 标准训练失败，回退到轻量级训练")
            success = run_lightweight_training()
    
    if success:
        print("🎉 训练成功完成")
        
        # 显示最终内存状态
        print("📊 最终内存状态:")
        monitor_memory()
        
        # 查找并显示结果
        results_dir = Path('results')
        if results_dir.exists():
            result_files = list(results_dir.glob('**/*.json'))
            if result_files:
                latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                print(f"📄 最新结果文件: {latest_result}")
                
                try:
                    import json
                    with open(latest_result, 'r') as f:
                        results = json.load(f)
                    
                    # 显示关键指标
                    key_metrics = ['attack_prediction_accuracy', 'early_warning_effectiveness', 
                                 'false_alarm_rate', 'accuracy', 'auc']
                    
                    print("🎯 关键性能指标:")
                    for metric in key_metrics:
                        if metric in results:
                            value = results[metric]
                            if isinstance(value, (int, float)):
                                print(f"  {metric}: {value:.4f}")
                            else:
                                print(f"  {metric}: {value}")
                                
                except Exception as e:
                    print(f"⚠️ 无法读取结果文件: {e}")
        
        return 0
    else:
        print("❌ 训练失败")
        return 1

if __name__ == "__main__":
    exit(main())
