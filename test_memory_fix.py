#!/usr/bin/env python3
"""
内存优化验证脚本

快速验证内存配置修复后的训练效果
"""

import subprocess
import sys
import time
import psutil
from pathlib import Path


def check_system_status():
    """检查系统状态"""
    print("🔍 系统状态检查:")
    
    # 内存状态
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"   💾 内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
    print(f"   💿 Swap: {swap.used/1024**3:.1f}GB / {swap.total/1024**3:.1f}GB ({swap.percent:.1f}%)")
    
    # Swap状态检查
    if swap.percent > 50:
        print(f"⚠️ Swap使用过高: {swap.percent:.1f}%，建议重启或清理内存")
        return False
    
    # 可用内存检查
    available_gb = memory.available / 1024**3
    if available_gb < 5:
        print(f"⚠️ 可用内存不足: {available_gb:.1f}GB，建议释放内存")
        return False
    
    print("✅ 系统状态正常")
    return True


def clear_cache():
    """清理缓存以减少内存占用"""
    print("🧹 清理系统缓存...")
    
    try:
        # 清理Python import缓存
        import importlib
        importlib.invalidate_caches()
        
        # 清理PyTorch缓存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 缓存清理完成")
        
    except Exception as e:
        print(f"⚠️ 缓存清理失败: {e}")


def run_memory_test():
    """运行内存优化测试"""
    print("🚀 开始内存优化训练测试...")
    print("📊 配置修改:")
    print("   - batch_size: 8 → 4")
    print("   - accumulate_steps: 1 → 2 (保持有效批次=8)")
    print("   - num_workers: 4 → 1")
    print("   - pin_memory: true → false")
    print("   - prefetch_factor: 2 → 1")
    
    # 启动内存监控
    monitor_cmd = [
        sys.executable, "tools/memory_monitor.py",
        "--log-file", "memory_test.log",
        "--warning-threshold", "0.80",
        "--critical-threshold", "0.90",
        "--interval", "3.0"
    ]
    
    print("🔍 启动内存监控...")
    try:
        monitor_process = subprocess.Popen(
            monitor_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # 等待监控器启动
        
        # 运行训练（限制3个epoch）
        train_cmd = [
            sys.executable, "train.py",
            "--config", "config/baseline_config.yaml"
        ]
        
        print("🏃 启动训练（预期运行3个epoch）...")
        train_process = subprocess.run(
            train_cmd,
            timeout=1800,  # 30分钟超时
            capture_output=True,
            text=True
        )
        
        # 停止监控
        monitor_process.terminate()
        
        # 分析结果
        if train_process.returncode == 0:
            print("✅ 训练成功完成！内存优化生效")
        else:
            print(f"❌ 训练失败，返回码: {train_process.returncode}")
            print("stderr:", train_process.stderr[-1000:])  # 最后1000字符
            
        return train_process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 训练超时，可能仍有内存问题")
        monitor_process.terminate()
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def analyze_memory_log():
    """分析内存使用日志"""
    log_file = Path("memory_test.log")
    if not log_file.exists():
        print("⚠️ 内存日志文件不存在")
        return
    
    print("📊 内存使用分析:")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[1:]  # 跳过头部
        
        if not lines:
            print("⚠️ 内存日志为空")
            return
        
        max_memory = 0.0
        max_swap = 0.0
        warning_count = 0
        critical_count = 0
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                memory_percent = float(parts[1])
                swap_percent = float(parts[4])
                status = parts[8]
                
                max_memory = max(max_memory, memory_percent)
                max_swap = max(max_swap, swap_percent)
                
                if status == 'warning':
                    warning_count += 1
                elif status == 'critical':
                    critical_count += 1
        
        print(f"   最大内存使用: {max_memory:.1f}%")
        print(f"   最大Swap使用: {max_swap:.1f}%")
        print(f"   警告次数: {warning_count}")
        print(f"   危险次数: {critical_count}")
        
        # 评估结果
        if critical_count > 0:
            print("🚨 仍有内存危险，需要进一步优化")
        elif warning_count > 10:
            print("⚠️ 内存使用较高，建议继续优化")
        elif max_memory < 70:
            print("✅ 内存使用优秀，优化成功")
        else:
            print("✅ 内存使用正常，优化有效")
            
    except Exception as e:
        print(f"❌ 日志分析失败: {e}")


def print_recommendations():
    """打印优化建议"""
    print("\n💡 进一步优化建议:")
    print("1. 如果仍有内存问题，可以:")
    print("   - 进一步减少batch_size到2")
    print("   - 增加accumulate_steps到4")
    print("   - 禁用某些缓存功能")
    
    print("\n2. 如果内存使用正常，可以:")
    print("   - 逐步增加batch_size到6或8")
    print("   - 重新启用pin_memory")
    print("   - 增加num_workers到2")
    
    print("\n3. 监控工具使用:")
    print("   - 训练时启动: python tools/memory_monitor.py")
    print("   - 查看缓存状态: python tools/cache_manager.py --action stats")


def main():
    """主函数"""
    print("🔧 RiskNICLP-5G 内存优化验证")
    print("=" * 50)
    
    # 1. 检查系统状态
    if not check_system_status():
        print("❌ 系统状态异常，请先解决系统问题")
        return
    
    # 2. 清理缓存
    clear_cache()
    
    # 3. 运行测试
    success = run_memory_test()
    
    # 4. 分析结果
    analyze_memory_log()
    
    # 5. 打印建议
    print_recommendations()
    
    if success:
        print("\n🎉 内存优化验证成功！")
    else:
        print("\n⚠️ 仍需进一步优化")


if __name__ == "__main__":
    main() 