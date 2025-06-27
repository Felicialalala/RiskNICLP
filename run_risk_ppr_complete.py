#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行完整的Risk-PPR计算脚本
处理所有75个图文件，为RiskNICLP-Sec模型训练准备数据

使用方法:
    python run_risk_ppr_complete.py
"""

import sys
import os
from pathlib import Path

# 添加scripts目录到路径
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# 确保必要的目录存在
os.makedirs("logs", exist_ok=True)
os.makedirs("data/processed/metadata", exist_ok=True)

if __name__ == "__main__":
    try:
        # 导入并运行主脚本
        from risk_weight_mapping_complete import main
        main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保scripts/risk_weight_mapping_complete.py文件存在")
    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc() 