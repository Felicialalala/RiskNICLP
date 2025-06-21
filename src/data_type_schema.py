"""
数据类型规范文件
解决异质图构建过程中的数据类型混合警告问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 完整的数据类型映射字典
DATA_TYPE_MAPPING = {
    # 索引和标识列
    'Unnamed: 0': 'int64',
    'SrcId': 'str',
    'Rank': 'int64',
    
    # 时间相关列  
    'StartTime': 'str',
    'LastTime': 'str',
    'StartTime_unix': 'float64',
    'LastTime_unix': 'float64',
    
    # 网络协议和标志
    'Trans': 'int64',
    'Flgs': 'str',
    'Seq': 'int64',
    'Proto': 'str',  # 第23列 - 问题列
    'Sport': 'int64',  # 第24列 - 问题列  
    'Dport': 'int64',
    'Proto_normalized': 'str',
    
    # 时间和流量指标
    'Dur': 'float64',
    'RunTime': 'float64',
    'IdleTime': 'float64',
    'Duration': 'float64',
    'Mean': 'float64',
    'StdDev': 'float64',
    'Sum': 'float64',
    'Min': 'float64',
    'Max': 'float64',
    
    # MAC地址
    'SrcMac': 'str',
    'DstMac': 'str',
    'SrcOui': 'str',
    'DstOui': 'str',
    
    # IP地址
    'SrcAddr': 'str',
    'DstAddr': 'str',
    
    # QoS和DSCP
    'sTos': 'str',  # 第28列 - 问题列
    'dTos': 'str',
    'sDSb': 'str',
    'dDSb': 'str',
    'sCo': 'str',
    'dCo': 'str',
    
    # TTL和跳数
    'sTtl': 'int64',
    'dTtl': 'int64',
    'sHops': 'int64',
    'dHops': 'int64',
    'sIpId': 'str',  # 第36列 - 问题列
    'dIpId': 'str',
    
    # MPLS标签
    'sMpls': 'str',
    'dMpls': 'str',
    
    # AS号码
    'AutoId': 'str',
    'sAS': 'str',
    'dAS': 'str',
    'iAS': 'str',
    
    # 数据包统计
    'Cause': 'str',
    'NStrok': 'int64',
    'sNStrok': 'int64',
    'dNStrok': 'int64',
    'TotPkts': 'int64',
    'SrcPkts': 'int64',
    'DstPkts': 'int64',
    'TotBytes': 'int64',
    'SrcBytes': 'int64',
    'DstBytes': 'int64',
    'TotAppByte': 'int64',
    'SAppBytes': 'int64',
    'DAppBytes': 'int64',
    
    # 比率和负载
    'PCRatio': 'float64',
    'Offset': 'float64',
    'sMeanPktSz': 'float64',
    'dMeanPktSz': 'float64',
    'Load': 'float64',
    'SrcLoad': 'float64',
    'DstLoad': 'float64',
    
    # 丢包和重传
    'Loss': 'float64',
    'SrcLoss': 'float64',
    'DstLoss': 'float64',
    'pLoss': 'float64',
    'Retrans': 'int64',
    'SrcRetra': 'int64',
    'DstRetra': 'int64',
    'pRetran': 'float64',
    'SrcGap': 'int64',
    'DstGap': 'int64',
    
    # 速率
    'Rate': 'float64',
    'SrcRate': 'float64',
    'DstRate': 'float64',
    'Dir': 'str',
    
    # 间隔时间统计
    'SIntPkt': 'float64',
    'SIntDist': 'float64',
    'SIntPktAct': 'float64',
    'SIntActDist': 'float64',
    'SIntPktIdl': 'float64',
    'SIntIdlDist': 'float64',
    'DIntPkt': 'float64',
    'DIntDist': 'float64',
    'DIntPktAct': 'float64',
    'DIntActDist': 'float64',
    'DIntPktIdl': 'float64',
    
    # 抖动
    'SrcJitAct': 'float64',
    'DstJitter': 'float64',
    'DstJitAct': 'float64',
    
    # TCP状态和窗口
    'State': 'str',
    'srcUdata': 'str',
    'dstUdata': 'str',
    'SrcWin': 'str',  # 第96列 - 问题列
    'DstWin': 'str',  # 第97列 - 问题列
    
    # VLAN标签
    'sVlan': 'str',
    'dVlan': 'str',
    'sVid': 'str',
    'dVid': 'str',
    'sVpri': 'str',
    'dVpri': 'str',
    
    # 范围和TCP选项
    'SRange': 'str',
    'ERange': 'str',
    'SrcTCPBase': 'str',
    'DstTCPBase': 'str',
    'TcpRtt': 'float64',
    'SynAck': 'str',
    'AckDat': 'str',  # 第109列 - 问题列
    'TcpOpt': 'str',
    
    # 标签和攻击信息
    'Label': 'str',
    'Attack Type': 'str',
    'Attack Tool': 'str',
    
    # 5G语义特征
    'is_5g_control_plane': 'bool',  # 第115列 - 问题列
    'qos_class_src': 'str',
    'qos_class_dst': 'str',
    'traffic_ratio': 'float64',
    'packet_ratio': 'float64',
    'has_5g_features': 'bool',
    
    # 实体分类
    'SrcAddr_Type': 'str',
    'DstAddr_Type': 'str',
    'SrcAddr_Type_Refined': 'str',
}

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全的数据类型转换函数
    
    Args:
        df: 输入DataFrame
        
    Returns:
        类型转换后的DataFrame
    """
    df_converted = df.copy()
    conversion_errors = []
    
    for column, target_dtype in DATA_TYPE_MAPPING.items():
        if column in df_converted.columns:
            try:
                if target_dtype == 'int64':
                    # 安全的整数转换
                    df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').astype('Int64')
                elif target_dtype == 'float64':
                    # 安全的浮点数转换
                    df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                elif target_dtype == 'bool':
                    # 布尔类型转换
                    df_converted[column] = df_converted[column].astype(str).str.lower().isin(['true', '1', 'yes']).astype('bool')
                elif target_dtype == 'str':
                    # 字符串类型转换
                    df_converted[column] = df_converted[column].astype(str)
                    # 将'nan'字符串转换为真正的NaN
                    df_converted[column] = df_converted[column].replace('nan', pd.NA)
                    
                #logger.info(f"成功转换列 {column} 为 {target_dtype}")
                    
            except Exception as e:
                error_msg = f"转换列 {column} 为 {target_dtype} 时出错: {str(e)}"
                conversion_errors.append(error_msg)
                logger.warning(error_msg)
    
    if conversion_errors:
        logger.warning(f"数据类型转换完成，共有 {len(conversion_errors)} 个警告")
    else:
        logger.info("所有数据类型转换成功完成")
        
    return df_converted

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    验证数据质量
    
    Args:
        df: 输入DataFrame
        
    Returns:
        数据质量报告字典
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_data': {},
        'data_types': {},
        'quality_score': 0.0
    }
    
    # 检查缺失数据
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            quality_report['missing_data'][column] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
    
    # 检查数据类型
    for column in df.columns:
        quality_report['data_types'][column] = str(df[column].dtype)
    
    # 计算质量分数
    total_missing_pct = sum([info['percentage'] for info in quality_report['missing_data'].values()])
    avg_missing_pct = total_missing_pct / len(df.columns) if len(df.columns) > 0 else 0
    quality_report['quality_score'] = max(0.0, 100.0 - avg_missing_pct)
    
    return quality_report 