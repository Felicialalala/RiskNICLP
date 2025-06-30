"""
RiskNICLP-5G Risk Bias Intelligent Diagnostic

智能风险偏置诊断工具 - 实时监控风险参数状态和梯度
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class RiskBiasIntelligentDiagnostic:
    """🔥 RiskNICLP风险偏置智能诊断工具 - 完全修复版"""
    
    def __init__(self, model, logger=None, training_stage="basic"):
        self.model = model
        self.logger = logger
        self.training_stage = training_stage
        self.gradient_threshold = 1e-7
        self.diagnostic_history = []  # 🔥 修复：初始化诊断历史
        
        # 🔥 修复：正确的参数查找映射
        self.risk_param_paths = {
            # SharedEncoder中的risk_bias_layers
            'shared_encoder_layers': 'shared_encoder.risk_bias_layers',
            # TrueRiskBiasGATConv中的实际参数名
            'param_names': ['beta_src', 'beta_dst', 'temperature']
        }
        
        if self.logger:
            self.logger.logger.info("🔧 RiskBiasIntelligentDiagnostic初始化 - 修复版")
    
    def comprehensive_risk_bias_diagnosis(self) -> Dict[str, Any]:
        """
        🔥 全面的风险偏置诊断
        
        Returns:
            诊断报告字典
        """
        diagnosis = {
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            'parameter_status': self._diagnose_parameter_status(),
            'gradient_status': self._diagnose_gradient_status(),
            'training_effectiveness': self._diagnose_training_effectiveness(),
            'risk_mechanism_health': self._diagnose_risk_mechanism_health(),
            'recommendations': []
        }
        
        # 🔥 修复：阶段感知的训练就绪性检查
        param_status = diagnosis['parameter_status']
        gradient_status = diagnosis['gradient_status']
        effectiveness = diagnosis['training_effectiveness']
        
        # 🔧 根据训练阶段调整就绪性检查标准
        if self.training_stage == 'basic':
            # Basic阶段：参数冻结、无梯度、机制未激活都是正常的
            readiness_checks = {
                'has_risk_params': param_status['total_risk_params'] > 0,
                'params_properly_frozen': param_status['frozen_params'] == param_status['total_risk_params'],
                'phase_aligned': effectiveness['training_phase_alignment'],
                'basic_stage_normal': True  # Basic阶段只要参数存在且正确冻结就算就绪
            }
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware阶段：参数解冻、有梯度、机制激活
            readiness_checks = {
                'has_risk_params': param_status['total_risk_params'] > 0,
                'params_trainable': param_status['trainable_params'] > 0,
                'has_gradients': gradient_status['has_gradients'],
                'phase_aligned': effectiveness['training_phase_alignment'],
                'mechanism_active': diagnosis['risk_mechanism_health']['mechanism_active']
            }
        else:
            # 未知阶段，使用严格检查
            readiness_checks = {
                'has_risk_params': param_status['total_risk_params'] > 0,
                'params_trainable': param_status['trainable_params'] > 0,
                'has_gradients': gradient_status['has_gradients'],
                'phase_aligned': effectiveness['training_phase_alignment'],
                'mechanism_active': diagnosis['risk_mechanism_health']['mechanism_active']
            }
        
        diagnosis['training_readiness'] = readiness_checks
        
        # 生成诊断建议
        diagnosis['recommendations'] = self._generate_recommendations(diagnosis)
        
        # 记录诊断历史
        self.diagnostic_history.append(diagnosis)
        
        # 输出诊断报告
        self._report_diagnosis(diagnosis)
        
        return diagnosis
    
    def _diagnose_parameter_status(self) -> Dict[str, Any]:
        """🔥 修复版：诊断风险参数状态 - 正确查找路径"""
        status = {
            'total_risk_params': 0,
            'frozen_params': 0,
            'trainable_params': 0,
            'risk_param_details': [],
            'parameter_distribution': {},
            'freeze_status_correct': False,
            'found_risk_layers': []
        }
        
        # 🔥 修复：检查SharedEncoder中的risk_bias_layers
        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'risk_bias_layers'):
            risk_layers = self.model.shared_encoder.risk_bias_layers
            
            for layer_idx, layer in enumerate(risk_layers):
                layer_name = f"risk_bias_layers[{layer_idx}]"
                status['found_risk_layers'].append(layer_name)
                
                # 检查每层的风险参数
                for param_name in self.risk_param_paths['param_names']:
                    if hasattr(layer, param_name):
                        param = getattr(layer, param_name)
                        if isinstance(param, nn.Parameter):
                            status['total_risk_params'] += 1
                            
                            param_info = {
                                'name': f"{layer_name}.{param_name}",
                                'requires_grad': param.requires_grad,
                                'shape': list(param.shape),
                                'value': param.data.clone().detach().cpu().numpy().tolist(),
                                'grad_norm': None
                            }
                            
                            if param.grad is not None:
                                param_info['grad_norm'] = param.grad.norm().item()
                            
                            status['risk_param_details'].append(param_info)
                            
                            if param.requires_grad:
                                status['trainable_params'] += 1
                            else:
                                status['frozen_params'] += 1
        
        # 🔥 修复：检查其他可能的风险参数位置
        # UEEncoder中的GAT层（如果存在）
        if hasattr(self.model, 'ue_encoder') and hasattr(self.model.ue_encoder, 'gat_layers'):
            for layer_idx, layer in enumerate(self.model.ue_encoder.gat_layers):
                for param_name in self.risk_param_paths['param_names']:
                    if hasattr(layer, param_name):
                        param = getattr(layer, param_name)
                        if isinstance(param, nn.Parameter):
                            status['total_risk_params'] += 1
                            
                            param_info = {
                                'name': f"ue_encoder.gat_layers[{layer_idx}].{param_name}",
                                'requires_grad': param.requires_grad,
                                'shape': list(param.shape),
                                'value': param.data.clone().detach().cpu().numpy().tolist(),
                                'grad_norm': None
                            }
                            
                            if param.grad is not None:
                                param_info['grad_norm'] = param.grad.norm().item()
                            
                            status['risk_param_details'].append(param_info)
                            
                            if param.requires_grad:
                                status['trainable_params'] += 1
                            else:
                                status['frozen_params'] += 1
        
        # 计算冻结状态正确性
        expected_frozen = status['total_risk_params'] if self.training_stage == "basic" else 0
        status['freeze_status_correct'] = (status['frozen_params'] == expected_frozen)
        
        # 参数分布统计
        if status['risk_param_details']:
            values = []
            for param_info in status['risk_param_details']:
                if isinstance(param_info['value'], list):
                    if len(param_info['value']) == 1:
                        values.append(param_info['value'][0])
                    else:
                        values.extend(param_info['value'])
                else:
                    values.append(param_info['value'])
            
            if values:
                status['parameter_distribution'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return status
    
    def _diagnose_gradient_status(self) -> Dict[str, Any]:
        """🔥 修复版：诊断梯度状态 - 正确查找路径"""
        gradient_info = {
            'has_gradients': False,
            'gradient_norms': {},
            'effective_gradients': 0,
            'zero_gradients': 0,
            'gradient_statistics': {},
            'gradient_health_score': 0.0
        }
        
        gradient_values = []
        
        # 🔥 修复：检查SharedEncoder中的risk_bias_layers梯度
        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'risk_bias_layers'):
            risk_layers = self.model.shared_encoder.risk_bias_layers
            
            for layer_idx, layer in enumerate(risk_layers):
                for param_name in self.risk_param_paths['param_names']:
                    if hasattr(layer, param_name):
                        param = getattr(layer, param_name)
                        if isinstance(param, nn.Parameter) and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            gradient_info['gradient_norms'][f"layers[{layer_idx}].{param_name}"] = grad_norm
                            gradient_values.append(grad_norm)
                            
                            if grad_norm > self.gradient_threshold:
                                gradient_info['effective_gradients'] += 1
                            else:
                                gradient_info['zero_gradients'] += 1
                            
                            gradient_info['has_gradients'] = True
        
        # 计算梯度统计
        if gradient_values:
            gradient_info['gradient_statistics'] = {
                'mean_grad_norm': float(np.mean(gradient_values)),
                'max_grad_norm': float(np.max(gradient_values)),
                'min_grad_norm': float(np.min(gradient_values)),
                'std_grad_norm': float(np.std(gradient_values))
            }
            
            # 梯度健康评分：0-1之间
            effective_ratio = gradient_info['effective_gradients'] / len(gradient_values)
            gradient_info['gradient_health_score'] = effective_ratio
        
        return gradient_info
    
    def _diagnose_training_effectiveness(self) -> Dict[str, Any]:
        """🔥 修复版：诊断训练有效性 - 阶段感知逻辑"""
        effectiveness = {
            'risk_params_learning': False,
            'parameter_change_rate': {},
            'learning_indicators': {},
            'training_phase_alignment': False
        }
        
        # 检查参数变化率
        if len(self.diagnostic_history) > 1:
            prev_diagnosis = self.diagnostic_history[-2]
            prev_params = prev_diagnosis['parameter_status']['risk_param_details']
            curr_params = self._diagnose_parameter_status()['risk_param_details']
            
            for prev_p, curr_p in zip(prev_params, curr_params):
                if prev_p['name'] == curr_p['name']:
                    prev_val = np.array(prev_p['value'])
                    curr_val = np.array(curr_p['value'])
                    
                    change_rate = np.linalg.norm(curr_val - prev_val)
                    effectiveness['parameter_change_rate'][prev_p['name']] = change_rate
                    
                    if change_rate > 1e-6:  # 参数有显著变化
                        effectiveness['risk_params_learning'] = True
        
        # 🔥 修复：使用诊断工具的training_stage，而不是model的training_phase
        param_status = self._diagnose_parameter_status()
        
        if self.training_stage == 'basic':
            # Basic阶段：风险参数应该被冻结，这是正确的状态
            expected_frozen = param_status['total_risk_params']
            actual_frozen = param_status['frozen_params']
            effectiveness['training_phase_alignment'] = (actual_frozen == expected_frozen)
            
            # Basic阶段风险参数冻结是正常的，不检查学习状态
            if not effectiveness['risk_params_learning'] and len(self.diagnostic_history) <= 1:
                effectiveness['risk_params_learning'] = True  # 标记为符合Basic阶段预期
                
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware阶段：风险参数应该解冻并参与学习
            expected_trainable = param_status['total_risk_params']
            actual_trainable = param_status['trainable_params']
            effectiveness['training_phase_alignment'] = (actual_trainable == expected_trainable)
            
            # Risk-aware阶段需要检查参数是否真正在学习
            # 如果历史数据不足，暂时标记为True等待观察
            if not effectiveness['risk_params_learning'] and len(self.diagnostic_history) <= 2:
                effectiveness['risk_params_learning'] = True
        else:
            # 未知阶段，保持原有严格检查
            effectiveness['training_phase_alignment'] = False
        
        return effectiveness
    
    def _diagnose_risk_mechanism_health(self) -> Dict[str, Any]:
        """🔥 修复版：诊断风险机制健康状态 - 正确查找路径"""
        health = {
            'risk_bias_values': {},
            'parameter_ranges': {},
            'mechanism_active': False,
            'configuration_issues': [],
            'per_layer_analysis': []
        }
        
        # 🔥 修复：检查SharedEncoder中的risk_bias_layers参数值
        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'risk_bias_layers'):
            risk_layers = self.model.shared_encoder.risk_bias_layers
            
            all_beta_src = []
            all_beta_dst = []
            all_temperature = []
            trainable_params_count = 0  # 🔥 新增：统计可训练参数数量
            
            for layer_idx, layer in enumerate(risk_layers):
                layer_analysis = {'layer_idx': layer_idx}
                
                if hasattr(layer, 'beta_src'):
                    beta_src_value = layer.beta_src.data.detach().cpu().numpy()
                    if beta_src_value.ndim == 0:
                        beta_src_value = float(beta_src_value)
                    else:
                        beta_src_value = beta_src_value.tolist()
                    layer_analysis['beta_src'] = beta_src_value
                    layer_analysis['beta_src_trainable'] = layer.beta_src.requires_grad  # 🔥 新增
                    all_beta_src.append(beta_src_value if isinstance(beta_src_value, (int, float)) else beta_src_value[0])
                    if layer.beta_src.requires_grad:
                        trainable_params_count += 1
                
                if hasattr(layer, 'beta_dst'):
                    beta_dst_value = layer.beta_dst.data.detach().cpu().numpy()
                    if beta_dst_value.ndim == 0:
                        beta_dst_value = float(beta_dst_value)
                    else:
                        beta_dst_value = beta_dst_value.tolist()
                    layer_analysis['beta_dst'] = beta_dst_value
                    layer_analysis['beta_dst_trainable'] = layer.beta_dst.requires_grad  # 🔥 新增
                    all_beta_dst.append(beta_dst_value if isinstance(beta_dst_value, (int, float)) else beta_dst_value[0])
                    if layer.beta_dst.requires_grad:
                        trainable_params_count += 1
                
                if hasattr(layer, 'temperature'):
                    temperature_value = float(layer.temperature.data.detach().cpu())
                    layer_analysis['temperature'] = temperature_value
                    layer_analysis['temperature_trainable'] = layer.temperature.requires_grad  # 🔥 新增
                    all_temperature.append(temperature_value)
                    if layer.temperature.requires_grad:
                        trainable_params_count += 1
                
                health['per_layer_analysis'].append(layer_analysis)
            
            # 聚合参数值
            if all_beta_src:
                health['risk_bias_values']['beta_src_mean'] = float(np.mean(all_beta_src))
                health['risk_bias_values']['beta_src_std'] = float(np.std(all_beta_src))
            
            if all_beta_dst:
                health['risk_bias_values']['beta_dst_mean'] = float(np.mean(all_beta_dst))
                health['risk_bias_values']['beta_dst_std'] = float(np.std(all_beta_dst))
            
            if all_temperature:
                health['risk_bias_values']['temperature_mean'] = float(np.mean(all_temperature))
                health['risk_bias_values']['temperature_std'] = float(np.std(all_temperature))
            
            # 参数范围检查
            if all_beta_src:
                beta_src_mean = health['risk_bias_values']['beta_src_mean']
                health['parameter_ranges']['beta_src_in_range'] = 0.0 <= beta_src_mean <= 5.0
            
            if all_beta_dst:
                beta_dst_mean = health['risk_bias_values']['beta_dst_mean']
                health['parameter_ranges']['beta_dst_in_range'] = 0.0 <= beta_dst_mean <= 5.0
            
            if all_temperature:
                temperature_mean = health['risk_bias_values']['temperature_mean']
                health['parameter_ranges']['temperature_in_range'] = 0.01 <= temperature_mean <= 1.0
            
            # 检查配置问题
            if all_beta_src and all_beta_dst:
                beta_src_mean = health['risk_bias_values']['beta_src_mean']
                beta_dst_mean = health['risk_bias_values']['beta_dst_mean']
                
                if abs(beta_src_mean - 0.5) < 0.01 and abs(beta_dst_mean - 0.5) < 0.01:
                    health['configuration_issues'].append("Beta参数未从初始值0.5改变，可能未参与训练")
            
            if all_temperature:
                temperature_mean = health['risk_bias_values']['temperature_mean']
                
                if temperature_mean < 0.05:
                    health['configuration_issues'].append("温度参数过小，可能导致注意力过于尖锐")
                
                if temperature_mean > 0.5:
                    health['configuration_issues'].append("温度参数过大，可能导致注意力过于平滑")
            
            # 🔥 修复：机制激活状态的正确定义
            # 在Basic阶段：mechanism_active应该基于参数是否可训练，而不是参数值
            # 在Risk-aware阶段：mechanism_active应该基于参数可训练且值有变化
            
            total_risk_params = len(all_beta_src) + len(all_beta_dst) + len(all_temperature)
            
            if self.training_stage == 'basic':
                # Basic阶段：如果参数不可训练，则机制应该视为"未激活"（即使forward时在计算）
                health['mechanism_active'] = trainable_params_count > 0
                health['mechanism_status'] = f"Basic阶段: {trainable_params_count}/{total_risk_params} 参数可训练"
            else:
                # Risk-aware阶段：参数可训练且值有变化才算激活
                params_changed = not (abs(health['risk_bias_values'].get('beta_src_mean', 0.5) - 0.5) < 0.01 and
                                     abs(health['risk_bias_values'].get('beta_dst_mean', 0.5) - 0.5) < 0.01)
                health['mechanism_active'] = trainable_params_count > 0 and params_changed
                health['mechanism_status'] = f"Risk-aware阶段: {trainable_params_count}/{total_risk_params} 参数可训练, 值变化: {params_changed}"
                
            # 🔥 新增：详细的激活状态信息
            health['trainable_params_count'] = trainable_params_count
            health['total_risk_params'] = total_risk_params
        
        return health
    
    def _generate_recommendations(self, diagnosis: Dict[str, Any]) -> List[str]:
        """🔥 修复版：生成阶段感知的诊断建议"""
        recommendations = []
        
        param_status = diagnosis['parameter_status']
        gradient_status = diagnosis['gradient_status']
        effectiveness = diagnosis['training_effectiveness']
        health = diagnosis['risk_mechanism_health']
        
        # 参数状态建议
        if param_status['total_risk_params'] == 0:
            recommendations.append("❌ 严重：未找到风险偏置参数，检查模型架构")
            return recommendations  # 严重错误，直接返回
        
        # 🔥 阶段感知的参数状态建议
        if self.training_stage == 'basic':
            # Basic阶段：参数冻结是正常的
            if param_status['trainable_params'] == 0:
                recommendations.append("✅ 正常：Basic阶段风险参数已冻结，符合预期")
            elif param_status['trainable_params'] > 0:
                recommendations.append("⚠️ 警告：Basic阶段风险参数应该被冻结")
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware阶段：参数应该解冻
            if param_status['trainable_params'] == 0:
                recommendations.append("❌ 严重：Risk-aware阶段风险参数仍被冻结，应该解冻")
            elif param_status['frozen_params'] > 0 and param_status['trainable_params'] > 0:
                recommendations.append("⚠️ 警告：风险参数部分冻结部分解冻，状态不一致")
            else:
                recommendations.append("✅ 正常：风险参数已解冻，符合Risk-aware阶段预期")
        
        # 🔥 阶段感知的梯度状态建议
        if self.training_stage == 'basic':
            # Basic阶段：无梯度是正常的（参数被冻结）
            if not gradient_status['has_gradients']:
                recommendations.append("✅ 正常：Basic阶段风险参数无梯度（参数冻结）")
            elif gradient_status['effective_gradients'] > 0:
                recommendations.append("⚠️ 警告：Basic阶段风险参数不应该有梯度")
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware阶段：应该有梯度
            if not gradient_status['has_gradients']:
                recommendations.append("❌ 严重：Risk-aware阶段风险参数无梯度，检查损失函数和反向传播")
            elif gradient_status['effective_gradients'] == 0:
                recommendations.append("❌ 严重：所有梯度为零，风险机制未参与学习")
            elif gradient_status['gradient_health_score'] < 0.3:
                recommendations.append("⚠️ 警告：梯度健康评分低，考虑调整学习率")
            else:
                recommendations.append("✅ 正常：风险参数梯度健康")
        
        # 训练有效性建议
        if not effectiveness['training_phase_alignment']:
            recommendations.append("❌ 严重：训练阶段与参数状态不对齐")
        else:
            recommendations.append("✅ 正常：训练阶段与参数状态对齐")
        
        # 🔥 阶段感知的风险机制建议
        if self.training_stage == 'basic':
            # Basic阶段：风险机制未激活是正常的
            if not health['mechanism_active']:
                recommendations.append("✅ 正常：Basic阶段风险机制暂未激活（参数冻结）")
            else:
                recommendations.append("⚠️ 警告：Basic阶段风险机制不应该激活")
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware阶段：风险机制应该激活
            if not health['mechanism_active']:
                recommendations.append("❌ 严重：Risk-aware阶段风险机制未激活，参数保持初始值")
            else:
                recommendations.append("✅ 正常：风险机制已激活")
        
        # 配置问题建议（所有阶段都适用）
        for issue in health['configuration_issues']:
            if self.training_stage == 'basic' and "Beta参数未从初始值0.5改变" in issue:
                recommendations.append(f"✅ 正常：Basic阶段{issue}是预期的")
            else:
                recommendations.append(f"⚠️ 配置：{issue}")
        
        # 参数学习建议
        if not effectiveness['risk_params_learning'] and len(self.diagnostic_history) > 2:
            if self.training_stage == 'basic':
                recommendations.append("✅ 正常：Basic阶段风险参数无变化（参数冻结）")
            else:
                recommendations.append("⚠️ 警告：风险参数长期无变化，检查优化器配置")
        
        if not recommendations:
            recommendations.append("✅ 优秀：风险偏置机制运行正常")
        
        return recommendations
    
    def _report_diagnosis(self, diagnosis: Dict[str, Any]) -> None:
        """🔥 修复版：输出诊断报告 - 阶段感知"""
        if not self.logger:
            return
        
        param_status = diagnosis['parameter_status']
        gradient_status = diagnosis['gradient_status']
        effectiveness = diagnosis['training_effectiveness']
        
        # 🔥 阶段感知的参数状态报告
        total_params = param_status['total_risk_params']
        frozen_params = param_status['frozen_params']
        trainable_params = param_status['trainable_params']
        
        self.logger.logger.info(f"🔍 风险参数状态({self.training_stage}): {frozen_params}/{total_params} 冻结, {trainable_params}/{total_params} 可训练")
        
        # 详细参数信息
        if param_status.get('risk_param_details'):
            self.logger.logger.info(f"📊 详细参数信息:")
            for param_info in param_status['risk_param_details']:
                grad_info = f", 梯度范数: {param_info.get('grad_norm', 'N/A')}" if param_info.get('grad_norm') is not None else ""
                self.logger.logger.info(f"  {param_info['name']}: requires_grad={param_info['requires_grad']}, 值={param_info['value']}{grad_info}")
        
        # 🔧 阶段感知的状态验证
        if self.training_stage == 'basic':
            # Basic阶段：期望所有参数冻结
            if total_params > 0:
                if frozen_params == total_params:
                    self.logger.logger.info(f"✅ Basic阶段：{frozen_params}个风险参数正确冻结")
                else:
                    self.logger.logger.warning(f"⚠️ Basic阶段：应冻结{total_params}个风险参数，实际冻结{frozen_params}个")
            else:
                self.logger.logger.error("❌ 严重错误：未找到任何风险参数！")
            
            # Basic阶段风险参数有梯度是异常的
            if gradient_status['has_gradients']:
                self.logger.logger.warning("⚠️ Basic阶段风险参数意外有梯度！")
            else:
                self.logger.logger.info("✅ Basic阶段：风险参数正确无梯度")
                
        elif self.training_stage in ['risk_aware', 'full_fusion']:
            # Risk-aware及以后阶段：期望所有参数可训练
            if total_params > 0:
                if trainable_params == total_params:
                    self.logger.logger.info(f"✅ {self.training_stage.title()}阶段：{trainable_params}个风险参数正确解冻")
                else:
                    self.logger.logger.warning(f"⚠️ {self.training_stage.title()}阶段：应解冻{total_params}个风险参数，实际解冻{trainable_params}个")
            
            # Risk-aware阶段应该有梯度
            if gradient_status['has_gradients']:
                effective_gradients = gradient_status['effective_gradients']
                self.logger.logger.info(f"✅ {self.training_stage.title()}阶段：{effective_gradients}个参数有有效梯度")
            else:
                self.logger.logger.warning(f"❌ {self.training_stage.title()}阶段：风险参数缺少梯度！")
        
        # 训练阶段对齐检查
        if effectiveness['training_phase_alignment']:
            self.logger.logger.info("✅ 训练阶段与参数状态正确对齐")
        else:
            self.logger.logger.warning("❌ 训练阶段与参数状态不对齐")
        
        # 梯度统计报告
        if gradient_status.get('gradient_statistics'):
            stats = gradient_status['gradient_statistics']
            self.logger.logger.info(
                f"📊 梯度统计: 平均={stats['mean_grad_norm']:.6f}, "
                f"最大={stats['max_grad_norm']:.6f}, "
                f"健康评分={gradient_status['gradient_health_score']:.3f}"
            )
        
        # 🔥 修复：建议报告，正常状态不输出为warning/error
        for rec in diagnosis['recommendations']:
            if rec.startswith("❌"):
                self.logger.logger.error(rec)
            elif rec.startswith("⚠️"):
                self.logger.logger.warning(rec)
            elif rec.startswith("✅"):
                self.logger.logger.info(rec)  # 正常状态作为info输出
            else:
                self.logger.logger.info(rec)
        
        # 🔥 新增：详细的机制状态报告
        health = diagnosis.get('risk_mechanism_health', {})
        if health:
            mechanism_status = health.get('mechanism_status', '未知状态')
            trainable_count = health.get('trainable_params_count', 0)
            total_count = health.get('total_risk_params', 0)
            
            self.logger.logger.info(f"🔧 风险机制状态: {mechanism_status}")
            
            if self.training_stage == 'basic':
                if trainable_count == 0:
                    self.logger.logger.info("✅ 正确：Basic阶段风险机制未激活（所有参数已冻结）")
                else:
                    self.logger.logger.warning(f"⚠️ 异常：Basic阶段有{trainable_count}个风险参数未冻结")
            else:
                if trainable_count == total_count and trainable_count > 0:
                    self.logger.logger.info(f"✅ 正确：{self.training_stage.title()}阶段风险机制已激活（{trainable_count}个参数可训练）")
                elif trainable_count == 0:
                    self.logger.logger.error(f"❌ 严重：{self.training_stage.title()}阶段风险机制未激活（所有参数仍被冻结）")
                else:
                    self.logger.logger.warning(f"⚠️ 部分：{self.training_stage.title()}阶段仅{trainable_count}/{total_count}个参数可训练")
    
    def force_unfreeze_risk_parameters(self) -> bool:
        """🔥 修复版：强制解冻风险参数 - 正确路径"""
        success = False
        unfrozen_count = 0
        
        # 🔥 修复：解冻SharedEncoder中的risk_bias_layers
        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'risk_bias_layers'):
            risk_layers = self.model.shared_encoder.risk_bias_layers
            
            for layer_idx, layer in enumerate(risk_layers):
                for param_name in self.risk_param_paths['param_names']:
                    if hasattr(layer, param_name):
                        param = getattr(layer, param_name)
                        if isinstance(param, nn.Parameter):
                            param.requires_grad = True
                            success = True
                            unfrozen_count += 1
                            
                            if self.logger:
                                self.logger.logger.info(f"🔓 强制解冻: risk_bias_layers[{layer_idx}].{param_name}")
        
        # 🔥 解冻其他可能的风险参数位置
        if hasattr(self.model, 'shared_encoder') and hasattr(self.model.shared_encoder, 'unfreeze_risk_params'):
            try:
                self.model.shared_encoder.unfreeze_risk_params()
                success = True
            except Exception as e:
                if self.logger:
                    self.logger.logger.warning(f"调用unfreeze_risk_params失败: {e}")
        
        if success and self.logger:
            self.logger.logger.info(f"✅ 风险参数强制解冻完成，共解冻 {unfrozen_count} 个参数")
        
        return success
    
    def verify_training_readiness(self) -> Tuple[bool, Dict[str, Any]]:
        """
        🔥 修复版：验证训练就绪状态 - 阶段感知
        
        Returns:
            (是否就绪, 检查结果)
        """
        diagnosis = self.comprehensive_risk_bias_diagnosis()
        
        # 使用comprehensive_risk_bias_diagnosis中已经计算的阶段感知readiness_checks
        readiness_checks = diagnosis.get('training_readiness', {})
        
        is_ready = all(readiness_checks.values())
        
        if self.logger:
            if is_ready:
                if self.training_stage == 'basic':
                    self.logger.logger.info("✅ Basic阶段风险偏置状态验证通过")
                else:
                    self.logger.logger.info("✅ 风险偏置训练就绪验证通过")
            else:
                failed_checks = [k for k, v in readiness_checks.items() if not v]
                if self.training_stage == 'basic':
                    self.logger.logger.warning(f"❌ Basic阶段状态验证失败: {failed_checks}")
                else:
                    self.logger.logger.warning(f"❌ 训练就绪验证失败: {failed_checks}")
        
        return is_ready, {
            'readiness_checks': readiness_checks,
            'full_diagnosis': diagnosis
        } 