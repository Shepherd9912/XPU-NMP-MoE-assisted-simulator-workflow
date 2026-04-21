# capacity_profiler.py
import math
# from hardware_config import (
#     PIM_CAPACITY_BYTES, 
#     NPU_SRAM_BYTES,
#     BYTES_PER_HOT_EXPERT, 
#     BYTES_PER_TOKEN_KV,
#     PIM_CAPACITY_WARNING_THRESHOLD
# )
from hardware_config import PIM_CAPACITY_BYTES, PIM_CAPACITY_WARNING_THRESHOLD, GLOBAL_MODEL, get_model_profile
class CapacityProfiler:
    def __init__(self, num_hot_experts=2):
        # self.num_hot_experts = num_hot_experts
        # self.static_weight_bytes = self.num_hot_experts * BYTES_PER_HOT_EXPERT
        self.num_hot_experts = num_hot_experts
        # 从加载的模型配置中动态获取占用
        # self.static_weight_bytes = self.num_hot_experts * GLOBAL_MODEL.BYTES_PER_HOT_EXPERT

    def analyze_memory_state(self, batch_size, seq_len):
        """
        核心算账逻辑：计算当前 B x L 下的内存占用，并生成调度建议
        """
        # # 1. 计算 PIM 侧动态 KV Cache 占用
        # dynamic_kv_bytes = batch_size * seq_len * BYTES_PER_TOKEN_KV
        # 
        # # 2. 计算 PIM 侧总占用
        # total_pim_bytes = self.static_weight_bytes + dynamic_kv_bytes
        # pim_utilization = total_pim_bytes / PIM_CAPACITY_BYTES
        
        # 从加载的模型配置中动态获取 KV 大小
        model = get_model_profile() # 动态获取！
        
        # 动态计算当前模型的权重和 KV 占用
        static_weight_bytes = self.num_hot_experts * model.BYTES_PER_HOT_EXPERT
        dynamic_kv_bytes = batch_size * seq_len * model.BYTES_PER_TOKEN_KV
        
        # total_pim_bytes = self.static_weight_bytes + dynamic_kv_bytes
        # pim_utilization = total_pim_bytes / PIM_CAPACITY_BYTES
        total_pim_bytes = static_weight_bytes + dynamic_kv_bytes
        pim_utilization = total_pim_bytes / PIM_CAPACITY_BYTES
        
        # 3. 诊断与调度策略生成
        is_oom = pim_utilization > 1.0
        is_warning = pim_utilization >= PIM_CAPACITY_WARNING_THRESHOLD
        
        # 动态计算 Attention 切分比例 (alpha: 留在 PIM 的比例)
        # 如果内存安全，100% 在 PIM 跑；如果溢出，强制切分给 NPU
        if is_oom:
            # 算出 PIM 还能塞下多少 KV Cache
            available_kv_bytes = PIM_CAPACITY_BYTES - static_weight_bytes
            safe_kv_ratio = available_kv_bytes / dynamic_kv_bytes
            pim_attn_ratio = max(0.0, safe_kv_ratio - 0.05) # 留 5% buffer
        elif is_warning:
            # 处于警戒水位，主动卸载 30% 给 NPU 以防万一
            pim_attn_ratio = 0.7 
        else:
            pim_attn_ratio = 1.0
            
        return {
            "weight_mb": static_weight_bytes / (1024**2),
            "kv_mb": dynamic_kv_bytes / (1024**2),
            "total_mb": total_pim_bytes / (1024**2),
            "utilization": pim_utilization,
            "status": "❌ OOM" if is_oom else ("⚠️ Warning" if is_warning else "✅ Safe"),
            "pim_attn_ratio": pim_attn_ratio
        }

def run_dse_profiling():
    profiler = CapacityProfiler(num_hot_experts=2)
    
    test_batches = [1, 4, 8, 16, 32]
    test_seq_lens = [1024, 2048, 4096, 8192]
    
    print("="*80)
    print(" 🚀 PIM 3D HBM 容量高水位与调度边界探测器 (Total: 4GB)")
    print("="*80)
    print(f"{'Batch':<6} | {'Seq_Len':<8} | {'Weight(MB)':<10} | {'KV(MB)':<10} | {'Total(MB)':<10} | {'PIM_Util':<8} | {'Status':<10} | {'PIM_Attn_Ratio':<12}")
    print("-" * 80)
    
    for b in test_batches:
        for l in test_seq_lens:
            res = profiler.analyze_memory_state(b, l)
            
            # 格式化输出
            util_str = f"{res['utilization']*100:.1f}%"
            ratio_str = f"{res['pim_attn_ratio']*100:.0f}% PIM"
            
            print(f"{b:<6} | {l:<8} | {res['weight_mb']:<10.0f} | {res['kv_mb']:<10.0f} | {res['total_mb']:<10.0f} | {util_str:<8} | {res['status']:<10} | {ratio_str:<12}")
        print("-" * 80)

# if __name__ == "__main__":
    # run_dse_profiling()
if __name__ == "__main__":
    import csv
    import os
    from hardware_config import init_global_model

    # 1. 定义要评估的三个模型
    test_configs = [
        "configs/phi_mini.json",
        "configs/olmoe_1b_7b.json",
        "configs/qwen_1_5.json"
    ]
    
    # 2. 划定极端边界测试范围
    test_batches = [1, 4, 8, 16, 32]
    test_seq_lens = [512, 1024, 2048, 4096, 8192, 16384] 
    
    results = []
    
    print("🚀 开始生成多模型 PIM 容量热力图数据...")
    
    for cfg in test_configs:
        if not os.path.exists(cfg):
            print(f"⚠️ 找不到配置文件: {cfg}，跳过...")
            continue
            
        init_global_model(cfg)
        model_profile = get_model_profile()
        model_name = model_profile.model_type.upper()
        
        # 动态读取当前模型每次激活的专家数 (K)，作为需要搬运的热专家目标数
        k_experts = model_profile.K
        profiler = CapacityProfiler(num_hot_experts=k_experts)
        
        for b in test_batches:
            for l in test_seq_lens:
                res = profiler.analyze_memory_state(b, l)
                
                results.append({
                    "Model": model_name,
                    "Batch_Size": b,
                    "Seq_Len": l,
                    "Target_Hot_Experts": k_experts,
                    "Weight_MB": round(res["weight_mb"], 2),
                    "KV_MB": round(res["kv_mb"], 2),
                    "Total_MB": round(res["total_mb"], 2),
                    "PIM_Utilization_Ratio": round(res["utilization"], 4), # 核心画图指标
                    "Status": res["status"]
                })
                
    # 3. 输出 CSV
    csv_file = "Capacity_OOM_Heatmap_MultiModel.csv"
    if results:
        with open(csv_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"🎉 容量评估完成！热力图数据已保存至: {csv_file}")
