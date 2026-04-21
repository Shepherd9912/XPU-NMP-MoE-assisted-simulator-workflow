# hardware_config.py
import math
import numpy as np
import json
import os
from multi_model_data import MULTI_MODEL_DATA
from multi_pim_attn_data import MULTI_PIM_ATTN_DATA
# ==============================================================================
# 1. 硬件常量与性能基准配置
# ==============================================================================
NPU_FREQ_MHZ = 1000            
BUS_BANDWIDTH_GBPS = 64.0      
HIDDEN_SIZE = 4096             
MAX_SEQ_LEN = 16384            

MAC_ROUTER = 4096 * 16
MAC_EXPERT = (4096 * 960 * 2) + (960 * 4096)  
MAC_SUM = 4096 * 2
# ==============================================================================
# 新增：物理容量与模型参数界限配置 (Capacity & Footprint Constants)
# ==============================================================================
# 1. 硬件物理容量上限
PIM_TOTAL_CAPACITY_GB = 4.0        # 假设 3D HBM 总容量为 4GB, 和h2llm对应
NPU_SRAM_CAPACITY_MB = 128.0       # NPU 侧片上 SRAM 为 128MB
PIM_CAPACITY_BYTES = PIM_TOTAL_CAPACITY_GB * (1024**3)
NPU_SRAM_BYTES = NPU_SRAM_CAPACITY_MB * (1024**2)

# 2. Phi-mini-MoE (FP16) 的模型静态/动态内存占用 (全层 32 Layers 累计)
BYTES_PER_HOT_EXPERT = 755 * (1024**2)  # 单个热专家 32 层总计约 755 MB
BYTES_PER_TOKEN_KV = 128 * 1024         # 单个 Token 32 层的 KV Cache 约 128 KB

# 3. 调度警戒线
PIM_CAPACITY_WARNING_THRESHOLD = 0.85   # PIM 内存占用达到 85% 时触发高水位预警
# ==============================================================================
# 2. 精细化组件级能耗模型
# ==============================================================================
class ComponentEnergyModel:
    def __init__(self):
        self.e_lpddr5 = 7.0       # 外部 LPDDR5 访存代价
        self.e_hbm = 0.88         # 3D Hybrid Bonding 内部红利
        self.e_bus = 1.0          # 片间总线通信能耗
        self.e_mac_xpu = 0.682    # NPU MAC
        self.e_mac_nmp = 0.604    # 3D HBM Tensor Core MAC @1GHz
        self.e_sram_xpu = 0.027   # xPU SRAM
        self.e_sram_nmp = 0.027   # NMP Shared Buffer
        self.REUSE_FACTOR = 2.0  

    def calc_joules(self, dram_bytes=0, hbm_bits=0, macs=0, is_xpu=True, bus_bits=0):
        energy_pj = 0.0
        energy_pj += (dram_bytes * 8) * self.e_lpddr5
        energy_pj += hbm_bits * self.e_hbm
        energy_pj += macs * (self.e_mac_xpu if is_xpu else self.e_mac_nmp)
        sram_bits = (macs * 3 * 16) / self.REUSE_FACTOR
        energy_pj += sram_bits * (self.e_sram_xpu if is_xpu else self.e_sram_nmp)
        energy_pj += bus_bits * self.e_bus
        return energy_pj / 1e12 

# ==============================================================================
# 4. 数据查询与插值函数 (对外暴露的 API)
# ==============================================================================
class ModelProfile:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
            
        self.model_type = self.cfg.get("model_type", "unknown")
        self.H = self.cfg["hidden_size"]
        self.L_layers = self.cfg["num_hidden_layers"]
        
        # QKV 和 Head 参数
        self.num_q_heads = self.cfg["num_attention_heads"]
        self.num_kv_heads = self.cfg["num_key_value_heads"]
        self.head_dim = self.cfg["head_dim"]
        
        self.H_q = self.num_q_heads * self.head_dim
        self.H_kv = self.num_kv_heads * self.head_dim
        
        # =====================================================================
        # ⭐️ 新增补丁：动态解析并计算所有的算子 MACs 和容量参数
        # =====================================================================
        # 获取每次激活的专家数 (K) 和 中间层维度 (I)
        self.K = self.cfg.get("num_experts_per_tok", 2)       # 默认 2
        self.I = self.cfg.get("intermediate_size", 960)       # 默认 Phi
        self.num_experts = self.cfg.get("num_local_experts", 16) 

        # 动态计算基础算子 MACs
        self.MAC_ROUTER = self.H * self.num_experts
        # SwiGLU 专家的 MACs = Gate/Up (H * 2I) + Down (I * H) = 3 * H * I
        self.MAC_EXPERT = (self.H * self.I * 2) + (self.I * self.H)  
        self.MAC_SUM = self.H * self.K

        # 动态计算内存占用 (假设模型全精度为 FP16 = 2 Bytes)
        # 1. 动态 KV Cache (单 Token 所有层): 2(K和V) * H_kv * 2(Bytes) * Layers
        self.BYTES_PER_TOKEN_KV = 4 * self.H_kv * self.L_layers
        
        # 2. 动态 Hot Expert 容量 (单个专家所有层): 参数量 * 2(Bytes) * Layers
        # 注意：Expert 的参数量正好等同于它的 MACs 次数
        self.BYTES_PER_HOT_EXPERT = self.MAC_EXPERT * 2 * self.L_layers
        # =====================================================================

        # 预先计算 Attention 的基础 MACs 常数项和系数项 (针对单层 Decoding)
        self._init_attention_macs_formula()

    def _init_attention_macs_formula(self):
        """严格推导 Attention 的理论 MACs 计算公式"""
        if self.model_type == "deepseek_v2":
            kv_rank = self.cfg.get("kv_lora_rank", 512)
            proj_macs = (self.H * self.H_q) + (self.H * kv_rank * 2) + (self.H_q * self.H)
            dynamic_coef = 2 * self.H_q 
            
            self.attn_const = proj_macs
            self.attn_coef = dynamic_coef
            
        else:
            w_q_macs = self.H * self.H_q
            w_k_macs = self.H * self.H_kv
            w_v_macs = self.H * self.H_kv
            w_o_macs = self.H_q * self.H
            
            self.attn_const = w_q_macs + w_k_macs + w_v_macs + w_o_macs
            self.attn_coef = 2 * self.H_q

    def get_attention_macs(self, L):
        return self.attn_const + (self.attn_coef * L)

# 初始化当前模型环境
current_model_path = os.environ.get("MOE_CONFIG_PATH", "configs/phi_mini.json")
GLOBAL_MODEL = ModelProfile(current_model_path)

def init_global_model(config_path):
    """供外界在循环切换模型时调用"""
    global GLOBAL_MODEL
    GLOBAL_MODEL = ModelProfile(config_path)
    print(f"⚙️ 已成功加载硬件评估模型: {GLOBAL_MODEL.model_type.upper()} ({config_path})")

def get_model_profile():
    """供各个组件动态获取当前模型"""
    if GLOBAL_MODEL is None:
        raise ValueError("GLOBAL_MODEL 未初始化！请先调用 init_global_model().")
    return GLOBAL_MODEL

# 全局函数，供 evaluator_light.py 调用
def get_attention_macs(L):
    return GLOBAL_MODEL.get_attention_macs(L)

PIM_HOT_EXPERT_CYCLES_BASE = 118324

def calculate_transfer_cycles_and_bits(has_hot_expert):
    if has_hot_expert == 0: return 0, 0
    # 动态读取当前模型的 hidden_size
    bits = 1 * GLOBAL_MODEL.H * 16  
    transfer_time_sec = (bits / 8) / (BUS_BANDWIDTH_GBPS * 1e9)
    cycles = int(transfer_time_sec * (NPU_FREQ_MHZ * 1e6))
    return cycles, bits

def get_attention_metrics(effective_L, batch_size):
    """【多模型适配版】获取 NPU Attention 性能"""
    model_type = get_model_profile().model_type
    
    if model_type not in MULTI_MODEL_DATA:
        raise ValueError(f"❌ 查表失败: MULTI_MODEL_DATA 中没有模型 {model_type} 的数据！")
        
    model_data = MULTI_MODEL_DATA[model_type]
    
    if batch_size not in model_data:
        raise ValueError(f"❌ 查表失败: {model_type} 没有 Batch_Size = {batch_size} 的数据！")
        
    batch_data = model_data[batch_size]
    
    # 动态扫描锚点
    available_L = []
    for op_name in batch_data.keys():
        if op_name.startswith("attention_L"):
            L_val = int(op_name.replace("attention_L", ""))
            available_L.append(L_val)
            
    if not available_L:
        raise KeyError(f"❌ 查表失败: {model_type} Batch {batch_size} 中没有任何 Attention 锚点数据！")
        
    available_L.sort()
    attn_cycles = [batch_data[f"attention_L{L}"]["Cycles"] for L in available_L]
    attn_bytes = [batch_data[f"attention_L{L}"]["DRAM_Bytes"] for L in available_L]
    
    c = int(np.interp(effective_L, available_L, attn_cycles))
    b = int(np.interp(effective_L, available_L, attn_bytes))
    return c, b
# protected
# def get_static_component(name, batch_size):
#     """【多模型适配版】获取静态算子的性能"""
#     model_type = get_model_profile().model_type
#     
#     # 拦截旧的硬编码后缀，防止外部调用报错
#     if name == "router_4096_16": name = "router"
#     if name == "weighted_sum_4096": name = "weighted_sum"
#     
#     try:
#         return MULTI_MODEL_DATA[model_type][batch_size][name]
#     except KeyError as e:
#         raise KeyError(f"❌ 查表失败: 无法获取 {model_type} -> Batch {batch_size} -> {name} 的数据。")
def get_static_component(name, hits):
    """【多模型适配版】获取静态算子的性能，支持任意命中次数的自动插值与线性外推"""
    import numpy as np
    
    model_type = get_model_profile().model_type
    
    # 拦截旧的硬编码后缀
    if name == "router_4096_16": name = "router"
    if name == "weighted_sum_4096": name = "weighted_sum"
    
    model_data = MULTI_MODEL_DATA[model_type]
    
    if hits == 0:
        return {"Cycles": 0, "DRAM_Bytes": 0}
        
    # ⭐️ 核心修复：只提取那些真正包含该算子 (name) 的 Batch 锚点！
    available_batches = sorted([k for k in model_data.keys() if isinstance(k, int) and name in model_data[k]])
    
    if not available_batches:
        raise KeyError(f"❌ 查表失败: 模型 {model_type} 的字典中完全找不到算子 {name}！")
        
    try:
        # 1. 刚好精确命中查表 (比如 hits = 8)
        if hits in available_batches:
            return model_data[hits][name]
            
        # 2. 超出查表上限 (例如 hits=256 > 32)，使用最大 Batch 数据等比例线性放大
        max_b = available_batches[-1]
        if hits > max_b:
            ref_data = model_data[max_b][name]
            scale = hits / max_b
            return {
                "Cycles": int(ref_data["Cycles"] * scale),
                "DRAM_Bytes": int(ref_data["DRAM_Bytes"] * scale)
            }
            
        # 3. 落在中间区间 (例如 hits=6)，使用 NumPy 进行线性插值
        cycles_list = [model_data[b][name]["Cycles"] for b in available_batches]
        bytes_list = [model_data[b][name]["DRAM_Bytes"] for b in available_batches]
        
        return {
            "Cycles": int(np.interp(hits, available_batches, cycles_list)),
            "DRAM_Bytes": int(np.interp(hits, available_batches, bytes_list))
        }
    except Exception as e:
        raise KeyError(f"❌ 查表插值计算失败: {model_type} -> {name} 发生错误: {e}")
def pad_sequence_length(L, granularity=16):
    """模拟硬件的数据对齐填充开销"""
    if L == 0: return 0
    return math.ceil(L / granularity) * granularity

def get_pim_attention_metrics(effective_L):
    """【多模型适配版】获取 PIM Attention 性能"""
    padded_L = pad_sequence_length(effective_L, granularity=16)
    model_type = get_model_profile().model_type
    
    if model_type not in MULTI_PIM_ATTN_DATA:
        raise ValueError(f"❌ 查表失败: MULTI_PIM_ATTN_DATA 中没有 {model_type} 的数据！")
        
    pim_data = MULTI_PIM_ATTN_DATA[model_type]
    
    pim_attn_anchors_L = []
    for key in pim_data.keys():
        if key.startswith("L"):
            pim_attn_anchors_L.append(int(key.replace("L", "")))
    pim_attn_anchors_L.sort()
    
    pim_attn_cycles = [pim_data[f"L{L}"]["Cycles"] for L in pim_attn_anchors_L]
    pim_attn_bytes = [pim_data[f"L{L}"]["DRAM_Bytes"] for L in pim_attn_anchors_L]
    
    c = int(np.interp(padded_L, pim_attn_anchors_L, pim_attn_cycles))
    b = int(np.interp(padded_L, pim_attn_anchors_L, pim_attn_bytes))
    return c, b
    