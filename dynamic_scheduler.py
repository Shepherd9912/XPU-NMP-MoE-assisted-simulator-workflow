# dynamic_scheduler.py
from capacity_profiler import CapacityProfiler
from hardware_config import ComponentEnergyModel
from latency_energy import LatencyEnergyEngine

class DynamicScheduler:
    def __init__(self, num_hot_experts=2):
        # 1. 物理容量哨兵
        self.capacity_profiler = CapacityProfiler(num_hot_experts=num_hot_experts)
        
        # 2. 调度大脑内置的“微型试算引擎” (Micro-Predictor)
        # 它和外层评估器使用的是绝对相同的物理法则，保证预测与执行的 100% 对齐
        self.energy_model = ComponentEnergyModel()
        self.predictor_engine = LatencyEnergyEngine(self.energy_model)
        
        # ⭐️ 新增：全局状态记忆锁
        self.global_alpha = 1.0
        self.prev_seq_L = 0  # 追踪序列长度，用于判断是否是新句子

    def get_optimal_dispatch(self, sys_batch_size, effective_L, cold_tokens, hot_tokens):
        """
        核心调度算法：在容量安全的阈值内，寻找 ED²P 最小的 Attention 切分比例 alpha
        alpha: 留在 PIM 执行的 Attention 比例 (0.0 到 1.0)
        """
        # 1. 自动检测：如果序列变短了，说明开始了新一轮对话（Prefill），重置状态！
        if effective_L < self.prev_seq_L:
            self.global_alpha = 1.0
        self.prev_seq_L = effective_L
        
        # 2. 探底物理容量限制 (获取当前 B x L 下的容量安全红线)
        mem_state = self.capacity_profiler.analyze_memory_state(sys_batch_size, effective_L)
        alpha_max = mem_state["pim_attn_ratio"]  

        # =====================================================================
        # ⭐️ 核心防抖补丁 A：迟滞大阶梯下折 (Hysteresis Step-down)
        # 解决“大雪山”问题：当容量红线碾压当前 alpha 时，不要每步微调，
        # 而是直接向下寻找最近的 0.1 整数关口，强行制造一个足够大的“容量安全缓冲区”。
        # =====================================================================
        if alpha_max < self.global_alpha:
            stepped_alpha = int(alpha_max * 10) / 10.0
            if stepped_alpha == self.global_alpha: 
                stepped_alpha -= 0.1 # 强制下台阶
            self.global_alpha = max(0.0, stepped_alpha)

        # =====================================================================
        # ⭐️ 核心防抖补丁 B：单向递减锁定 (Monotonicity Lock)
        # 解决“条形码”问题：泼出去的水收不回来，alpha 只能降不能升！
        # =====================================================================
        actual_upper_bound = min(self.global_alpha, alpha_max)

        best_alpha = 0.0
        min_ed2p = float('inf')
        
        # 3. 生成扫描候选点 (仅在合法的 actual_upper_bound 以下扫描)
        test_alphas = [i / 10.0 for i in range(11) if (i / 10.0) <= actual_upper_bound]
        
        # 细节拉满：将极限边界值也强行加入候选池
        if actual_upper_bound not in test_alphas:
            test_alphas.append(actual_upper_bound)
        test_alphas = sorted(list(set(test_alphas)))
        
        # 4. 开启异构屋顶线试算 (Data-Driven Empirical Roofline Intersection)
        for alpha in test_alphas:
            # 物理硬墙：绝不评估超过当前锁定上限的切分方案
            if alpha > actual_upper_bound + 1e-6:
                continue  
                
            # 呼叫微型物理引擎，模拟在 "Hetero_Dynamic_Optimal" 模式下跑这个 alpha 会怎样
            res = self.predictor_engine.predict_performance(
                sys_batch_size=sys_batch_size, 
                effective_L=effective_L, 
                cold_tokens=cold_tokens, 
                hot_tokens=hot_tokens, 
                is_dense=False, 
                override_mode="Hetero_Dynamic_Optimal", 
                alpha=alpha
            )
            
            cycles = res["critical_cycles"]
            joules = res["energy_total"]
            
            # 🏆 核心目标函数：ED²P (Energy-Delay^2 Product)
            ed2p = (cycles ** 2) * joules
            
            if ed2p < min_ed2p:
                min_ed2p = ed2p
                best_alpha = alpha
                
        # 5. 确认决策并更新全局记忆锁 (锁死当前的下限)
        self.global_alpha = best_alpha
        
        return best_alpha, mem_state
    