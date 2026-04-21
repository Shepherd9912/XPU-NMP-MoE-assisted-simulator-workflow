import csv
import copy
import os
import multiprocessing as mp
import traceback # ⚠️ 新增：用于打印子进程报错
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 导入物理基座与模块
from hardware_config import init_global_model, get_model_profile, ComponentEnergyModel
from latency_energy import LatencyEnergyEngine
from dynamic_scheduler import DynamicScheduler

class StatsTracker:
    def __init__(self):
        self.energy = {"Router": 0.0, "Attention": 0.0, "Cold": 0.0, "Hot": 0.0, "Sum": 0.0, "In_Re": 0.0, "Out": 0.0}
        self.delay_active = {"Router": 0, "Attention": 0, "Cold": 0, "Hot": 0, "Sum": 0, "In_Re": 0, "Out": 0}
        self.delay_critical = {"System_Total": 0}
        self.counts = {"Token_2Hot": 0, "Token_1Hot1Cold": 0, "Token_2Cold": 0}

def evaluate_layer_batch(layer_buffer, tracker, effective_L, strategy_name, override_mode, engine, scheduler, sys_batch_size, alpha_state, current_token_id="0", layer_id="0"):
    raw_hot_tokens = 0
    raw_cold_tokens = 0
    
    for token in layer_buffer:
        e1, e2 = token["exp1"], token["exp2"]
        if e1 == "hot": raw_hot_tokens += 1
        elif e1 == "cold": raw_cold_tokens += 1
        if e2 == "hot": raw_hot_tokens += 1
        elif e2 == "cold": raw_cold_tokens += 1

    model = get_model_profile()
    scale_factor = model.K / 2.0  
    
    hot_tokens = int(raw_hot_tokens * scale_factor)
    cold_tokens = int(raw_cold_tokens * scale_factor)

    if model.model_type in ["deepseek_v2", "qwen_moe"]:
        shared_experts = model.cfg.get("num_shared_experts", 2)
        hot_tokens += (len(layer_buffer) * shared_experts)

    if override_mode == "All_NPU_Cold":
        actual_total_experts_per_token = model.K + (model.cfg.get("num_shared_experts", 0) if model.model_type in ["deepseek_v2", "qwen_moe"] else 0)
        cold_tokens = len(layer_buffer) * actual_total_experts_per_token
        hot_tokens = 0
        is_dense = False
    elif override_mode == "Dense_Attn":
        cold_tokens = 0
        hot_tokens = 0
        is_dense = True
    else:
        is_dense = False

    alpha = 0.0  
    if override_mode in ["All_PIM_Attn", "Hetero_PIM_Attn_Base", "Hetero_PIM_Attn_DyCAR"]:
        mem_state = scheduler.capacity_profiler.analyze_memory_state(sys_batch_size, effective_L)
        if mem_state["status"] == "❌ OOM":
            alpha = mem_state["pim_attn_ratio"]
        else:
            alpha = 1.0  
            
    elif override_mode == "Hetero_Dynamic_Optimal":
        alpha, mem_state = scheduler.get_optimal_dispatch(
            sys_batch_size=sys_batch_size, 
            effective_L=effective_L, 
            cold_tokens=cold_tokens, 
            hot_tokens=hot_tokens
        )
    else:
        alpha = 0.0 

    kv_evict_bytes = 0
    if override_mode == "Hetero_Dynamic_Optimal":
        model = get_model_profile()
        layer_kv_bytes = model.BYTES_PER_TOKEN_KV / model.L_layers
        
        current_pim_tokens = effective_L * alpha
        prev_pim_tokens = alpha_state["prev_seq_len"] * alpha_state["prev_alpha"]
        
        evicted_tokens = max(0, prev_pim_tokens - current_pim_tokens)
        kv_evict_bytes = evicted_tokens * layer_kv_bytes
        
        alpha_state["prev_alpha"] = alpha
        alpha_state["prev_seq_len"] = effective_L
        
        if evicted_tokens > 0 or effective_L % 500 == 0:
            alpha_state["logs"].append({
                "Token_ID": current_token_id,
                "Layer_ID": layer_id,
                "Seq_Len": effective_L,
                "Alpha": round(alpha, 3),
                "Evicted_KB": round(kv_evict_bytes / 1024, 2)
            })

    res = engine.predict_performance(
        sys_batch_size=sys_batch_size, effective_L=effective_L, 
        cold_tokens=cold_tokens, hot_tokens=hot_tokens, 
        is_dense=is_dense, override_mode=override_mode, 
        alpha=alpha, kv_evict_bytes=kv_evict_bytes 
    )
    
    tracker.delay_critical["System_Total"] += res["critical_cycles"]
    for key in tracker.delay_active.keys():
        tracker.delay_active[key] += res["delay_dict"][key]
    for key in tracker.energy.keys():
        tracker.energy[key] += res["energy_dict"][key]

def evaluate_system_pim(trace_csv_path, strategy_name, override_mode=None, sys_batch_size=1):
    energy_model = ComponentEnergyModel()
    engine = LatencyEnergyEngine(energy_model)
    scheduler = DynamicScheduler(num_hot_experts=2)
    
    model = get_model_profile()
    max_seq_len = model.cfg.get("max_position_embeddings", 16384)
    
    checkpoints = [1024, 2048, 4096, 8192, 16384, float('inf')]
    checkpoint_names = ["1~1024", "1~2048", "1~4096", "1~8192", "1~16384"]
    tracker = StatsTracker()
    snapshots = {}
    
    alpha_state = {"prev_alpha": 1.0, "prev_seq_len": 0, "logs": []}
    
    current_checkpoint_idx = 0
    current_seq_len = 1
    current_token_id = None
    current_layer_id = None
    current_state_key = None
    layer_buffer = []

    with open(trace_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_id = row["Token_ID"].strip()
            layer_id = row.get("Layer_ID", "Unknown").strip()
            
            if current_token_id is None:
                current_token_id = token_id
                current_layer_id = layer_id
                
            state_key = f"{token_id}_{layer_id}"
            if current_state_key is None:
                current_state_key = state_key
                
            if state_key != current_state_key:
                evaluate_layer_batch(
                    layer_buffer, tracker, min(current_seq_len, max_seq_len), 
                    strategy_name, override_mode, engine, scheduler, sys_batch_size,
                    alpha_state, current_token_id, current_layer_id
                )
                
                layer_buffer = []
                current_state_key = state_key
                current_layer_id = layer_id
                
                if token_id != current_token_id:
                    if current_checkpoint_idx < len(checkpoints) and current_seq_len == checkpoints[current_checkpoint_idx]:
                        snapshots[checkpoint_names[current_checkpoint_idx]] = copy.deepcopy(tracker)
                        current_checkpoint_idx += 1
                    current_seq_len += 1
                    current_token_id = token_id

            layer_buffer.append({
                "exp1": row["Selected_Expert_1"].strip().lower(),
                "exp2": row["Selected_Expert_2"].strip().lower()
            })

        if layer_buffer:
            evaluate_layer_batch(
                layer_buffer, tracker, min(current_seq_len, max_seq_len), 
                strategy_name, override_mode, engine, scheduler, sys_batch_size,
                alpha_state, current_token_id, current_layer_id
            )
            if current_checkpoint_idx < len(checkpoints) and current_seq_len == checkpoints[current_checkpoint_idx]:
                snapshots[checkpoint_names[current_checkpoint_idx]] = copy.deepcopy(tracker)

    snapshots["All_Tokens"] = copy.deepcopy(tracker)
    
    flat_data = []
    for cp_name, cp_tracker in snapshots.items():
        for metric_category, metric_dict in vars(cp_tracker).items():
            metric_type = metric_category.capitalize()
            unit = "Count" if metric_type == "Counts" else ("Joules" if metric_type == "Energy" else "Cycles")
            for comp, value in metric_dict.items():
                flat_data.append([cp_name, strategy_name, metric_type, comp, value, unit])
                
    returned_logs = alpha_state["logs"] if strategy_name == "Hetero_Dynamic_Optimal" else []
    return flat_data, returned_logs

def run_single_evaluation(task):
    """
    ⚠️ 核心修复区：捕获异常防崩溃死锁，大数据直接在这里落盘，不走 IPC 返回
    """
    try:
        model_config = task["model_config"]
        batch_size = task["batch_size"]
        exp = task["exp"]
        
        init_global_model(model_config)
        model_name = get_model_profile().model_type
        
        flat_data, alpha_logs = evaluate_system_pim(exp["file"], exp["strategy"], exp["override"], sys_batch_size=batch_size)
        formatted_data = [[model_name, batch_size] + row for row in flat_data]
        
        # ⭐️ 修复核心：大体积 Log 数据直接在子进程写入文件，不要通过 return 塞入进程管道！
        if alpha_logs:
            log_key = f"{model_name}_B{batch_size}"
            log_filename = f"Alpha_Eviction_Trace_{log_key}.csv"
            
            with open(log_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Token_ID", "Layer_ID", "Seq_Len", "Alpha", "Evicted_KB"])
                writer.writeheader()
                writer.writerows(alpha_logs)
            # print(f"   [Trace] 成功导出: {log_filename}") # 如果觉得终端太乱可注释

        # 返回空列表代替 alpha_logs，保持管道轻量
        return formatted_data, True
        
    except Exception as e:
        # 如果代码有隐藏 BUG（如字典不存在 KeyError），这里会大声喊出来而不是默默卡死！
        print(f"\n❌ 子进程发生异常! 任务: {task['model_config']} - {task['exp']['strategy']} Batch {task['batch_size']}")
        print(traceback.format_exc())
        return [], False

if __name__ == "__main__":
    base_experiments = [
        {"file": "phi_16hot4top2_mmlu_wr_baseline.csv", "strategy": "Dense_Attn", "override": "Dense_Attn"},
        {"file": "phi_16hot4top2_mmlu_wr_baseline.csv", "strategy": "All_NPU_Cold", "override": "All_NPU_Cold"},
        {"file": "phi_16hot4top2_mmlu_wr_baseline.csv", "strategy": "Hetero_Base", "override": None},
        {"file": "phi_16hot4top2_mmlu_wr_DyCAR.csv", "strategy": "Hetero_DyCAR", "override": None},
        {"file": "phi_16hot4top2_mmlu_wr_baseline.csv", "strategy": "Hetero_PIM_Attn_Base", "override": "All_PIM_Attn"},
        {"file": "phi_16hot4top2_mmlu_wr_DyCAR.csv", "strategy": "Hetero_PIM_Attn_DyCAR", "override": "All_PIM_Attn"},
        {"file": "phi_16hot4top2_mmlu_wr_DyCAR.csv", "strategy": "Hetero_Dynamic_Optimal", "override": "Hetero_Dynamic_Optimal"}
    ]

    test_suite = {
        "configs/phi_mini.json": base_experiments,
        "configs/olmoe_1b_7b.json": base_experiments,
        "configs/qwen_1_5.json": base_experiments
    }
    
    TEST_BATCH_SIZES = [1, 4, 8, 16, 32]
    
    tasks = []
    for model_config, experiments in test_suite.items():
        if not os.path.exists(model_config):
            continue
        for batch_size in TEST_BATCH_SIZES:
            for exp in experiments:
                if os.path.exists(exp["file"]):
                    tasks.append({
                        "model_config": model_config,
                        "batch_size": batch_size,
                        "exp": exp
                    })
                    
    all_results = []
    max_cores = mp.cpu_count()
    print(f"\n🔥 检测到 {max_cores} 个逻辑核心，启动基于 ED²P 的架构评估矩阵 (任务数: {len(tasks)})...")
    
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        futures = [executor.submit(run_single_evaluation, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="🚀 整体评估进度"):
            flat_data, is_success = future.result()
            if is_success and flat_data:
                all_results.extend(flat_data)
                
    if all_results:
        output_filename = "Ultimate_Flat_Metrics_MultiModel_pim.csv"
        with open(output_filename, "w", newline='') as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(["Model", "Batch_Size", "Interval", "Strategy", "Metric_Type", "Component", "Value", "Unit"])
            writer.writerows(all_results)
        print(f"\n🎉 完美！五层物理引擎已全部闭环执行完毕，大一统数据保存至: {output_filename}")
        print("💡 注: Trace 驱逐日志已在各子进程运行期间独立保存为 CSV 文件。")
        