# 消融实验文件说明

## 文件结构

```
ablation_experiments/
├── README.md                 # 本说明文件
├── ablation_experiments.md   # 消融实验设计方案（详细）
├── config_rcc.yaml           # RCC数据集配置文件
├── run_ablation.py           # Python自动化运行脚本
├── run_experiments.sh        # Shell批量运行脚本
└── analyze_results.py        # 结果分析与LaTeX表格生成
```

## 使用步骤

### 1. 修改配置文件
编辑 `config_rcc.yaml`，填入实际的数据路径：
```yaml
data_split_json: "/your/path/to/RCC/splits.json"
data_csv: "/your/path/to/RCC/labels.csv"
h5_file_dir: "/your/path/to/RCC/h5_files/"
# ... 其他路径
```

### 2. 运行消融实验

**方式一：使用Shell脚本（推荐）**
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

**方式二：使用Python脚本**
```bash
# 运行所有实验
python run_ablation.py --config config_rcc.yaml --output_dir ./results

# 只运行特定实验
python run_ablation.py --config config_rcc.yaml --experiments Full A1_no_dynamic_prototype

# 测试模式（只打印命令）
python run_ablation.py --config config_rcc.yaml --dry_run
```

### 3. 分析结果
```bash
python analyze_results.py --results_dir ./results
```

输出：
- `ablation_summary_main.csv` - 主消融实验结果
- `ablation_summary_k_sensitivity.csv` - K值敏感性结果
- `ablation_summary_main.tex` - LaTeX格式表格（可直接用于论文）
- `ablation_summary_k_sensitivity.tex` - K值敏感性LaTeX表格

---

## 消融实验列表

### 主消融实验

| ID | 实验名称 | 消融组件 | 说明 |
|----|----------|----------|------|
| Full | Full Model | - | 完整模型 (Baseline) |
| A1 | w/o Dynamic Prototype | 动态原型生成 | 使用静态原型P_vis |
| A2 | w/o Optimal Transport | 最优传输算法 | 用余弦相似度替代Sinkhorn |
| A3 | w/o Gating Mechanism | 门控机制 | 使用静态prompt_bag |
| A4 | w/o Bag-level Prior | 包级文本先验 | 使用可学习Query |

### K值敏感性分析

| ID | 实验名称 | K值 | 说明 |
|----|----------|-----|------|
| A5_K2 | K=2 | 2 | 极少原型 |
| A5_K4 | K=4 | 4 | 默认值 (Baseline) |
| A5_K6 | K=6 | 6 | 中等数量 |
| A5_K8 | K=8 | 8 | 较多原型 |
| A5_K16 | K=16 | 16 | 大量原型 |

---

## 各消融实验详细说明

### A1: 动态原型消融
- **目的**: 验证动态原型生成的有效性
- **实现**: 跳过Cross-Attention，直接使用静态正交初始化的原型
- **预期**: 性能下降，因为原型无法适应不同WSI

### A2: 最优传输消融
- **目的**: 验证Sinkhorn算法相比简单相似度的优势
- **实现**: 用余弦相似度计算融合权重，替代最优传输
- **预期**: 性能下降，次优匹配

### A3: 门控机制消融
- **目的**: 验证门控动态Query的有效性
- **实现**: 直接使用静态文本Query，无视觉条件化
- **预期**: 灵活性降低

### A4: 包级先验消融
- **目的**: 验证Bag级文本提示的作用
- **实现**: 用可学习参数替代文本嵌入
- **预期**: 失去语义引导

### A5: K值敏感性
- **目的**: 分析原型数量对性能的影响
- **实现**: 测试不同K值 (2, 4, 6, 8, 16)
- **预期**: K=4或K=6最佳

---

## 注意事项

1. 每个实验运行5折交叉验证，预计耗时较长
2. 建议使用GPU运行，确保 `cuda` 可用
3. 结果会自动保存到 `./results/` 目录
4. 日志会保存到 `./logs/` 目录
5. 需要修改 `lp_model.py` 添加消融参数支持（详见 `ablation_experiments.md`）
