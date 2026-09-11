# Forecast Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可从小时观测表构造样本、运行朴素基线与MLP训练、独立评估测试集的第一版流程。

**Architecture:** NumPy/pandas负责数据与评估，PyTorch仅用于MLP训练。prepare/fit/evaluate阶段通过不可覆盖的清单与运行目录连接；训练和测试在接口上隔离。

**Tech Stack:** Python3.10、NumPy2.2.6、pandas2.3.3、PyTorch2.10.0、unittest。

## Global Constraints

- 用户已于本轮明确批准书面设计并要求实现、git push；不再次请求同一批准。
- 数据与代码分离，既有脚本与原始数据不修改。
- 默认输入11列、输出10列、历史168小时、未来24小时，UTC边界2023-09-01和2023-11-01。
- 标准化只拟合训练时段，标签窗口不跨集合，测试不参与早停。
- 输出目录拒绝覆盖，负发电量保留，未定义的单能源因子不作为本版输入/目标。
- 没有GPU实测就不得声称GPU测试通过；本地合成测试不等于真实数据预测结果。

## 文件结构和实施顺序

核心实现为一个相互耦合、可独立验收的端到端任务，内部按数据→基线→训练→CLI的红绿循环执行。主代理并行负责隔离依赖环境及README，避免争用核心文件。实现完成后安排只读审查，修复重要问题，再回归与推送。

## Task 1: 最小可用的预测训练流程

### 范围和文件
新增 configs/erco_2023.json；energy_forecast/__init__.py、data.py、baselines.py、models.py、training.py、evaluation.py、artifacts.py（可选，用于共享运行记录）；prepare_forecast_data.py；run_forecast.py；tests/test_data.py、test_baselines.py、test_training.py、test_cli.py；requirements-train.txt。不得编辑已有两个 Python 文件。README 与设计状态由主代理更新。

必须先阅读已批准设计 docs/superpowers/specs/2026-09-11-forecast-training-design.md。所有设计数值为准。用户明确授权实现和 git push，代码在 D:/learning/研究生项目/课题2-2 的 codex/forecast-training 分支；子代理只能提交本任务新增文件，主代理负责推送和合入。

### 接口约定
- 数据根目录 --root 默认 /workspace/energy-carbon；源文件相对根路径 data/processed/erco_2023_v1/observations.csv。
- prepare_forecast_data.py --root ROOT --config CONFIG --output PREPARED_DIR。CONFIG 默认脚本所在仓库的 configs/erco_2023.json，不能依赖 cwd。不训练、无需 torch，输出 manifest.json（生效配置、source SHA256、实际字段顺序、三段起止与样本数、训练集拟合的输入/输出标准化统计）。
- run_forecast.py fit --prepared PREPARED_DIR --model {persistence,seasonal24,mlp} --output RUN_DIR [--device cpu|cuda] [--epochs N]。fit 仅训练与验证；两个朴素模型不拟合参数，只保存方法和验证指标。MLP 保存 best.pt、history.csv。保存 run.json（配置含 CLI 覆盖后的实际值、数据清单哈希、源文件哈希、状态、版本、Git 提交及 dirty 状态）。
- run_forecast.py evaluate --run RUN_DIR --output EVAL_DIR [--device cpu|cuda]。这是显式测试入口，验证 source/manifest 哈希未变，加载该次 run 保存的方法与最佳权重及标准化参数，不接受隐式重拟合。输出 metrics.json 和 predictions.csv。
- 模型/训练模块内部接口可按上述边界自行确定，但命名保持简洁；在报告中记录用于其他入口的函数签名。
- 每个输出目录独立，禁止覆盖。必要的父目录可创建。输入文件永不写入。不要调用 eval 或加载任意未受信任 pickle；torch.load 使用 weights_only=True，checkpoint 只含 tensor 和安全基本类型，版本字符串转普通 str。
- 默认输入 demand_mw + 九个 generation_<fuel>_mwh + factor_generated_kg_per_mwh，共11列；默认输出九个generation及factor，共10列。顺序见设计。只允许配置中批准的历史字段，不允许将时间戳、未来质量标记或官方reference字段动态加入。
- 配置至少含 lookback=168,horizon=24,stride=1,train_end=2023-09-01T00:00:00Z,val_end=2023-11-01T00:00:00Z,hidden_size=128,learning_rate=0.001,batch_size=64,epochs=20,patience=5,seed=42,num_workers=0。
- 通用代码接受小型合成数据用于测试；第一版默认仍针对ERCO。时间戳必须带时区并在整点，严格递增连续唯一，不静默排序或去重。报错要用明确异常，不依赖可被 python -O 移除的 assert。
- 以目标首行 j 标识起点，X=[j-L,j)，y=[j,j+H)。整段 y 属于且仅属于一个集合。默认真实数据原始行数5826/1464/1470，窗口数5635/1441/1447（用于测试期望，不将数量硬编码逻辑）。
- 标准化先用训练时段原始行拟合，然后滑窗；验证测试可以使用已经发生的历史，不重新训练和重拟合。常数列scale=1且记录。
- 惰性滑窗避免持久化三维全量数组。保留raw y以便原单位评估；baseline使用raw历史y和相同起点。
- 两个基线的 horizon>24 也不能用未来值；seasonal24历史长度不足24应提前报错。
- MLP Linear(L*F,128)-ReLU-Linear(128,128)-ReLU-Linear(128,H*K)，无额外注意力/卷积。seed42。训练MSE按总元素加权累计，验证仅用于早停，测试只经 evaluate 调用。
- CUDA请求不可用必须失败；CPU必须可测试。支持从best.pt复评，不实现断点续训。
- 指标为每个目标及每步长MAE/RMSE，物理单位写清；不合并不同单位总分，不输出MAPE或精度百分比。
- 预测明细一行一个origin/horizon/target，保留origin_time、target_time、horizon(1-based)、target、actual、prediction。不能漏掉最后不足batch_size的批次。
- 非有限输入、参数、预测、损失、尺度均应拒绝；对epochs/batch_size/patience/L/H/stride的非正、非整型值提前报错。学习率有限正数。失败运行写failed/error记录后仍非零退出，不自动重试。
- 模型权重不直接裁剪为正，不宣称物理一致性，不从综合因子反推分能源排放量。

### 开发步骤与测试（先红后绿）
- [x] 先新增 unittest 测试文件，以缺失模块/API为失败原因运行，记录RED输出，不先写实现。对每一模块使用小型可手算的真实数组，不用伪造mock替代计算。
- [x] 数据测试必须含：168/24真实网格窗口计数；小序列精确X/y；跨边界拒绝；验证测试极端值不改变训练均值；修改起点未来不会改变历史X；常数列；NaN只在未选字段可接受；重复/缺小时/无时区/非整点/不足历史；源文件修改后拒绝load。
- [x] 基线测试必须含：最近值重复；固定24小时循环到48/50步；短历史错误；相同起点未来不影响结果；手算MAE/RMSE与逐步长一致。
- [x] 写最小实现，通过针对性测试；严禁在测试失败时只改期望来迎合错误实现。
- [x] 训练测试必须含：小型合成样本真实优化，权重改变、有限loss；保存加载预测匹配；触发早停时记录best epoch且加载best；最后不足batch的加权验证损失正确；训练期间不调用测试；CUDA不可用失败；NaN/Inf失败并记录。
- [x] CLI集成用 tempfile 中明确标注的合成小时数据：prepare → fit persistence/seasonal24 → evaluate；fit MLP少量轮次 → evaluate；校验输出列、数量、哈希、源文件未变、重复输出目录拒绝覆盖。不将这些临时数据或测试结果提交Git。
- [x] Python测试命令：.venv-test/Scripts/python.exe -m unittest discover -s tests -v（Windows）；Linux为 python -m unittest discover -s tests -v。数据-only测试必须不导入torch；全套测试环境应安装torch后执行，不能用跳过训练测试冒充通过。
- [x] 帮助命令：python prepare_forecast_data.py --help；python run_forecast.py --help；python run_forecast.py fit --help；python run_forecast.py evaluate --help。
- [x] 依赖：numpy==2.2.6,pandas==2.3.3，训练固定torch==2.10.0（官方支持Python3.10，Windows本地CPU测试，Linux云端cu126），标准库unittest不依赖pytest。训练requirements引用requirements-data.txt并固定torch；torch轮子源由README安装命令指定，不同时安装多个CUDA发行版。依据官方安全公告GHSA-63cw-57p8-fm3p，将初始暂定旧版本更新为2.10.0；这不是整体安全认证，仍不加载不受信任的checkpoint。
- [x] 自检字段顺序、半开区间、评估缩放、检查点版本兼容、所有输入输出不可覆盖；在干净输出中跑全套测试。
- [x] git diff --check 后仅stage本任务文件并提交，禁止git add .、禁止push、禁止编辑README/设计/计划/已有代码。给主代理提供commit与测试证明。

### 报告
将完整报告用apply_patch保存到 C:/Users/22164/Documents/Codex/2026-09-04/bang/forecast-work/task-1-report.md。包含API、文件清单、RED/GREEN命令和关键输出、测试总数、已知未验证项、提交号。最后回复不超过15行，使用DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT之一。


## Task 2: 主代理集成与发布

- [x] 创建被.gitignore排除的.venv-test，用Python3.10与固定依赖执行测试；不修改系统Python。
- [x] 更新README：标明云端观测表已回传成功；记录prepare、fit、evaluate命令、两套依赖安装方法、输出位置、验证边界和CPU/GPU区别。
- [x] 完成核心任务只读审查；重要缺陷修复后重跑覆盖测试，记录具体结果。
- [x] 主代理独立运行全部测试与CLI帮助，检查Git差异、临时文件、凭据及大文件。
- [x] 在docs中记录实际完成项与未验证项；原始CSV不入库。
- [ ] 将开发分支以fast-forward方式合入main并推送现有origin；绝不force push、创建无请求的PR或覆盖用户改动。
- [ ] 比对本地HEAD与GitHub main SHA，确认干净工作区。给出服务器先prepare、后训练的短命令。

## 实施进度

- 设计确认：已完成。
- 核心代码：已完成（28d6bdb；修复01f9a1a、4fa2f91）。
- 测试与审查：独立19项全部通过（76.448秒、无跳过），核心任务两轮审查已通过；发布前整体审查接续执行。
- 文档：README、设计状态与本实施记录已更新。
- 推送：本记录提交时尚待最终整体审查；随后合入main、推送并现场核对远程SHA。最终发布状态以交付回复和Git历史为准。

## 已执行验证与边界

- 本地隔离环境：Windows / Python3.10.20 / NumPy2.2.6 / pandas2.3.3 / PyTorch2.10.0+cpu，15个已安装包依赖检查通过。云端Python3.10.16与CUDA未由本地测试替代。
- 完整测试命令：`.venv-test/Scripts/python.exe -m unittest discover -s tests -v`；显式设置PYTHONIOENCODING=utf-8、PYTHONUTF8=0，覆盖原先暴露的中文Windows编码失败。
- 首次独立回归曾因子进程UTF-8与GBK解码不一致出现2项错误，修复后19项全部通过；另外补齐坏配置失败记录与已保存运行配置一致性检查。
- 四份本地原始文件SHA256与云端记录一致，重建8760行41列观测表；实际样本数5635/1441/1447，X=(168,11)、y=(24,10)。两个朴素基线只运行验证阶段，没有读取真实测试指标。
- MLP优化、早停、保存及重载仅使用合成CPU测试数据；真实数据完整训练、GPU兼容、论文复现与效果优越性均未宣称完成。
- 四CLI帮助入口、Git空白检查、原有两个Python文件哈希与忽略规则已核对。没有替换或删除旧代码，无遗留弃用模块；原始CSV、检查点和本地虚拟环境不进入Git。
