# 课题 2-2：多能源发电与动态碳因子研究

代码版本管理仓库。ERCO 2023 年小时观测表已在用户云端生成并回传记录；本轮新增样本构造、朴素基线与 MLP 训练评估流程。MLP 是流程基线，不是 TiDE 论文复现，也不是已验证的创新模型。完整真实数据训练效果仍以服务器后续结果为准。

研究主线：分能源电量与 CO₂ 排放量 → 动态碳因子核算 → 未来碳因子及分能源序列预测。

## 文件用途与边界

- `build_erco_2023_observations.py`：读取 OGE 与 EIA 原始文件，核算、对齐并导出观测表。使用本地数据，不联网下载，不训练模型，不修改原始文件；已有输出目录会被拒绝覆盖。
- `work/render_docx.py`：文档渲染辅助脚本，不参与能源数据计算。需另行准备 `pdf2image`、Poppler 和 LibreOffice 等依赖；运行主数据脚本不需要这些软件。
- `requirements-data.txt`：主数据脚本的直接依赖。
- `prepare_forecast_data.py` 与 `energy_forecast/data.py`：校验观测表、按时间划分、保存训练集标准化统计、按需构造窗口。
- `run_forecast.py` 与 `energy_forecast/` 中的基线、模型、训练和评估模块：提供相互分离的 `fit` / `evaluate` 流程。
- `configs/erco_2023.json`：第一版字段顺序、168/24 小时窗口、UTC 时间边界与训练参数。
- `tests/` 与 `requirements-train.txt`：自动化测试和固定的训练依赖。

原始数据、虚拟环境、模型与实验输出不进入 Git。此前粘贴到终端运行的检查片段并非独立脚本，也未被此仓库自动收录。

## 云服务器首次使用

将代码克隆到独立目录，不要覆盖已有的数据目录 `/workspace/energy-carbon`：

```bash
git clone https://github.com/napawam913-byte/energy-carbon-topic-2-2.git /workspace/energy-carbon-code
```

本仓库为公开仓库，使用以上 HTTPS 地址克隆和拉取更新不需要登录 GitHub，也不需要令牌。公开不等于任何人都能直接修改仓库；推送代码仍需要有写权限的账号完成身份验证。

参考：[GitHub 克隆说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)。

## 数据与环境位置

已有的数据环境：`/workspace/energy-carbon/.venv-data`。

运行脚本需要以下原始文件，克隆代码不会自动下载或搬动它们：

```text
/workspace/energy-carbon/data/raw/eia/EIA930_BALANCE_2023_Jan_Jun.csv
/workspace/energy-carbon/data/raw/eia/EIA930_BALANCE_2023_Jul_Dec.csv
/workspace/energy-carbon/data/raw/oge/v0.8.0/2023/power_sector_data/ERCO.csv
/workspace/energy-carbon/data/raw/oge/v0.8.0/2023/carbon_accounting/ERCO.csv
```

当前云端已具备所需 NumPy 与 pandas，不必重复安装。若重建一个环境，使用 Python 3.10，然后执行：

```bash
python -m pip install -r /workspace/energy-carbon-code/requirements-data.txt
```

## 运行观测表整理

```bash
source /workspace/energy-carbon/.venv-data/bin/activate &&
python /workspace/energy-carbon-code/build_erco_2023_observations.py --root /workspace/energy-carbon
```

运行成功时生成：

```text
/workspace/energy-carbon/data/processed/erco_2023_v1/observations.csv
/workspace/energy-carbon/data/processed/erco_2023_v1/negative_generation_audit.csv
/workspace/energy-carbon/data/processed/erco_2023_v1/metadata.json
```

若该输出目录已经存在，脚本会停止。不要为重跑而直接删除既有结果，先确认并归档旧版本。

脚本目前针对固定的 ERCO 2023 / OGE v0.8.0 数据校验值，不是任意年份通用的数据处理器。用户已回传完整 EIA 合并、小时键及文件回读通过的记录：8760 行、41 列。相同 OGE 文件本地复算的分能源求和差异与云端相符；这不构成对原始数据真实性的独立认证。

## 构建训练数据集：现在先运行这一步

准备数据不需要 PyTorch，也不会开始训练。默认使用过去 168 小时的 11 个变量，预测未来 24 小时的 10 个变量：九类能源净发电量与综合发电侧 CO₂ 因子；用电负荷只作输入。

```bash
git -C /workspace/energy-carbon-code pull --ff-only &&
source /workspace/energy-carbon/.venv-data/bin/activate &&
python /workspace/energy-carbon-code/prepare_forecast_data.py \
  --root /workspace/energy-carbon \
  --output /workspace/energy-carbon/data/prepared/erco_168_24_v1
```

配置文件默认取代码仓库中的 `configs/erco_2023.json`，与运行命令时的当前目录无关。需要改变实验设置时，复制配置并通过 `--config` 指定；不要在已保存的实验目录中修改清单。

默认划分以 UTC 时间为准，目标窗口不跨集合；验证和测试可使用起点之前已经发生的历史：

| 集合 | 原始小时数 | 默认有效预测窗口数 |
|---|---:|---:|
| 训练：起点至 2023-09-01 00:00 UTC | 5826 | 5635 |
| 验证：2023-09-01 至 2023-11-01 00:00 UTC | 1464 | 1441 |
| 测试：2023-11-01 至数据结束 | 1470 | 1447 |

输出 `manifest.json`，记录实际配置、字段顺序、样本数、训练集标准化统计和观测表哈希。窗口按需切片，不保存巨大的全量三维样本文件。

本地已用与云端 SHA256 相同的四份原始文件重建观测表，实测窗口数为 `5635 / 1441 / 1447`；数据准备与两个朴素基线的验证阶段均已运行通过。没有因此执行真实数据测试集评估或完整 MLP 训练。

## 两个无需神经网络训练的基线

准备成功后，可继续使用现有数据环境：

```bash
python /workspace/energy-carbon-code/run_forecast.py fit \
  --prepared /workspace/energy-carbon/data/prepared/erco_168_24_v1 \
  --model persistence \
  --output /workspace/energy-carbon/results/persistence_v1

python /workspace/energy-carbon-code/run_forecast.py fit \
  --prepared /workspace/energy-carbon/data/prepared/erco_168_24_v1 \
  --model seasonal24 \
  --output /workspace/energy-carbon/results/seasonal24_v1
```

`persistence` 沿用最后一个已知值；`seasonal24` 重复最近 24 小时的已知目标序列。这里的 `fit` 对这两种方法不优化参数，只保存固定方法、运行记录与验证结果。固定 24 小时滞后不等于夏令时切换日的当地钟表“同一时刻”。

## MLP 训练：另建训练环境，确认数据准备后再运行

下面的环境创建命令只在该路径尚不存在时执行，不覆盖旧环境：

```bash
test ! -e /workspace/energy-carbon/.venv-train &&
python -m venv /workspace/energy-carbon/.venv-train
```

服务器使用 RTX 4090，拟安装 CUDA 12.6 版 PyTorch；本机仅使用 CPU 版进行流程测试。镜像中的 CUDA 标签不替代实际运行验证。

```bash
source /workspace/energy-carbon/.venv-train/bin/activate &&
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126 &&
python -m pip install -r /workspace/energy-carbon-code/requirements-train.txt &&
python -m pip check
```

无 GPU 的机器可以将上面的 PyTorch 安装源换为 `https://download.pytorch.org/whl/cpu`，运行时用 `--device cpu`。下载失败不应关闭 TLS 校验；先处理网络连接。

```bash
python /workspace/energy-carbon-code/run_forecast.py fit \
  --prepared /workspace/energy-carbon/data/prepared/erco_168_24_v1 \
  --model mlp \
  --device cuda \
  --output /workspace/energy-carbon/results/mlp_v1
```

默认最多 20 轮、patience 5；每轮只使用训练集和验证集。训练优化标准化空间的 MSE，最终报告原单位 MAE/RMSE。最佳模型只由验证损失选择，保存 `best.pt`、`history.csv` 与 `run.json`；`--epochs` 可覆盖配置且会记录实际值。

## 测试集：最后显式评估

固定配置、根据验证集选择模型后，再执行测试。不要根据测试结果反复调参：

```bash
python /workspace/energy-carbon-code/run_forecast.py evaluate \
  --run /workspace/energy-carbon/results/mlp_v1 \
  --device cuda \
  --output /workspace/energy-carbon/results/mlp_v1_test
```

评估朴素基线时将 `--run` 换成对应目录，省略 `--device cuda`。每个方法指定新的输出目录。输出 `metrics.json` 和 `predictions.csv`，指标按目标及预测步长分别记录，预测明细保留起点、目标时刻、步长、真实值及预测值。

所有输出目录拒绝覆盖。对源 CSV、准备清单的修改应开启新实验；评估会核对数据身份，不应通过修改哈希字段绕过检查。只加载本项目自己生成且未被篡改的 checkpoint，不从陌生来源下载 `.pt` 文件。

## 本地或服务器验证代码

安装训练依赖后，在仓库根目录执行：

```bash
cd /workspace/energy-carbon-code &&
python -m unittest discover -s tests -v
```

测试使用明确标注的合成数据验证时间边界、数据泄漏、标准化、基线、训练和保存加载；不能把其误差作为 ERCO 实验结果。完整真实数据训练、GPU 运行与论文级效果对比尚未因此完成。

2026-09-11 本地独立回归：19 项测试全部通过（无跳过）；环境为 Windows / Python 3.10.20 / NumPy 2.2.6 / pandas 2.3.3 / PyTorch 2.10.0+cpu。检查了中文路径输出、配置读取失败记录及评估配置一致性。云端 Python 为 3.10.16，本地测试不代替 Linux/CUDA 验证。

## 后续更新代码

代码修改并推送到 GitHub 后，在服务器运行：

```bash
git -C /workspace/energy-carbon-code pull --ff-only
git -C /workspace/energy-carbon-code rev-parse HEAD
```

不需要重新克隆，也不需要搬动 `/workspace/energy-carbon/data`。若服务器上修改过代码，先用 `git -C /workspace/energy-carbon-code status --short` 检查；不要使用强制重置覆盖本地改动。

## 核算与建模注意事项

1. 分能源求和排除 `total` 行，避免重复计数。发电侧因子的排放量与电量均取自 OGE，不把 EIA 电量替换为分母。
2. 当前主核算字段为 `co2_mass_kg_for_electricity`，未使用 `adjusted` 后缀字段；不是 CO₂e，也不是全生命周期因子。
3. 发电侧、消费侧因子分别保留，不视为相同核算口径。区域电网数据不等于园区实测数据。
4. 负净发电量保留并标记，具体成因尚未确定。单能源净发电量非正时，其计算因子留空；不要整表填零或因这些空值删除小时。
5. `observations.csv` 是同一时刻的历史观测，不是已经构造好的预测特征。未来实际电量、排放量、因子及质量标记都不能泄漏进预测输入。
6. 第一版默认设置见配置与设计文档。标准化仅拟合训练时段；第一版不预测单能源因子及分能源排放量，不从综合因子反推分能源排放，也暂不施加物理一致性约束。
7. OGE 数据为事后整理发布，当前结果不证明实时数据可得性；核算一致也不等于独立真实性验证或预测创新。

## 来源

- [OGE 官方说明与下载](https://singularity.energy/open-grid-emissions)
- [OGE 官方开源项目](https://github.com/singularity-energy/open-grid-emissions)
- [PyTorch 官方安装版本表](https://pytorch.org/get-started/previous-versions/)
- [PyTorch checkpoint 安全公告](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)：所列问题在 2.10.0 修复；版本选择不代表可安全加载任意不可信文件。
- [EIA 2023 上半年小时平衡表](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv)
- [EIA 2023 下半年小时平衡表](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv)

使用数据时按原数据提供方的许可与引用要求注明来源；本仓库未擅自为已有辅助代码声明新的开源许可证。
