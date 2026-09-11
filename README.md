# 课题 2-2：多能源发电与动态碳因子研究

代码版本管理仓库。当前阶段是 ERCO 2023 年小时观测表整理，还没有在这个仓库中完成预测模型训练或效果验证。

研究主线：分能源电量与 CO₂ 排放量 → 动态碳因子核算 → 未来碳因子及分能源序列预测。

## 文件用途与边界

- `build_erco_2023_observations.py`：读取 OGE 与 EIA 原始文件，核算、对齐并导出观测表。使用本地数据，不联网下载，不训练模型，不修改原始文件；已有输出目录会被拒绝覆盖。
- `work/render_docx.py`：文档渲染辅助脚本，不参与能源数据计算。需另行准备 `pdf2image`、Poppler 和 LibreOffice 等依赖；运行主数据脚本不需要这些软件。
- `requirements-data.txt`：主数据脚本的直接依赖。

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

脚本目前针对固定的 ERCO 2023 / OGE v0.8.0 数据校验值，不是任意年份通用的数据处理器。完整 EIA 合并和文件导出的云端执行结果仍待回传；本地已检查 OGE 转换及用户提供的前三行 EIA 时间对齐。

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
6. 训练前仍需确定预测目标、预测时长、时间顺序划分与可用信息边界；标准化和其他可学习的预处理仅拟合训练数据。
7. OGE 数据为事后整理发布，当前结果不证明实时数据可得性；核算一致也不等于独立真实性验证或预测创新。

## 来源

- [OGE 官方说明与下载](https://singularity.energy/open-grid-emissions)
- [OGE 官方开源项目](https://github.com/singularity-energy/open-grid-emissions)
- [EIA 2023 上半年小时平衡表](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv)
- [EIA 2023 下半年小时平衡表](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv)

使用数据时按原数据提供方的许可与引用要求注明来源；本仓库未擅自为已有辅助代码声明新的开源许可证。
