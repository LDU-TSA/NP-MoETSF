# NP-MoETSF

NP-MoETSF（Non-Prior Graph Learning for Multi-Expert Time Series Forecasting）是一个面向高维多变量时间序列预测的统一框架，能够在无先验图结构的条件下学习稀疏、可解释的变量依赖关系，并结合稀疏专家网络完成预测。

## 项目简介

本项目用于高维多变量时间序列预测任务，重点解决以下问题：

- 高维变量之间依赖关系未知且随时间变化
- 长预测跨度下的非平稳性与误差累积
- 全连接注意力带来的噪声传播与计算开销

## 运行示例
下面给出一个在 ETTh1.csv 数据集上的基准运行示例：

python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" \
  --strategy-args '{"horizon": 96}' \
  --model-name "NPMoETSF.NPMoETSF" \
  --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 2, "patch_len": 48, "patience": 5, "seq_len": 512}' \
  --deterministic "full" \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "ETTh1/NPMoETSF"

## 安装依赖

请直接执行：

```bash
pip install -r requirements.txt

