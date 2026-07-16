# Ascend-NPU support

1. 修改代码，兼容NPU设备
要求torch_npu, python存在
`npu-smi info` 查看NPU信息

```2. 进入NPU环境
```bash
./run_docker.sh
```
3. smoke test
```bash
ASCEND_RT_VISIBLE_DEVICES=7 python \
    /workspace/dag_kv_ascend/PathWeaver/tests/smoke_qwen3_backbone.py \
    --device npu \
    --max-new-tokens 16
```
4. DAG-KV
安装一些依赖包：
```bash
pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --trusted-host mirrors.tuna.tsinghua.edu.cn \
  --no-cache-dir \
  evaluate rouge_score
```
5. 运行完整测试
- dag_kv_ascend_qwen4B.md
- dag_kv_ascend_qwen14B.md