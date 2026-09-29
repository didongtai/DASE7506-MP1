# v6 最终合规审计

审计对象为 `runs/final-candidate-depth6-v6/checkpoint.pt`，实现为
`student_fast`。本文件只记录在本地文件与已保存的实验记录中能够核验的结论；
它不替代课程方的最终审核。

| 要求 | 结论 | 可核验证据 |
|---|---|---|
| 固定数据、tokenizer、评测器 | 通过 | `data/manifest.json` 的四个 SHA-256 均与当前文件一致；冻结记录中的 `common.py`、`evaluate.py` 哈希与当前文件一致。 |
| 仅训练集衍生的运行时资产 | 代码与记录通过 | `student_fast.py`、`student_cache.py` 和 `student.py` 没有读取 validation/test 文件、网络或预训练模型的路径；`EXPERIMENTS.md` 记录 n-gram 表来自训练集。 |
| 因果性、窗口独立性、概率归一化 | 通过 | `unittest discover -s tests -v` 于 2026-09-29 通过 15/15；`freeze.json` 记录未来后缀、批样本和重复窗口的最大差均为 0，归一化最大误差为 `9.52e-7`。 |
| CPU 时间不超过基线 5 倍 | 通过 | CPU/FP32/4 线程的三次完整测试为 26.4081、29.4858、34.7009 秒；较快基线为 8.8254 秒，5 倍限额为 44.1268 秒，最坏比值为 3.93196。 |
| 峰值评测内存不超过 4 GiB | 通过 | 三次测试中最大工作集为 2.20684 GiB。 |
| 未压缩推理资产不超过 64 MiB | 通过 | checkpoint 加 `student_fast.py`、`student_cache.py`、`student.py` 为 63,905,769 字节，即 60.945 MiB。 |
| 测试分数可复现 | 通过 | 三次 BPB 均为 `1.5189136237091485`，每个窗口的 NLL 完全一致；测试前后均验证冻结文件哈希。 |
| v6 冻结后才进行其测试 | 通过 | `freeze.json` 记录冻结时间为 2026-09-29 12:12:57 UTC，且字段 `test_evaluated_before_this_freeze` 为 false；选择指标为 validation BPB 1.5022012895。 |
| 最终提交材料 | 尚未完成 | 工作区未发现不超过 10 页的报告，也没有不可变代码链接和匹配 checkpoint bundle 链接；课程网页提交同样无法由本地审计验证。 |
| 整个历史开发过程未受公开测试影响 | 无法事后认证 | `EXPERIMENTS.md` 已披露较早候选在继续开发前曾运行公开 test。当前 v6 的冻结顺序正确，但这不能消除历史流程的疑义；应完整保留该披露并由课程方判断。 |

## 冻结身份

```text
checkpoint.pt    7e1a556f26f047b02d78d97b74564598913f1b18b7318f2275c846124666555c
student_fast.py  2455f59b6221522483d07d190329b376002bbecc68a15389b16e1da2f219ce6f
student_cache.py 214b6711e62db78ef571ff295b2ce46510893339545b3f9f862da67a82f0f5c4
student.py       5d480d349cee7c913ce2757ba30cd99daebd39c7769b0ef3fb3d46d281796f91
```

完整重复测试材料在 `audit/v6/frozen-test-repeat/summary.json`，冻结时的合约检查在
`audit/v6/freeze.json`。提交时只应使用本文件所列的 v6 checkpoint，不能以已超过 CPU
时间限制的 v5 checkpoint 替代。
