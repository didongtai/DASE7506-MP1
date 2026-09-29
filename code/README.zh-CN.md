# MP1 代码——安装与使用

作业内容、考核方式、截止日期和同伴复核规则见[项目指南](../GUIDE.zh-CN.md)。本 README 介绍运行方法和技术规则。原始材料中只有这两份说明文档。

以下所有命令均在 **`code/`** 目录下运行。数据和分词器已包含在材料中。无需 API 密钥、预训练权重或额外下载数据集；安装依赖后，训练和评测均可离线运行。

## 1. 安装

使用 **Python 3.12**。从解压后的项目目录开始：

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

在 Windows PowerShell 中，改用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。

只为你使用的**一种**设备安装 PyTorch：

```bash
# Linux/Windows CPU：推荐；不需要 GPU
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

如果有驱动兼容的 NVIDIA GPU，则**改用**以下命令：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

在 macOS 上，从默认 PyPI 软件源安装 `torch==2.7.1`，并使用 CPU 运行。安装 PyTorch 后，安装其余依赖并检查模型：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

原始材料记录了 Python 3.12、PyTorch 2.7.1+cpu 下的 Linux CPU 验证。
学生 v5 的本机实测环境为 Windows 11、Python 3.12.10、PyTorch 2.7.1+cu128；
评分使用 CPU、FP32、4 线程，GPU 训练使用 RTX 5060 Laptop GPU。尚未测量 macOS。

## 2. 训练与评测

**快速安装检查**——训练 10 步，然后对完整测试集运行评测：

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

这只用于检查流程能否正常运行；所得分数**不是**完整基线的分数。每次训练都必须使用一个新的输出目录。

**完整基线**——进行 1,200 次训练参数更新，然后评测：

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

基线模型有 4 个 GPT 块、宽度为 128、4 个注意力头，共 **1,088,256 个参数**；其测试集成绩约为 **2.10 BPB**。在作为参考的四线程 Xeon Platinum 8457C 上，测得训练约耗时 **311 秒**，评分约耗时 **5.92 秒**，均不计安装和加载时间。这些只是参考测量值，并非笔记本电脑上的保证用时或固定时间额度。

**你的模型**——修改 `student.py` 及所需的辅助文件，然后运行：

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# 在测试前冻结最终方法：
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

训练会写出 `checkpoint.pt` 和 `metrics.json`。评测会写出 `test_cpu_fp32.json`（或与实际设备、数据划分相对应的文件名）以及各窗口的损失。提交完整测试集 JSON 中的 **`bpb`** 值，不要提交 token 困惑度或验证集 BPB。评测默认使用 FP32。GPU 运行时可添加 `--device cuda`；训练可以使用 BF16，但参与排名的评测必须使用 FP32，且结果必须能在 CPU 上复现。提供的 CUDA 运行程序将 PyTorch 的显存分配上限设为 20 GB；驱动程序开销另计。

## 3. 文件与模型接口

| 文件 | 用途 |
|---|---|
| `model.py`、`configs/baseline.json` | 可运行的基线；保留用于对照。 |
| `student.py`、`train.py` | 你的模型构造函数和训练方案；可按需增加辅助代码。 |
| `common.py`、`evaluate.py` | 固定的数据检查、窗口划分和评分器；保持不变。 |
| `data/` | 提供的数据划分、分词器和数据集哈希；保持不变。 |
| `tests/test_contract.py` | 检查模型的因果性、概率归一化、样本独立性和梯度。 |
| `RUN_LOG_TEMPLATE.csv` | 可选的实验日志模板。 |
| `PACKAGE_MANIFEST.json` | 发布文件的哈希；原文称路径相对于包含 `code/` 和 `guide/` 的包根目录。 |

- `build_model(config)` 返回一个 `context=256` 的 PyTorch 模型。
- 提供的训练程序调用 `forward(ids)`，要求返回未经归一化的 logits；评分器调用 `predict_log_probs(ids)`，要求返回有限且已归一化的自然对数概率。两者输出形状均为 `[batch, time, 2048]`。
- 位置 `t` 的预测只能使用截至 `t` 的已观察前缀。不同窗口、样本和评分过程之间必须重置临时状态。从训练数据得到的小型资产可以跨窗口复用；评测前缀产生的状态不能跨窗口复用。
- Checkpoint 会记录实现模块和配置。须提供该模块及所有必需资产，使评测器能够重建提交的预测模型。直接评测不需要优化器状态。
- 在指南限制内，可以改变训练时长、架构、优化器、正则化、基于自身训练权重的平均方法以及模型集成。须记录所有随机种子、处理过的训练目标数、checkpoint 继承关系和方案搜索成本；复用 checkpoint 不会抹去其已有训练成本。没有规定必须使用某个随机种子，也没有规定最低分数提升幅度。

## 4. 基准与资源测量

**固定的评分方式。** 协议 `7506-mp1-wt2-v2`：使用 WikiText-2 原始文本、仅用训练集拟合的 BPE-2048 分词器，以及相互独立、每个窗口最多包含 256 个预测目标的窗口；最后一个不足长度的窗口也计入。每个数据划分中，除首个 token 外的每个目标都恰好评分一次。相邻输入窗口共用一个边界 token，但不传递状态。BPB 等于所有下一个 token 的负对数（以 2 为底）概率之和，除以该数据划分原始文本的完整 UTF-8 字节数；分母包含首个 token 对应的字节。

| 数据划分 | 评分目标数 | UTF-8 字节数 |
|---|---:|---:|
| 验证集 | 376,599 | 1,148,007 |
| 测试集 | 428,405 | 1,292,013 |

所有开发工作，以及 checkpoint 或概率混合方案的选择，都应使用验证集。模型权重、统计量和检索条目只能来自训练文本。测试文本公开是为了支持结果复现，不能用于调节方法。方法冻结后，可以对同一个预测模型重复评测，以测量时间或复现结果。token 困惑度不能直接与已发表的词级困惑度比较。

对同一个已冻结的预测模型测量以下三个限制：

- **CPU 时间 ≤ 基线的 5 倍：**
- **峰值内存 ≤ 4 GiB：**
- **未经压缩的推理资产 ≤ 64 MiB：**

## 5. 准备提交与复现同学的结果

截止日期和网站提交流程见[指南](../GUIDE.zh-CN.md)。不可变的代码仓库中应包含：

- **报告：**包括图、表和参考文献在内，最多 10 页。
- **复现说明。**

网站上的最终提交必须链接到此代码版本以及与之匹配的完整 checkpoint 文件包。网站会自动生成 GitHub issue 的 JSON。所有推理资产均须可下载，以供核验。

复核其他同学的结果时，获取对方准确的代码版本和 checkpoint，遵循其安装说明，并使用提供的评测器对冻结模型进行完整测试集评分：

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

将复现的 BPB 与对方报告的分数比较。提交 **Peer Review Report** 时必须填写复现分数；也可以附上运行命令、环境、分数差异以及证据或日志链接。差异由教师裁定。根据公布的评分政策，在七天公开复核期内，经确认的分数差异报告可获得额外加分。

## 6. 数据来源说明

WikiText-2 由 Stephen Merity、Caiming Xiong、James Bradbury 和 Richard Socher 在论文 [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) 中提出。文本由维基百科贡献者撰写。[上游数据集](https://huggingface.co/datasets/Salesforce/wikitext) 标明了 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 和 [GNU 自由文档许可证](https://www.gnu.org/licenses/fdl-1.3.html)；重新分发数据时须保留这些声明。

提供的 `wikitext-2-raw-v1` 数据划分对应修订版本 `b08601e04326c79dfdd32d625aee71d232d685c3`。各行以单个换行符连接，并编码为 UTF-8；分词器仅使用训练文本拟合。数据集哈希见 `data/manifest.json`。这些数据声明并未为周围的课堂代码指定新的许可证。

## 7. 学生实现

冻结后的 v5 模型由 691 万参数、8 层 SwiGLU Transformer、仅使用训练集构建的
2-gram 至 5-gram 紧凑表，以及只使用当前窗口已读前文的神经缓存组成。最终 checkpoint：
`runs/final-candidate-depth8-v5/checkpoint.pt`，验证集/测试集 BPB 分别为
**1.482692/1.499413**。checkpoint 为 63.928 MiB，四线程 CPU 测试耗时 43.01 秒
（基线的 4.91 倍），评测进程峰值工作集为 2.206 GiB。这是初次测量，后续重复测量未通过时间限制。

9 月 29 日合规复测得到完全相同的测试 BPB：v5 耗时 42.9658 秒，同环境前后两次基线
分别为 8.6980/8.6952 秒，倍率至多 4.9414；峰值工作集 2.207 GiB。
checkpoint 加两个推理源码文件合计 63.955 MiB。
[COMPLIANCE.md](COMPLIANCE.md) 记录了详细检查、1～3 token 输入缺陷、修正的训练成本
以及尚需补齐的提交材料；不能据此认为完整提交已全部达标。

**最新重复测试：CPU 耗时超过 5 倍限制。** 保持冻结模型不变连续运行 3 次，BPB 均为
1.4994128611199007，逐窗口损失也完全一致；耗时分别为 34.1322/34.1866/34.2711 秒，
前后两次基线为 6.0920/6.4525 秒。即使采用较慢基线，三次也全部超限。
v5 耗时中位数除以基线耗时均值为 5.4505 倍。内存和资产体积仍达标。
详见[完整重复测试结果](audit/v5/repeated-evaluation-20260929/RESULTS.md)。

推理必须同时提供 `student_cache.py` 和 `student.py`；复制目标训练代码在
`student_pointer.py`，没有增加模型参数。评测仍使用原始 FP32 scorer。
`runs/` 和 `*.pt` 被 Git 忽略，checkpoint 需要单独上传。

**AI 协助声明：** OpenAI Codex 实质参与了方案设计、代码实现、训练、实验、调试、
资源测量和文档编写。学生需要审阅并理解提交的实现，保留本声明。

训练继承关系、消融实验、完整复现命令、资源测量、冻结哈希和 AI 协助声明见
[EXPERIMENTS.md](EXPERIMENTS.md)。提交前请自行审阅并理解所有改动。

**当前满足耗时限制的候选版本：** `runs/final-candidate-depth6-v6/checkpoint.pt`，
使用 `student_fast`，验证集/测试集 BPB 为 1.502201/1.518914。三次完整测试耗时
26.4081、29.4858、34.7009 秒，均低于同轮基线的 5 倍。详见
[v6 重复测试结果](audit/v6/frozen-test-repeat/RESULTS.md)。

---

*译文供阅读参考；如有歧义，以英文原文 [README.md](README.md) 为准。原文在第 4 节列出三项资源限制后，没有给出进一步的具体测量命令；本译文也未补写。*
