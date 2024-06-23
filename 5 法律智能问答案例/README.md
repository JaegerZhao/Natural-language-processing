# 司法阅读理解

## 1 任务目标

### 1.1 任务说明

裁判文书中包含了丰富的案件信息，比如时间、地点、人物关系等等，通过机器智能化地阅读理解裁判文书，可以更快速、便捷地辅助法官、律师以及普通大众获取所需信息。
本次任务覆盖多种法律文书类型，包括民事、刑事、行政，问题类型为多步推理，即对于给定问题，只通过单句文本很难得出正确回答，模型需要结合多句话通过推理得出答案。

### 1.2 评分要求

分数由两部分组成。首先，读懂已有代码并添加适量注释。使用已有代码在训练数据上进行训练，并且完成开发集评测，这部分占60%，评分依据为模型的开发集性能和报告，报告主要包括对于模型基本原理的介绍，需要同学阅读代码进行学习。
第二部分，进行进一步的探索和尝试，我们将在下一小节介绍可能的尝试，并在报告中汇报尝试的方法以及结果，这部分占40%。同学需要提交代码和报告，在报告中对于两部分的实验都进行介绍。

### 1.3 探索和尝试

- 使用2019年的[阅读理解数据集（CJRC）](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data)作为辅助数据集，帮助模型提高阅读理解能力
- 使用别的预训练语言模型完成该实验，例如THUNLP提供的[司法BERT](https://github.com/thunlp/OpenCLaP)
- 对于新的模型架构进行探索，例如加入图神经网络（GNN）来加强模型的推理能力

### 1.4 参考资料

- [CAIL2020——阅读理解](https://github.com/china-ai-law-challenge/CAIL2020/tree/master/ydlj)

## 2 数据集

### 2.1 数据说明

本任务数据集包括约5100个问答对，其中民事、刑事、行政各约1700个问答对，均为需要多步推理的问题类型。为了进行评测，按照9:1的划分，数据集分为了训练集和测试集。**注意** 该数据仅用于本课程的学习，请勿进行传播。

发放的文件为``train.json``和``dev.json``，为字典列表，字典包含字段为：

- ``_id``：案例的唯一标识符。
- ``context``：案例内容，抽取自裁判文书的事实描述部分。数据格式与HotpotQA数据格式一致，不过只包含一个篇章，篇章包括标题（第一句话）和切割后的句子列表。
- ``question``：针对案例提出的问题，每个案例只标注一个问题。
- ``answer``：问题的回答，包括片段、YES/NO、据答几种类型，对于拒答类，答案应该是"unknown"。
- ``supporting_facts``：回答问题的依据，是个列表，每个元素包括标题（第一句话）和句子编号（从0开始）。

同学们需根据案例描述和问题，给出答案及答案依据，最终会综合两部分的效果并作为评判依据，评价方法与HotpotQA一致。

我们提供基础的模型代码在`baseline`目录下

### 2.2 数据处理代码

​	本案例通过 `data_process.py` 对数据进行处理，该代码主要功能是读取问答数据文件，解析并转换数据为适合BERT模型输入的格式，并保存处理后的数据。通过定义`Example`和`InputFeatures`类，代码能够有效地组织和处理问答样本的数据。下面对该代码进行解释。

1. **导入必要的库**：

   - `argparse`：用于解析命令行参数。
   - `json`：用于解析JSON格式的数据。
   - `gzip`和`pickle`：用于数据的压缩和序列化。
   - `tqdm`：用于显示处理进度。
   - `BertTokenizer`：来自`transformers`库，用于文本的分词处理。
   - `os`：用于操作文件和目录。

2. **定义数据模型类**：

   - `Example`：用于存储单个问答样本的原始数据，包括问题ID、类型、文档标记、问题文本等信息。

     ```py
     class Example(object):
         def __init__(self, qas_id, qas_type, doc_tokens, question_text, sent_num, sent_names, sup_fact_id, para_start_end_position, sent_start_end_position, entity_start_end_position, orig_answer_text=None, start_position=None, end_position=None):
             # 初始化问答样本的数据
     ```

   - `InputFeatures`：用于存储转换后的特征数据，这些特征将用于模型的输入，包括经过分词处理的文档和问题、输入ID、掩码和段落ID等。

     ```py
     class InputFeatures(object):
         def __init__(self, qas_id, doc_tokens, doc_input_ids, doc_input_mask, doc_segment_ids, query_tokens, query_input_ids, query_input_mask, query_segment_ids, sent_spans, sup_fact_ids, ans_type, token_to_orig_map, start_position=None, end_position=None):
             # 初始化转换后的特征数据
     ```

3. **数据读取函数**(`read_examples`)：

   ```py
   def read_examples(full_file):
       # 打开并读取输入文件（JSON 格式）
       with open(full_file, 'r', encoding='utf-8') as reader:
           full_data = json.load(reader)
       ...
       return examples  # 返回所有样本列表
   ```

   - 读取JSON格式的HotpotQA数据文件。
   - 处理每个问题案例，包括分词、标注支持事实、确定答案位置等。
   - 创建`Example`对象的列表。

4. **特征转换函数**(`convert_examples_to_features`)：

   ```py
   def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
       # max_query_length = 50
       features = []
       for (example_index, example) in enumerate(tqdm(examples)):
           ...
           features.append(InputFeatures(...))
       return features  # 返回所有转换后的特征
   ```

   - 将`Example`对象转换为`InputFeatures`对象，包括使用Bert分词器处理文档和问题文本。
   - 处理答案文本，将字符位置转换为分词后的标记位置。
   - 创建句子跨度、支持事实ID等特征。
   - 对特征数据进行填充，以满足模型输入的序列长度要求。

5. **辅助函数**：

   - `check_in_full_paras`：检查答案是否在段落中。
   - `_largest_valid_index`：获取有效的最大索引。
   - `get_valid_spans`：获取有效的跨度列表。
   - `_improve_answer_span`：改进答案的标记跨度，以更好地匹配注释答案。

6. **主函数**(`if __name__ == '__main__':`)：

   - 解析命令行参数，包括输出文件路径、是否进行小写处理、序列最大长度等。
   - 加载Bert分词器。
   - 读取并处理数据，将原始数据转换为模型输入所需的特征数据。
   - 序列化特征数据并保存到文件。

### 2.3 数据处理

1. 下载Bert

   首先需要下载合适的分词器模型，如 `chinese_bert_wwm` ，下载[链接](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)。

2. 训练数据预处理

   执行以下命令，运行 `data_process.py` 对训练数据进行数据处理，得到 `train_example.pkl.gz` 与 `train_feature.pkl.gz`。

   -  `--example_output `将原始数据处理为示例数据，存储在 `Example` 对象中。

   - `--feature_output` 将示例数据转换为模型可以直接使用的特征数据，存储在 `InputFeatures` 对象中。

   ```
   !python baseline/data_process.py \
       --tokenizer_path ./models/chinese_bert_wwm \
       --full_data ./data/train.json \
       --example_output ./output/data/chinese-bert-wwm/train_example.pkl.gz \
       --feature_output ./output/data/chinese-bert-wwm/train_feature.pkl.gz 
   ```

3. 测试数据预处理

   执行以下命令，运行 `data_process.py` 对测试数据进行数据处理，得到 `dev_example.pkl.gz` 与 `dev_feature.pkl.gz`。

   ```
   !python baseline/data_process.py \
       --tokenizer_path ./models/chinese_bert_wwm \
       --full_data ./data/dev.json \
       --example_output ./output/data/chinese-bert-wwm/dev_example.pkl.gz \
       --feature_output ./output/data/chinese-bert-wwm/dev_feature.pkl.gz 
   ```

## 3 模型训练

### 3.1 中文预训练的模型

​	本实验需要通过自己寻找一个比较好的中文预训练模型用于训练数据，通过寻找找到了以下中文预训练模型。

1. Chinese-BERT-wwm

   **Chinese-BERT-wwm** (Whole Word Masking BERT for Chinese) 是一种基于 BERT 的中文预训练模型，专门针对中文文本进行了优化，仓库为 [ymcui/Chinese-BERT-wwm: Pre-Training with Whole Word Masking for Chinese BERT（中文BERT-wwm系列模型） (github.com)](https://github.com/ymcui/Chinese-BERT-wwm)。该模型的主要特点是采用了全词掩码策略，即在训练过程中掩码的是整个词而不是单个汉字。这种策略有助于模型更好地理解词汇级别的信息，而不仅仅是字符级别的信息。

   **模型特点：**

   - **Whole Word Masking**：在训练时，将整个词作为一个单元进行掩码，提高模型对词汇的理解能力。
   - **适用领域**：适用于各种中文自然语言处理任务，如文本分类、问答系统、文本生成等。
   - **使用场景**：适合需要精确理解中文词汇语义的任务。

2. Chinese-RoBERTa-wwm-ext

   **Chinese-RoBERTa-wwm-ext** 是在 `Chinese-BERT-wwm` 基础上进一步优化的模型。`RoBERTa` (Robustly optimized BERT approach) 是 BERT 的改进版，仓库为 [ymcui/Chinese-BERT-wwm: Pre-Training with Whole Word Masking for Chinese BERT（中文BERT-wwm系列模型） (github.com)](https://github.com/ymcui/Chinese-BERT-wwm)。该模型通过更大的数据集和更长时间的训练，提高了模型的性能。`Chinese-RoBERTa-wwm-ext` 继承了这些改进，并结合了全词掩码策略。

   **模型特点：**

   - **Robustly Optimized**：优化了训练过程和超参数，增强了模型的鲁棒性和性能。
   - **Whole Word Masking**：继续采用全词掩码策略，提高中文词汇理解。
   - **Extended Dataset**：使用了更大规模的数据集进行训练，提高了模型的泛化能力。
   - **使用场景**：适合更高精度要求的中文自然语言处理任务。

3. thunlp_ms

   **thunlp_ms** 是由清华大学自然语言处理与社会人文计算实验室 (THUNLP) 提供的一个民事文书BERT预训练模型。数据来源为全部民事文书，训练数据大小有2654万篇文书，词表大小22554，模型大小370MB。仓库为[thunlp/OpenCLaP: Open Chinese Language Pre-trained Model Zoo (github.com)](https://github.com/thunlp/OpenCLaP)，下载链接 https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/ms.zip 。

4. thunlp_xs

   **thunlp_xs** 是由清华大学自然语言处理与社会人文计算实验室 (THUNLP) 提供的一个刑事文书BERT预训练模型。数据来源为全部刑事文书，训练数据大小有663万篇文书，词表大小22554，模型大小370MB。仓库为[thunlp/OpenCLaP: Open Chinese Language Pre-trained Model Zoo (github.com)](https://github.com/thunlp/OpenCLaP)，下载链接 https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/xs.zip 。

### 3.2 模型训练代码

​	本案例通过 `run_cail.py` 训练和评估基于BERT模型的问答系统。以下是对代码中关键功能的详细介绍：

1. 导入必要的库和模块

   - `argparse` 用于解析命令行参数。

   - `os.path.join` 用于路径拼接。

   - `tqdm` 用于显示进度条。

   - `transformers.BertModel` 和 `transformers.BertConfig` 用于加载和配置BERT模型。

   - `transformers.optimization.AdamW` 和 `transformers.optimization.get_linear_schedule_with_warmup` 用于优化和学习率调度。

   - `torch` 和 `torch.nn` 为PyTorch库，用于构建和训练神经网络。

2. 全局配置和辅助函数
   - `set_seed` 函数设置随机种子，确保实验可复现。

3. 数据处理和分发函数
   - `dispatch` 函数将数据分发到GPU。

4. 损失计算函数
   - `compute_loss` 函数计算模型的损失值，包括起始位置、结束位置、类型预测和支持段落的损失。

5. 预测函数
   - `predict` 函数在模型评估阶段运行，使用模型对数据进行预测，并生成预测结果的字典。

6. 训练函数

   - `train_epoch` 函数执行一个训练周期，调用`train_batch`函数处理每个批次的数据。

   - `train_batch` 函数处理单个批次的数据，执行前向传播、损失计算、反向传播和优化器步骤。

7. 主函数

   在 `if __name__ == "__main__":` 块中，脚本执行以下操作：

   - 解析命令行参数。

   - 设置GPU数量和随机种子。

   - 初始化数据加载器和数据集。

   - 加载或配置BERT模型。

   - 初始化优化器、学习率调度器和损失函数。

   - 执行训练循环，包括训练和评估阶段。

8. 训练和评估循环

   - 训练循环包括多次迭代（由 `args.epochs` 指定），每个迭代都会遍历训练数据集。

   - 在训练过程中，如果设置了 `predict_during_train`，则在每个epoch的指定步骤进行评估。

   - 训练结束后，保存模型的状态字典，并记录训练和评估的损失。

9. 混合精度训练
   - 如果 `args.fp16` 为真，则使用Apex库的自动混合精度（AMP）功能来加速训练并减少内存使用。

10. 数据并行
    - 使用 `torch.nn.DataParallel` 实现模型的数据并行，可以在多个GPU上同时训练模型。

11. 日志记录
    - 训练过程中，通过打印语句记录损失和其他统计信息。

12. 结束训练
    - 当达到最大epoch数或满足其他退出条件时，脚本将停止训练。

### 3.3 模型训练

​	输入以下命令进行模型训练。

```py
!python baseline/run_cail.py \
    --name chinese-bert-wwm \
    --bert_model ./models/chinese_bert_wwm \
    --data_dir ./output/data/chinese-bert-wwm \
    --batch_size 2 \
    --eval_batch_size 32 \
    --lr 1e-5 \
    --gradient_accumulation_steps 4 \
    --seed 56 \
    --epochs 25
```

​	其中参数含义如下：

- `--name chinese-bert-wwm`: `--name` 指定了运行此次实验的名称或标识，这里设置为 `chinese-bert-wwm`。
- `--bert_model ./models/chinese_bert_wwm`: 指定BERT模型的路径。
- `--data_dir ./output/data/chinese-bert-wwm`: 指定存放数据的目录，数据可能包括预处理后的训练集、验证集等。
- `--batch_size 2`: 设置训练时每个batch的大小为2。
- `--eval_batch_size 32`: 设置评估时每个batch的大小为32。
- `--lr 1e-5`: 设置学习率为 `1e-5`，即0.00001。
- `--gradient_accumulation_steps 4`: 设置梯度累积的步数为4，这意味着每4个batch执行一次优化器更新。
- `--seed 56`: 设置随机种子为56，以确保结果的可复现性。
- `--epochs 25`: 设置训练的总周期数为25。

## 4 模型测试

