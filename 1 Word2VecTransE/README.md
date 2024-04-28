# 作业一 Word2Vec&TranE的实现

## 1 任务目标

### 1.1 案例简介

Word2Vec是词嵌入的经典模型，它通过词之间的上下文信息来建模词的相似度。TransE是知识表示学习领域的经典模型，它借鉴了Word2Vec的思路，用“头实体+关系=尾实体”这一简单的训练目标取得了惊人的效果。本次任务要求在给定的框架中分别基于Text8和Wikidata数据集实现Word2Vec和TransE，并用具体实例体会词向量和实体/关系向量的含义。


### 1.2 Word2Vec实现

在这个部分，你需要基于给定的代码实现Word2Vec，在Text8语料库上进行训练，并在给定的WordSim353数据集上进行测试

WordSim353是一个词语相似度基准数据集，在WordSim353数据集中，表格的第一、二列是一对单词，第三列中是该单词对的相似度的人工打分(第三列也已经被单独抽出为ground_truth.npy)。我们需要用我们训练得到的词向量对单词相似度进行打分，并与人工打分计算相关性系数，总的来说，越高的相关性系数代表越好的词向量质量。

我们提供了一份基于gensim的Word2Vec实现，请同学们阅读代码并在Text8语料库上进行训练, 关于gensim的Word2Vec模型更多接口和用法，请参考[2]。

由于gensim版本不同，模型中的size参数可能需要替换为vector_size（不报错的话不用管）

运行`word2vec.py` 后，模型会保存在`word2vec_gensim`中，同时代码会加载WordSim353数据集，进行词对相关性评测，得到的预测得分保存在score.npy文件中
之后在Word2Vec文件夹下运行 ``python evaluate.py score.npy``, 程序会自动计算score.npy 和ground_truth.npy 之间的相关系数得分，此即为词向量质量得分。

1. 任务

   - 运行`word2vec.py`训练Word2Vec模型, 在WordSim353上衡量词向量的质量。

   - 探究Word2Vec中各个参数对模型的影响，例如词向量维度、窗口大小、最小出现次数。

   - （选做）对Word2Vec模型进行改进，改进的方法可以参考[3]，包括加入词义信息、字向量和词汇知识等方法。请详细叙述采用的改进方法和实验结果分析。


2. 快速上手（参考）

   在Word2Vec文件夹下运行 ``python word2vec.py``, 即可成功运行, 运行生成两个文件 word2vec_gensim和score.npy。


### 1.3 TransE实现

这个部分中，你需要根据提供的代码框架实现TransE，在wikidata数据集训练出实体和关系的向量表示，并对向量进行分析。

在TransE中，每个实体和关系都由一个向量表示，分别用$h, r,t$表示头实体、关系和尾实体的表示向量，首先对这些向量进行归一化
$$
h=h/||h|| \\
r=r/||r||\\
t=t/||t||
$$


则得分函数(score function)为
$$
f(h,r,t)=||h+r-t||
$$
其中$||\cdot||$​表示向量的范数。得分越小，表示该三元组越合理。

在计算损失函数时，TransE采样一对正例和一对负例，并让正例的得分小于负例，优化下面的损失函数

$$
\mathcal{L}=\sum_{(h,r,t)\in\Delta,(h',r',t')\in\Delta'}\max\left( 0, [\gamma+f(h,r,t)-f(h',r',t')]\right)
$$
其中$(h,r,t), (h',r',t')$分别表示正例和负例，$\gamma$​是​一个超参数(margin)，用于控制正负例的距离。

1. 任务

   - 在文件`TransE.py`中，你需要补全`TransE`类中的缺失项，完成TransE模型的训练。需要补全的部分为：
     - `_calc()`：计算给定三元组的得分函数(score function)
     - `loss()`：计算模型的损失函数(loss function)

   - 完成TransE的训练，得到实体和关系的向量表示，存储在`entity2vec.txt`和`relation2vec.txt`中。

   - 给定头实体Q30，关系P36，最接近的尾实体是哪些？

   - 给定头实体Q30，尾实体Q49，最接近的关系是哪些？

   - 在 https://www.wikidata.org/wiki/Q30 和 https://www.wikidata.org/wiki/Property:P36 中查找上述实体和关系的真实含义，你的程序给出了合理的结果吗？请分析原因。

   - （选做）改变参数`p_norm`和`margin`，重新训练模型，分析模型的变化。


2. 快速上手（参考）

   在TransE文件夹下运行 ``python TransE.py``, 可以看到程序在第63行和第84行处为填写完整而报错，将这两处根据所学知识填写完整即可运行成功代码（任务第一步），然后进行后续任务。

### 1.4 评分标准

请提交代码和实验报告，评分将从代码的正确性、报告的完整性和任务的完成情况等方面综合考量。

### 1.5 参考资料

[1] https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

[2] https://radimrehurek.com/gensim/models/word2vec.html

[3] A uniﬁed model for word sense representation and disambiguation. in Proceedings of EMNLP, 2014.

## 2 Word2Vec 实现

### 2.1 Word2Vec 基本原理

​	Word2Vec 是一种用于生成单词嵌入（word embeddings）的计算模型，由 Google 的研究人员在 2013 年提出。它属于无监督学习算法，可以从大量文本数据中学习到 **单词的向量表示** 。Word2Vec 的核心思想是捕捉 **单词之间的语义和句法** 关系，使得在向量空间中，语义上相似的单词会有相近的向量表示。

​	Word2Vec 通过 **滑动窗口** 实现，word2vec使用一个固定大小的窗口沿着句子滑动。在每一个窗口中，中间的词作为目标词，其他的词作为上下文。下图为滑动窗口为5时的训练示例。

![image-20240427180703325](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240427180703325.png)

​	Word2Vec 主要有两种架构：连续词袋（Continuous Bag of Words, CBOW）和跳跃式模型（Skip-Gram）。

![image-20240427174905030](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240427174905030.png)

1. **连续词袋（CBOW）**

   ![image-20240427175610257](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240427175610257.png)

   - 在 CBOW 模型中，目标是 **根据上下文单词预测目标单词** 。具体来说，模型会取目标单词周围的上下文单词（例如，前后各 N 个单词），将这些上下文单词的向量表示平均或加权求和，然后使用这个综合向量来预测目标单词。
   - 通过词袋假设，假设上下文词的顺序不影响模型的预测结果。
   - 这种模型适合于较小的词汇表和较短的上下文窗口，可以快速训练。

2. **跳跃式模型（Skip-Gram）**

   ![image-20240427175842463](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240427175842463.png)

   - 在 Skip-Gram 模型中，目标是 **使用单个目标单词来预测整个上下文中的单词** 。也就是说，给定一个目标单词，模型需要预测它可能出现的上下文。
   - Skip-Gram 通常比 CBOW 更准确，特别是对于更大词汇表和更长的上下文窗口。它在小数据集上表现更好，但在处理大量数据时训练速度较慢。

### 2.2 代码实现

​	Word2Vec 实验中，已给出了完整的实验代码，无需自己补充。实验使用 `gensim` 库来训练一个 Word2Vec 模型，并对其进行评估的示例。代码包括数据加载、模型训练、模型保存以及在标准测试集上的评估，下面介绍代码的主要部分。

1. 训练 Word2Vec 模型

   实验使用 `word2vec.Text8Corpus` 函数加载文本数据集。然后通过 `gensim.models.Word2Vec` 创建 Word2Vec 模型进行训练，并传入了加载的语料库 `sents`。

   ```py
   # loading dataset
   sents = word2vec.Text8Corpus(data_path)
   
   # training word2vec
   model = gensim.models.Word2Vec(sents,
                                  vector_size=200,
                                  window=10,
                                  min_count=10,
                                  workers=multiprocessing.cpu_count())
   ```

   - `vector_size` 参数设置为 200，表示每个单词向量的维度。
   - `window` 参数设置为 10，表示考虑的上下文窗口大小。
   - `min_count` 参数设置为 10，表示忽略频率低于 10 的单词。
   - `workers` 参数设置为 `multiprocessing.cpu_count()`，表示使用 CPU 的核心数进行并行计算。

   这里未设置模型参数 `sg`，默认使用CBOW(`sg=0`)进行训练。

2. 模型保存

   ```py
   # saving to file
   model.save("word2vec_gensim")
   model.wv.save_word2vec_format("word2vec_org",
                                 "vocabulary",
                                 binary=False)
   
   print ("Total time: %d s" % (time() - t))
   ```

   - 使用 `model.save` 方法将训练好的 Word2Vec 模型保存到文件 "word2vec_gensim"。
   - 使用 `model.wv.save_word2vec_format` 方法将模型的词向量以 Word2Vec 格式保存到 "word2vec_org" 文件中，同时保存词汇表到 "vocabulary" 文件中，并且设置 `binary=False` 以保存为文本格式。

3. 模型测试

   在 wordsim353 数据集上测试模型。

   ```py
   # testing on wordsim353
   sims = []
   ground_truth = []
   with open('wordsim353/combined.csv') as f:
       for line in f.readlines()[1:]:
           l = line.strip().split(',')
           if l[0] in model.wv.key_to_index and l[1] in model.wv.key_to_index: # 过滤掉不在词表内的词
               sims.append(model.wv.similarity(l[0], l[1])) # 模型打分
               ground_truth.append(float(l[2]))
   
   np.save('score.npy', np.array(sims))
   np.save('ground_truth.npy', np.array(ground_truth))
   ```

   - 初始化两个空列表 `sims` 和 `ground_truth`，分别用于存储模型计算的相似度分数和 ground truth（真实）分数。
   - 打开 'wordsim353/combined.csv' 文件，并读取除标题外的所有行。
   - 对每一行进行处理，首先去除前后的空白字符，然后使用 `split(',')` 按逗号分割。
   - 检查两个单词是否都在模型的词汇表中，如果是，则使用 `model.wv.similarity` 方法计算这两个单词的相似度，并将其添加到 `sims` 列表中。
   - 同时，将该行的第三个元素（相似度分数）转换为浮点数，并添加到 `ground_truth` 列表中。
   - 使用 `numpy.save` 方法将 `sims` 和 `ground_truth` 列表保存为 `.npy` 文件，这些文件可以用于后续的评估和分析。

### 2.3 项目训练

​	本次训练采用在本地进行训练，下面介绍具体训练流程。

1. 数据集

   本实验采用了两个数据集，在 **Text8语料库** 上进行训练，并在给定的 **WordSim353数据集** 上进行测试。

   - Text8语料库

     Text8 数据集仅包含小写英文字母、空格以及数字被转换成了对应的英文单词，去除了所有的标点符号和非英文字符。其包含了从 Wikipedia 抓取的前 100,000,000 个字符，部分内容如下：

     ```
      anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organization of society it has also been taken up as a positive label by self defined anarchists the word ...
     ```

   - WordSim353数据集

     WordSim-353 数据集是一个用于评估英语单词对相似性或相关性的测试集合。它包含了两组英语单词对 set1 和 set2，以及由人类评估者赋予的相似性分数。数据集文件组成如下：

     ```
     ──wordsim353
             combined.csv
             combined.tab
             instructions.txt
             set1.csv
             set1.tab
             set2.csv
             set2.tab
     ```

     - 第一组（set1）包含 153 对单词对，以及由 13 名评估者赋予的相似性分数。

       ```
       Word 1	Word 2	Human (mean)	1	2	3	4	5	6	7	8	9	10	11	12	13	
       love	sex	6.77	9	6	8	8	7	8	8	4	7	2	6	7	8	
       tiger	cat	7.35	9	7	8	7	8	9	8.5	5	6	9	7	5	7	
       tiger	tiger	10.00	10	10	10	10	10	10	10	10	10	10	10	10	10	
       book	paper	7.46	8	8	7	7	8	9	7	6	7	8	9	4	9
       ...
       ```

     - 第二组（set2）包含 200 对单词对，由另外 16 名评估者评估。

       ```
       Word 1	Word 2	Human (mean)	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16
       energy	secretary	1.81	1	0	4	2	4	5	1	1	1	0	1	1	4	0	2	2
       secretary	senate	5.06	7	1	7	4	4	7	1	3	4	8	4	5	6	7	6	7
       energy	laboratory	5.09	7	1	7.5	4	6	7	4	6	1	2	4	3	7	9	6	7
       computer	laboratory	6.78	8	5	8	7	6	9	6	7	6	7.5	4	5	8	9	7	6
       ...
       ```

     - 结合（combined）将 set1 和 set2 集合并成一个单一的集合

       ```
       Word 1	Word 2	Human (mean)
       love	sex	6.77
       tiger	cat	7.35
       tiger	tiger	10.00
       book	paper	7.46
       ...
       ```

2. 代码训练

   修改代码中 `data_path` 的路径，改为 `text8` 的本地相对路径。跳转到 Word2Vec 文件夹下，输入以下指令开始训练。

   ```cmd
   python word2vec.py
   ```

   看到以下内容，说明训练成功开始。

   ![image-20240427220442219](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240427220442219.png)

   训练结束后，显示如下内容。

   ```
   2024-04-25 11:06:10,798 : INFO : Word2Vec lifecycle event {'msg': 'training on 85026035 raw words (61669556 effective words) took 66.9s, 922157 effective words/s', 'datetime': '2024-04-25T11:06:10.798335', 'gensim': '4.3.2', 'python': '3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19043-SP0', 'event': 'train'}
   2024-04-25 11:06:10,799 : INFO : Word2Vec lifecycle event {'params': 'Word2Vec<vocab=47134, vector_size=200, alpha=0.025>', 'datetime': '2024-04-25T11:06:10.799334', 'gensim': '4.3.2', 'python': '3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19043-SP0', 'event': 'created'}
   2024-04-25 11:06:10,800 : INFO : Word2Vec lifecycle event {'fname_or_handle': 'word2vec_gensim', 'separately': 'None', 'sep_limit': 10485760, 'ignore': frozenset(), 'datetime': '2024-04-25T11:06:10.800330', 'gensim': '4.3.2', 'python': '3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19043-SP0', 'event': 'saving'}
   2024-04-25 11:06:10,812 : INFO : not storing attribute cum_table
   2024-04-25 11:06:10,929 : INFO : saved word2vec_gensim
   2024-04-25 11:06:10,958 : INFO : storing vocabulary in vocabulary
   2024-04-25 11:06:11,009 : INFO : storing 47134x200 projection weights into word2vec_org
   Total time: 77 s
   ```

   最终存储 `vocabulary`， `score.npy`，`ground_truth.npy` 文件。

3. 代码测试

   本实验通过 `evaluation.py` 计算 **Spearman 相关系数** ，它用于评估 Word2Vec 模型生成的单词相似度分数与真实分数之间的相关性。

   ```py
   def eval(submit_file):
       sims = np.load(submit_file[1])
       score = np.load('ground_truth.npy')
       # 计算Spearman相关系数
       spcor = spearmanr(score, sims)[0]
       # 以下返回值主要用于aistudio的比赛，但是本次实验不设置比赛，大家只用看score的值
       return {
           "score": spcor,              #替换value为最终评测分数
           "errorMsg": "success",      #错误提示信息，仅在code值为非0时打印
           "code": 0,                  #code值为0打印score，非0打印errorMsg
           "data": [
               {
                   "score": spcor 
               }
           ]
       }
   ```

   输入如下指令，运行 `evaluation.py` 测试代码，进行测试。

   ```cmd
   python evaluation.py score.npy
   ```

   得到的结果如下。

   ```
   {"score": 0.6886039757987104, "errorMsg": "success", "code": 0, "data": [{"score": 0.6886039757987104}]}
   ```

   得到了 score 的值为 0.6886。

### 2.4 参数对模型的影响

​	本部分实验从 **词向量维度**、**窗口大小**、**最小出现次数** 三个维度对比其对模型的影响。

1. **词向量维度**（vector_size）

   实验分别选择了 50,100,150,200,250,300 的词向量维度，其他值选择默认值，即 window=10，min_count=10，进行实验。记录得到 score 与耗费的时间 time，结果如下。

   ```py
   score = [0.6606661257461373, 0.692237913064484, 0.6924479511296131, 0.6951971378884105, 0.6807892808556052, 0.6903557686853921]
   time = [21, 57, 75, 96, 98, 51]
   ```

   用折线图的形式表示，结果如下。

   ![image-20240428003905281](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240428003905281.png)

   可以看出，随着词向量维度的增加，训练时间逐步增加，直到300出现骤降。而 Score 在 0.66-0.7 之间波动，在词向量维度为200时，Score达到最大值 0.6952。

2. **窗口大小**（window）

   实验分别选择了 5,10,15,20,25,30 的窗口大小，其他值选择默认值，即 vector_size=200，min_count=10，进行实验。记录得到 score 与耗费的时间 time，结果如下。

   ```py
   score = [0.6440411455933666, 0.684678346325392, 0.7034369918501572, 0.7132489339996146, 0.7193900469514843, 0.7170033849093149]
   time = [50, 49, 44, 45, 42, 47]
   ```

   用折线图的形式表示，结果如下。

   ![image-20240428011049995](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240428011049995.png)

   可以看出，随着窗口大小的增加，训练时间在40-50之间波动。而 Score 在逐步上升，在窗口大小为25时，Score达到最大值 0.7194。

3. **最小出现次数**（min_count）

   实验分别选择了 5,10,15,20,25,30 的最小出现次数，其他值选择默认值，即 vector_size=200，window=10，进行实验。记录得到 score 与耗费的时间 time，结果如下。

   ```py
   score = [0.6927590997286565, 0.6934430495037492, 0.6904133055101986, 0.6893439535738417, 0.704707239257633, 0.6936508753087103]
   time = [44, 39, 36, 35, 33, 33]
   ```

   ![image-20240428012657140](https://raw.githubusercontent.com/ZzDarker/figure/main/img/image-20240428012657140.png)

   可以看出，随着最小出现次数的增加，训练时间在逐步减小。而 Score 在0.69-0.71之间波动，在最小出现次数为25时，Score达到最大值 0.7047。

### 2.5 实验改进

​	本实验改进，和在 vector_size=200, window=10, min_count=10 参数下的模型，进行比较。其运行时长为 77 s ， score 为 0.6886 。

1. 使用 Skip-Gram 进行训练

   在 `gensim.models.Word2Vec` 中添加参数 `sg=1` ，表示使用 Skip-Gram 进行训练，其他参数保持不变，即 vector_size=200, window=10, min_count=10。训练结果如下。

   ```cmd
   Total time: 147 s
   "score": 0.7124438700653573
   ```

   可以看到训练时间变得更长，相比于使用 CBOW 模式训练，score 从 0.6886 提升到了 0.7124。

2. 增长训练轮次

   默认训练 5 轮，增加到 10 轮，训练结果如下。

   ```cmd
   Total time: 104 s
   "score": 0.7277341821071591
   ```

   可以看到增长训练轮次，有效的提升了 score 值，同时增加了训练时长。

## 3 TransE 实现

### 3.1 TransE 原理

​	TransE 是一种知识图谱嵌入（Knowledge Graph Embedding, KGE）模型，由 Antoine Bordes 等人在 2013 年提出。它的目标是将知识图谱中的实体（entities）和关系（relations）映射到连续的向量空间中，使得可以通过向量运算来推断实体间的关系。

​	TransE 模型的核心思想是，对于一个三元组（头实体-关系-尾实体），例如 `(华盛顿，是美国的首都，美国)`，TransE 试图通过将头实体的向量与关系向量相加，然后减去尾实体的向量，来使得三元组的得分尽可能高。换句话说，TransE 模型试图在向量空间中捕捉头实体和尾实体通过关系相连的事实。



