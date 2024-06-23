import argparse
from os.path import join
from tqdm import tqdm
from transformers import BertModel
from transformers import BertConfig as BC

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper
from data_process import InputFeatures,Example
try:
    from apex import amp
except Exception:
    print('Apex not imoport!')


import torch
from torch import nn

# 设置随机种子的函数，确保结果可复现
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



# 将数据分发到GPU的函数
def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch

# 计算损失的函数
def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])

    sent_num_in_batch = batch["start_mapping"].sum()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3



import json

# 预测函数，用于在模型评估阶段生成预测结果
@torch.no_grad()  # 禁用梯度计算，加速推理过程并节省内存
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    model.eval()  # 将模型设置为评估模式
    answer_dict = {}  # 用于存储预测的答案
    sp_dict = {}  # 用于存储预测的支持句子
    dataloader.refresh()  # 刷新数据加载器
    total_test_loss = [0] * 5  # 用于存储总的测试损失

    for batch in tqdm(dataloader):  # 迭代数据加载器中的每个批次数据
        batch['context_mask'] = batch['context_mask'].float()  # 将上下文掩码转换为浮点型
        # 通过模型获得预测结果
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
        # 计算损失
        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
        
        # 累加每个损失值
        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()

        # 将预测的开始和结束位置转换为答案
        answer_dict_ = convert_to_tokens(
            example_dict, feature_dict, batch['ids'], 
            start_position.data.cpu().numpy().tolist(),
            end_position.data.cpu().numpy().tolist(), 
            np.argmax(type_logits.data.cpu().numpy(), 1)
        )
        answer_dict.update(answer_dict_)  # 更新答案字典

        # 计算支持句子的预测概率
        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):  # 遍历每个样本
            cur_sp_pred = []  # 当前样本的支持句子预测结果
            cur_id = batch['ids'][i]  # 当前样本的ID

            cur_sp_logit_pred = []  # 存储支持句子的logit值（如果需要）
            for j in range(predict_support_np.shape[1]):  # 遍历每个句子的预测概率
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:  # 如果预测概率超过阈值，则认为是支持句子
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})  # 更新支持句子字典

    # 去除答案中的空格
    new_answer_dict = {key: value.replace(" ", "") for key, value in answer_dict.items()}
    # 将答案和支持句子写入预测文件
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)

    # 打印每个损失值的平均损失
    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    # 记录测试损失
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))

# 训练一个epoch的函数
def train_epoch(data_loader, model, predict_during_train=False):
    model.train()  # 将模型设置为训练模式
    pbar = tqdm(total=len(data_loader))  # 进度条，显示训练进度
    epoch_len = len(data_loader)  # 计算一个epoch的长度
    step_count = 0  # 记录当前的训练步数
    predict_step = epoch_len // 5  # 预测步数间隔，默认为一个epoch的1/5

    while not data_loader.empty():  # 当数据加载器不为空时进行循环
        step_count += 1  # 增加训练步数计数器
        batch = next(iter(data_loader))  # 获取一个批次的数据
        batch['context_mask'] = batch['context_mask'].float()  # 将上下文掩码转换为浮点型
        train_batch(model, batch)  # 训练一个批次的数据
        del batch  # 删除批次数据以释放内存

        # 如果启用训练期间预测，并且当前步数是预测步数的倍数
        if predict_during_train and (step_count % predict_step == 0):
            # 进行预测，并保存预测结果到指定文件
            predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                    join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
            # 保存当前模型的状态
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pth".format(args.seed, epc, step_count)))
            model.train()  # 重新设置模型为训练模式
        pbar.update(1)  # 更新进度条

    # 训练结束后，进行最后一次预测并保存结果
    predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
            join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pth".format(args.seed, epc)))

# 训练一个batch的函数
def train_batch(model, batch):
    global global_step, total_train_loss

    start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
    loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
    loss_list = list(loss_list)
    if args.gradient_accumulation_steps > 1:
        loss_list[0] = loss_list[0] / args.gradient_accumulation_steps
    
    if args.fp16:
        with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(total_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        total_train_loss = [0] * 5


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    args = set_config()  # 设置配置参数

    # 设置GPU数量
    args.n_gpu = torch.cuda.device_count()

    # 设置随机种子
    if args.seed == 0:
        args.seed = random.randint(0, 100)  # 如果种子为0，则随机生成一个种子
    set_seed(args)  # 设置随机种子

    # 初始化数据帮助类
    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 设置类型数量，默认为2

    # 设置数据加载器
    Full_Loader = helper.train_loader  # 全量训练数据加载器
    # Subset_Loader = helper.train_sub_loader  # 子集训练数据加载器（注释掉）
    dev_example_dict = helper.dev_example_dict  # 验证集示例字典
    dev_feature_dict = helper.dev_feature_dict  # 验证集特征字典
    eval_dataset = helper.dev_loader  # 验证集数据加载器

    # 初始化BERT模型和配置
    roberta_config = BC.from_pretrained(args.bert_model)
    encoder = BertModel.from_pretrained(args.bert_model)
    args.input_dim = roberta_config.hidden_size  # 设置输入维度
    model = BertSupportNet(config=args, encoder=encoder)  # 初始化模型
    if args.trained_weight is not None:
        model.load_state_dict(torch.load(args.trained_weight))  # 加载预训练权重
    model.to('cuda')  # 将模型加载到GPU

    # 初始化优化器和损失函数
    lr = args.lr  # 学习率
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps  # 总训练步骤数
    warmup_steps = 0.1 * t_total  # 预热步骤数
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)  # AdamW优化器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)  # 学习率调度器
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  # 交叉熵损失
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # 二分类损失
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  # 支持损失函数

    # 如果使用16位浮点数训练
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, "einsum")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # 使用数据并行
    model = torch.nn.DataParallel(model)
    model.train()  # 设置模型为训练模式

    # 开始训练
    global_step = epc = 0  # 初始化全局步数和epoch计数器
    total_train_loss = [0] * 5  # 初始化总训练损失
    test_loss_record = []  # 初始化测试损失记录
    VERBOSE_STEP = args.verbose_step  # 设置每个verbose step

    # 训练循环
    while True:
        if epc == args.epochs:  # 如果达到指定的epoch数，退出
            exit(0)
        epc += 1  # 增加epoch计数

        Loader = Full_Loader  # 设置数据加载器为全量加载器
        Loader.refresh()  # 刷新加载器

        if epc > 2:
            train_epoch(Loader, model, predict_during_train=True)  # 如果epoch大于2，在训练过程中进行预测
        else:
            train_epoch(Loader, model)  # 否则，只进行训练

