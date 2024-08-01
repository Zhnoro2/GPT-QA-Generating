#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import re
import os
import openai
from openai import OpenAI
from tqdm import tqdm

#选择文件所在位置，读取原始数据
df=pd.read_excel(r"C:\Users\Desktop\Data.xlsx")

#打印前几行数据以确认读取正确
print(df.head())

#获取已保存在环境变量中的API密钥
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)

#定义工作角色和背景
system_message = {"role": "system", "content": "你是一名具有丰富国际工程合同管理经验、AI相关经验的LLM模型训练师，我需要大量的问答数据对模型进行训练，因此需要你帮助生成。请站在承包商一方从业人员的角度，以符合逻辑的方式进行提问和回答，你可以生成多条，从而覆盖这个知识点，并展现出不同的形式。要求领域为国际工程商务合同管理，从“小于5年的从业者”，“5-10年的从业者”，“大于10年的从业者”，根据这些用户可能出现的疑问生成不同难度的问答，问答前先展示是三类从业者中哪一类，然后给出问题：Q计数:问题内容，如Q1....Q2；答案：A计数:答案内容,"}

#定义函数，对模型生成的文本提取经验分类、问题、答案
def extract_qa_with_categories(text):
    # 正则表达式匹配经验分类和后续的问题与答案
    category_pattern = r"(小于5年的从业者|5-10年的从业者|大于10年的从业者)"
    # 通过正则表达式对问答进行分块，并且处理可能的空白字符
    qa_pattern = r"Q(\d*)\s*:\s*\**\s*(.+?)\s*\**\s*(?:\n|\s)+A\1\s*:\s*(.+?)(?=(?:\n\s*|\s*)Q\d*|\Z)"
    
    categories = re.findall(category_pattern, text)
    split_text = re.split(category_pattern, text)
    
    questions = []
    answers = []
    categories_result = []
    
    # 对每个分类进行处理
    for i in range(1, len(split_text), 2):
        # 找到每个分类后的文本块
        category = split_text[i]
        category_text = split_text[i+1]
        
        # 在当前分类文本块中匹配问题和答案
        qas = re.findall(qa_pattern, category_text, re.DOTALL)
        
        for q, question, answer in qas:
            questions.append(question.strip())
            answers.append(answer.strip())
            categories_result.append(category)
    
    # 创建 DataFrame 进行存储
    df = pd.DataFrame({
        'Category': categories_result,
        'Question': questions,
        'Answer': answers
    })
    
    return df


#初始化一个列表来存储问答对和模块类别
qa_pairs = []


# 根据原始数据表的每一行的审查规则，生成文本。提取问答，同时加上审查点和审查规则，其中tqdm提供了进度条功能
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating QA Pairs"):
    user_message = {
        "role": "user", 
        "content": f"下面是相关知识点名称: {row['审查规则']}"
    }
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.8,
        messages=[system_message, user_message]
    )
    
    generated_text = response.choices[0].message.content

    # 提取生成的问答对，并包含审查点名称
    extracted_qa = extract_qa_with_categories(generated_text)
    extracted_qa['审查点'] = row['审查点']  # 为每个问答对添加审查点名称
    extracted_qa['审查规则'] = row['审查规则']  # 为每个问答对添加审查规则

    # 将提取的数据加入到总列表中
    qa_pairs.append(extracted_qa)

# 将所有问答对合并到一个 DataFrame 中
qa_df = pd.concat(qa_pairs, ignore_index=True)


#选择输出位置并输入文档
file_path = r'C:\Users\Desktop\QA_Data.xlsx'  
# 检查文件是否存在
if os.path.exists(file_path):
    # 文件已存在，询问用户是否删除
    response = input(f"文件 '{file_path}' 已存在。是否要删除它并创建一个新文件？(y/n): ")
    if response.lower() == 'y':
        os.remove(file_path)  # 删除文件
        print("旧文件已删除。")
        qa_df.to_excel(file_path, index=False, sheet_name='QA Data')  # 写入新文件
        print("数据已成功写入到新的 Excel 文件中！")
    else:
        print("操作已取消，没有写入数据。")
else:
    # 文件不存在，直接写入
    qa_df.to_excel(file_path, index=False, sheet_name='QA Data')
    print("数据已成功写入到 Excel 文件中！")





