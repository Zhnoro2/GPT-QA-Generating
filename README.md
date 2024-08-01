# 结合专业材料与GPT生成问答对

>在基于已有大语言模型，训练专业领域大模型的过程中，除了通过增量预训练（Continual pre-training）为大模型增加特定领域的知识，以提升专业能力外。通常还需要大模型能够正确理解用户的问题或指令，并给出符合要求的回答格式。这一过程不仅仅依赖于模型对领域知识的掌握，还需要确保模型在理解和生成自然语言方面表现得足够准确和灵活。为了实现这一目标，训练者需要为大模型输入整理好的问答对。通过使用这些问答对材料，模型可以学习如何从用户的问题中提取关键信息，理解问题的意图，并生成相应且符合格式要求的回答。这一过程称为指令微调（Instruction Tuning）。

>人工整理的问答对可能来自于通讯软件中的交流信息，比如邮件、会议记录、聊天信息等，虽然包含专业知识，但也往往存在大量噪音，数据质量难以把握，以及上下文依赖性强、数据分布不平衡等问题。尽管人工整理这些信息并将其转化为需要的问答对是可行的，但效率极低且成本较高。考虑到目前GPT-4的对话能力已非常强大，可以利用其生成的对话作为训练专业大模型的材料。通过调用API，我们可以高效地将专业材料交给GPT-4，生成大量高质量的问答数据，从而为模型的训练提供支持。

## 数据准备
本代码之前用于合同领域的问答对生成，使用的数据包含了两个要素，一个是合同要点，即该部分内容与合同的哪个方面相关，或属于哪个部分；另一个是具体的审查规则，用于明确某个审查点的定义、如何判断是否存在风险等。以下以合同中常见的支付方式为例，后续会展示GPT基于该要点生成的问答对。

![image](https://github.com/user-attachments/assets/4fec875e-5616-49ad-9571-90a75bc9cc9d)

## 代码说明

`GPT-QA-Generating` 使用方法:

准备工作
>导入文件并获取API密钥。出于安全考虑，密钥已提前存在环境变量中，以下代码直接获取。

```python
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

#获取已保存在环境变量中的API密钥，或换成自己的密钥
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
```

定义工作角色和背景
>主要包含两个方面：（1）定义大模型的角色和项目的背景，以更好发挥大模型的能力；（2）要求大模型按照按照不同的“难度”来生成问题，以此确保问题的多样化，更全面地覆盖现实世界中从业者可能给出的问题。在本项目中，问题的难度、深度被细化为“不同年限经验的从业者”可能提出的问题。

```python
system_message = {"role": "system", "content": "你是一名具有丰富国际工程合同管理经验、AI相关经验的LLM模型训练师，我需要大量的问答数据对模型进行训练，因此需要你帮助生成。请站在承包商一方从业人员的角度，以符合逻辑的方式进行提问和回答，你可以生成多条，从而覆盖这个知识点，并展现出不同的形式。要求领域为国际工程商务合同管理，从“小于5年的从业者”，“5-10年的从业者”，“大于10年的从业者”，根据这些用户可能出现的疑问生成不同难度的问答，问答前先展示是三类从业者中哪一类，然后给出问题：Q计数:问题内容，如Q1....Q2；答案：A计数:答案内容,"}
```

定义函数
>假设模型已经生成回答文本，对文本提取难度分类、问题、答案，导入到数据表中

```python
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
```

问答生成
>调用API生成问答对，利用函数提取数据。调用前确保已经正确设置代理

```python
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
```

将存储在DataFrame的数据存储在excel表格中
```python
# 设置存储路径
file_path = r'C:\Users\Desktop\QA_Data.xlsx'

# 检查同名文件是否存在
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
```
    
