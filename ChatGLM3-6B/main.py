import warnings
warnings.filterwarnings('ignore')

# 初始化
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()

history = [
    {
        'role': 'user', 
        'content': '你知道什么是JSON格式吗？虽然JSON格式不能添加注释，但是你能理解含有"//"注释的JSON格式文件吗'
    }, 
    {
        'role': 'assistant', 
        'metadata': '', 
        'content': 'JSON是一种轻量级的数据交换格式。它基于JavaScript编程语言的一个子集，但是由于其文本格式清晰且易于解析，因此它被许多编程语言广泛支持。JSON用于数据的存储和传输。在JSON中，数据通常以键值对的形式出现，用冒号":"分隔键和值。对象由花括号"{"和"}"括起来，数组由方括号"["和"]"括起来。虽然JSON格式本身不支持注释，但是许多开发者在编写JSON文件时可能会无意中包含"//"这样的注释符号。'
    },
    {
        'role': 'user',
        'content': '你知道什么是Python列表吗？'
    },
    {
        'role': 'assistant',
        'metadata': '',
        'content': '当然，Python列表是Python编程语言中的一种内置数据类型，它是一种可变的序列，可以包含任意类型的元素，包括数字、字符串、其他列表，甚至是函数和对象。列表用方括号"[]"来表示，元素之间用逗号","分隔。'
    }
]

# 打印所有对话历史
for item in history:
    if item['role'] == 'user':
        print("User: " + item['content'] + "\n")
    else:
        print("Assistant: " + item['content'] + "\n")

# 读取当前脚本文件所在文件夹下的 input.json
with open("ChatGLM3-6B/input.json", "r", encoding="utf-8") as f:
    text = "这是一个JSON格式的文件: \n" + f.read()

# 进行对话
print("User: " + text + "\n")
response, history = model.chat(tokenizer, text, history=history, temperature=0.2, top_p=1)
print("Assistant: " + response + "\n")

# 读取当前脚本文件所在文件夹下的 input.list
with open("ChatGLM3-6B/input.list", "r", encoding="utf-8") as f:
    text = "这是一个list文件: \n" + f.read()

# 进行对话
print("User: " + text + "\n")
response, history = model.chat(tokenizer, text, history=history, temperature=0.2, top_p=1)
print("Assistant: " + response + "\n")

# 分析请求
text = "请你阅读列表文件里面的数据，并依据列表尝试更新JSON的结果，并且你只需要告诉我修改过了的JSON部分。注意，你不可以添加键值对，也不可以修改键，只可以修改值。"

# 进行对话
print("User: " + text + "\n")
response, history = model.chat(tokenizer, text, history=history, temperature=0.8, top_p=0.9)
print("Assistant: " + response + "\n")

# 将回复写入当前脚本文件所在文件夹下的 output.txt
with open("ChatGLM3-6B/output.txt", "w", encoding="utf-8") as f:
    f.write(response)
with open("ChatGLM3-6B/history.txt", "w", encoding="utf-8") as f:
    for item in history:
        f.write(str(item) + "\n")
