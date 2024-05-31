import warnings
warnings.filterwarnings('ignore')

# 初始化
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True).half().cuda()
model = model.eval()

history = []


# 读取当前脚本文件所在文件夹下的 input.json
with open("ChatGLM3-6B-32k/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 进行对话
print("User: " + text + "\n")
response, history = model.chat(tokenizer, text, history=history, temperature=0.2, top_p=1)
print("Assistant: " + response + "\n")


# 将回复写入当前脚本文件所在文件夹下的 output.txt
with open("ChatGLM3-6B-32k/output.txt", "w", encoding="utf-8") as f:
    f.write(response)
with open("ChatGLM3-6B-32k/history.txt", "w", encoding="utf-8") as f:
    for item in history:
        f.write(str(item) + "\n")
