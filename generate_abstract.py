import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AbstractGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
    def _read_file(self, filepath: str) -> Optional[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"读取文件 {filepath} 失败: {str(e)}")
            return None
            
    def generate_abstract(self, background: str, content: str, number: int) -> Optional[str]:
        prompt = f"""请参考以下背景材料，为目标文本生成一个500-1000字的摘要。摘要应该：
1. 概括目标文本的主要内容和关键信息
2. 突出重要的规定、数字和具体要求

背景材料：
==========
{background}
==========

需要总结的文本：
==========
{content}
==========
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-long",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"为文件 {number} 生成摘要时出错: {str(e)}")
            return None
            
    def process_file(self, number: int) -> bool:
        """处理单个文件的摘要生成
        
        Args:
            number: 文件编号
            
        Returns:
            bool: 是否成功生成摘要
        """
        # 读取文件
        background = self._read_file("output/0-abstract.txt")
        if not background:
            return False
            
        content = self._read_file(f"output/{number}.txt")
        if not content:
            return False
            
        # 生成摘要
        logging.info(f"正在为文件 {number} 生成摘要...")
        abstract = self.generate_abstract(background, content, number)
        if not abstract:
            return False
            
        # 保存摘要
        try:
            with open(f"output/{number}-abstract.txt", 'w', encoding='utf-8') as f:
                f.write(abstract)
            logging.info(f"文件 {number} 的摘要已保存")
            return True
        except Exception as e:
            logging.error(f"保存摘要到文件 {number}-abstract.txt 失败: {str(e)}")
            return False

def main():
    load_dotenv()
    # 从环境变量获取API密钥
    api_key = os.getenv("DASH_SCOPE_API_KEY")
    if not api_key:
        logging.error("未设置 DASH_SCOPE_API_KEY 环境变量")
        return
        
    generator = AbstractGenerator(api_key)
    
    # 获取data目录下的所有txt文件
    files = [f for f in os.listdir("output") if f.endswith(".txt") and not f.endswith("-abstract.txt")]
    file_numbers = sorted([int(f.split('.')[0]) for f in files])
    
    for number in file_numbers:
        print(f"处理文件 {number}")
        if number == 0:  # 跳过背景文件
            continue
        
        # 检查是否已经存在摘要文件
        if os.path.exists(f"output/{number}-abstract.txt"):
            logging.info(f"文件 {number} 的摘要已存在，跳过")
            continue
            
        if generator.process_file(number):
            logging.info(f"文件 {number} 处理完成")
        else:
            logging.error(f"文件 {number} 处理失败")

if __name__ == "__main__":
    main()
