import os
import logging
import argparse
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AbstractGenerator:
    def __init__(self, api_key: str, output_dir):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.output_dir = output_dir
        
    def _read_file(self, filepath: str) -> Optional[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"读取文件 {filepath} 失败: {str(e)}")
            return None

    def _call_llm(self, prompt: str, error_prefix: str = "生成摘要") -> Optional[str]:
        """调用 LLM API 的通用方法"""
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
            logging.error(f"{error_prefix}时出错: {str(e)}")
            return None
            
    def generate_abstract(self, background: str, content: str, number: int) -> Optional[str]:
        base_prompt = """请为目标文本生成一个500-1000字的摘要。摘要应该：
1. 概括目标文本的主要内容和关键信息
2. 突出重要的规定、数字和具体要求

"""
        if background:
            base_prompt += f"""背景材料：
==========
{background}
==========

"""

        prompt = base_prompt + f"""需要总结的文本：
==========
{content}
==========
"""
        return self._call_llm(prompt, f"为文件 {number} 生成摘要")
            
    def generate_background_abstract(self, content: str) -> Optional[str]:
        prompt = """请根据以下文本，生成一个500字左右的背景摘要。这个摘要将用作后续文件解读的参考背景。
请着重总结：
1. 文件的基本背景和目的
2. 主要政策方向和要求
3. 关键的时间节点和目标

需要总结的文本：
==========
{content}
==========
"""
        return self._call_llm(prompt.format(content=content), "生成背景摘要")

    def process_file(self, number: int) -> bool:
        """处理单个文件的摘要生成
        
        Args:
            number: 文件编号
            
        Returns:
            bool: 是否成功生成摘要
        """
        # 读取文件
        background = self._read_file(os.path.join(self.output_dir, "0-abstract.txt"))
        if not background:
            return False
            
        content = self._read_file(os.path.join(self.output_dir, f"{number}.txt"))
        if not content:
            return False
            
        # 生成摘要
        logging.info(f"正在为文件 {number} 生成摘要...")
        abstract = self.generate_abstract(background, content, number)
        if not abstract:
            return False
            
        # 保存摘要
        try:
            output_path = os.path.join(self.output_dir, f"{number}-abstract.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(abstract)
            logging.info(f"文件 {number} 的摘要已保存")
            return True
        except Exception as e:
            logging.error(f"保存摘要到文件 {number}-abstract.txt 失败: {str(e)}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description='生成文档摘要')
    parser.add_argument('--output_dir', 
                       type=str, 
                       default="output/purchase/",
                       help='输出目录路径 (默认: output/purchase/)')
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    # 从环境变量获取API密钥
    api_key = os.getenv("DASH_SCOPE_API_KEY")
    if not api_key:
        logging.error("未设置 DASH_SCOPE_API_KEY 环境变量")
        return
        
    output_dir = args.output_dir  # 使用命令行参数
    generator = AbstractGenerator(api_key, output_dir)
    
    # 首先检查并生成背景文件
    background_path = os.path.join(output_dir, "0-abstract.txt")
    if not os.path.exists(background_path):
        logging.info("开始生成背景摘要...")
        background_content = generator._read_file(os.path.join(output_dir, "0.txt"))
        if background_content:
            background_abstract = generator.generate_background_abstract(background_content)
            if background_abstract:
                try:
                    with open(background_path, 'w', encoding='utf-8') as f:
                        f.write(background_abstract)
                    logging.info("背景摘要已生成并保存")
                except Exception as e:
                    logging.error(f"保存背景摘要失败: {str(e)}")
                    return
            else:
                logging.error("生成背景摘要失败")
                return
        else:
            logging.error("读取背景文件失败")
            return
    
    # 获取目录下的所有txt文件
    files = [f for f in os.listdir(output_dir) if f.endswith(".txt") and not f.endswith("-abstract.txt")]
    file_numbers = sorted([int(f.split('.')[0]) for f in files])
    
    for number in file_numbers:
        print(f"处理文件 {number}")
        if number == 0:  # 跳过背景文件
            continue
        
        # 检查是否已经存在摘要文件
        if os.path.exists(os.path.join(output_dir, f"{number}-abstract.txt")):
            logging.info(f"文件 {number} 的摘要已存在，跳过")
            continue
            
        if generator.process_file(number):
            logging.info(f"文件 {number} 处理完成")
        else:
            logging.error(f"文件 {number} 处理失败")

if __name__ == "__main__":
    main()
