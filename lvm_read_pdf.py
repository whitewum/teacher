from pdf2image import convert_from_path
from openai import OpenAI
import os
import re
import json
import base64
import time
from dotenv import load_dotenv

def process_image_with_qwen(image_path, client):
    """
    使用qwen-vl处理单张图片
    """
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    },
                    {
                        "type": "text",  
                        "text": "请识别并输出这张图片中的所有文字内容，按照阅读顺序输出。对于图片中的表格，以表格的形式输出。不要加任何解释。"
                    }
                ]
            }]
        )
        ret = completion.choices[0].message.content
        print(f"处理图片成功: {ret}")
        return ret
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return None

def encode_image(image_path):
    """Base64编码图片"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_pdf_with_layout(pdf_path, checkpoint_file=".checkpoint.json"):
    """
    使用qwen-vl处理PDF文件，支持断点续传
    """
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("DASH_SCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 加载断点信息
    processed_pages = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            processed_pages = json.load(f)
    
    try:
        print("正在转换PDF为图片...")
        images = convert_from_path(
            pdf_path,
            dpi=300,
            fmt='jpg'
        )
        
        full_text = ""
        
        for i, image in enumerate(images):

                
            # 检查是否已处理过该页
            if str(i) in processed_pages:
                print(f"页面 {i+1} 已处理，使用缓存结果...")
                full_text += processed_pages[str(i)]
                continue
                
            print(f"正在处理页面 {i+1}/{len(images)}...")
            
            temp_path = f"temp_page_{i}.jpg"
            image.save(temp_path, 'JPEG', quality=80)
            
            # 使用qwen-vl处理图片
            content = process_image_with_qwen(temp_path, client)
            
            if content:
                page_text = f"\n=== 第{i+1}页  ===\n{content}\n"
                full_text += page_text
                
                # 保存断点信息
                processed_pages[str(i)] = page_text
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_pages, f, ensure_ascii=False, indent=2)
                
                # 每处理完一页就保存一次结果
                save_partial_result(full_text, f"output/partial_result_{i+1}.txt")
            
            os.remove(temp_path)
            # 添加延时避免API限制
            time.sleep(1)
        
        # 处理完成后删除断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("处理完成，已删除断点文件")
            
        return full_text
    
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return ""

def save_partial_result(text, filename):
    """
    保存部分处理结果
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 修正参数名 exist_okay -> exist_ok
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF文档处理工具')
    parser.add_argument('--pdf', required=True, help='输入PDF文件路径，例如：data/purchase.pdf')
    parser.add_argument('--output_dir', required=True, help='输出目录路径，例如: output/purchase/')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(args.pdf):
            raise FileNotFoundError(f"输入的PDF文件不存在: {args.pdf}")
            
        print("开始处理PDF...")
        text = process_pdf_with_layout(args.pdf)
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 构建输出文件路径
        output_file = os.path.join(args.output_dir, "0.txt")
        
        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"转换完成！结果已保存到 {output_file}")
        print("请检查后根据需要将文件重命名为0.txt(总则)或1.txt（细节）")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()