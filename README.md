# PDF 文档识别系统 (LVM)

本系统使用 Qwen-VL 模型进行 PDF 文档识别，通过将 PDF 转换为图片并逐页处理来提取文本内容。同时提供文档摘要生成功能，并支持向量化存储和语义检索。

## 主要功能

1. PDF 文档识别：将 PDF 转换为文本，保留表格等格式
2. 文档摘要生成：为识别后的文档生成摘要，包括背景摘要和详细摘要
3. 向量化存储：将文档内容转换为向量并存储到向量数据库，支持语义检索

## 环境要求

- Python 3.9+ (推荐 3.9.6)
- 操作系统：支持 macOS、Linux 和 Windows
- 必需的包：
  - pdf2image
  - openai
  - python-dotenv
  - chromadb
  - torch
  - transformers
  - modelscope
  - tqdm

## 安装设置

1. 安装必需的包：
```bash
pip install pdf2image openai python-dotenv chromadb torch transformers modelscope tqdm
```

2. 在项目根目录创建 `.env` 文件，添加配置：
```
# API密钥
DASH_SCOPE_API_KEY=你的API密钥

# 检索参数配置
INITIAL_TOP_K=5  # 初始检索的候选数量
FINAL_TOP_K=3    # 最终返回的结果数量
```

## 使用方法

```bash
source venv/bin/activate
```

### PDF 文档识别

使用以下命令运行文档识别：
```bash
python lvm_read_pdf.py --pdf PDF文件路径 --output_dir output/folder_name/
```

示例：
```bash
python lvm_read_pdf.py --pdf data/purchase.pdf --output_dir output/purchase/
```

### 摘要生成

在完成文档识别后，使用以下命令生成摘要：
```bash
python generate_abstract.py --output_dir output/folder_name/
```

示例：
```bash
python generate_abstract.py --output_dir output/purchase/
```

摘要生成说明：
1. 系统会首先生成背景摘要（保存为 `0-abstract.txt`）
2. 然后依次为每个文件生成详细摘要（保存为 `1-abstract.txt`、`2-abstract.txt` 等）
3. 摘要生成会自动跳过已存在的摘要文件
4. 每个摘要约 500-1000 字，突出重要规定和具体要求

### 向量化存储和检索

在完成文档识别和摘要生成后，可以将文档向量化并存储到向量数据库：

```bash
python embedding_to_db.py --model BAAI/bge-m3 \
                         --db-path ./chroma_db \
                         --collection your_collection_name \
                         --input-dir output/folder_name/ \
                         --overwrite
```

测试语义检索：
```bash
python embedding_to_db.py --model BAAI/bge-m3 \
                         --db-path ./chroma_db \
                         --collection your_collection_name \
                         --test-query "你的查询语句" \
                         --top-k 3  # 可选，默认使用 FINAL_TOP_K
```

向量化说明：
1. 使用 BGE-M3 模型进行文本向量化（支持其他模型）
2. 自动进行文本分块和上下文重叠处理
3. 支持断点续传，可以随时中断和继续处理
4. 支持 GPU 加速（如果可用）
5. 检索结果数量可通过 `.env` 文件的 `INITIAL_TOP_K` 和 `FINAL_TOP_K` 参数调整

## 重要说明

### PDF 识别部分
1. 系统会逐页处理 PDF，并保存中间结果以防数据丢失
2. 实现了断点续传系统，如果处理中断可以从上次成功的页面继续处理
3. 处理结果将保存为输出目录中的 `0.txt` 文件
4. 处理时间取决于 PDF 的大小和页数
5. 系统在处理每页之间有 1 秒的延迟，以遵守 API 调用限制

### 摘要生成部分
1. 需要先完成 PDF 识别后才能生成摘要
2. 使用 Qwen-long 模型生成摘要
3. 支持断点续传，可以随时中断和继续
4. 生成的摘要包含背景摘要和具体文件摘要两部分

### 向量化存储部分
1. 需要先完成文档识别和摘要生成
2. 自动进行文本分块，每块约 2000 字
3. 为每个文本块添加上下文信息（前文、后文和摘要）
4. 支持多种向量化模型，默认使用 BGE-M3
5. 使用 ChromaDB 作为向量数据库，支持持久化存储
6. 支持相似度检索和语义搜索
7. 检索结果数量分两级控制：
   - `INITIAL_TOP_K`：初始检索的候选结果数量（默认5）
   - `FINAL_TOP_K`：最终返回给用户的结果数量（默认3）

## 输出格式

### PDF 识别输出
- 处理后的文本将保存在指定的输出目录中
- 每页内容都会标记为 `=== 第X页 ===`
- PDF 中的表格将以文本格式保留

### 摘要输出
- `0-abstract.txt`：背景摘要文件
- `N-abstract.txt`：各个文件的详细摘要（N 为文件编号）

### 向量数据库输出
- 向量数据存储在指定的 ChromaDB 目录中
- 每个文档块包含完整的上下文信息
- 支持按相似度排序的语义检索

## 常见问题解决

- 如果处理大型 PDF 时遇到内存问题，可以尝试降低 DPI（当前设置为 300）
- 确保具有读取输入文件和写入输出目录的权限
- 处理过程中如有错误，请查看控制台输出的错误信息
- 如果摘要生成失败，检查是否已完成 PDF 识别，以及输出目录中是否存在对应的文本文件
- 向量化处理时如果内存不足，可以减小文本块的大小
- 如果 GPU 内存不足，系统会自动切换到 CPU 处理

## 完整依赖列表

你可以使用以下命令创建虚拟环境并安装所有依赖：

```bash
# 确保你的 Python 版本 >= 3.9
python --version

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

requirements.txt 内容：
```
# Python >= 3.9
```
