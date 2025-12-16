import os
import re
import sys

def process_script_file(input_file_path, output_file_path, encoding='shift_jis'):
    """
    处理单个游戏脚本文件
    
    Args:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
        encoding: 文件编码（默认为日文shift_jis）
    """
    
    try:
        with open(input_file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 如果默认编码失败，尝试其他常见日文编码
        try:
            with open(input_file_path, 'r', encoding='cp932') as f:
                lines = f.readlines()
        except:
            print(f"无法解码文件: {input_file_path}，请手动指定正确的编码")
            return
    
    output_lines = []
    current_block = []
    in_block = False
    last_was_scene_change = False
    
    for line in lines:
        line = line.rstrip()  # 移除行尾空白字符
        
        # 检查是否是新块开始
        if line.startswith('％v'):
            # 如果之前有块，先处理并保存
            if current_block:
                processed_block = process_text_block(current_block)
                if processed_block:
                    # 如果上次有场景切换，添加标记
                    if last_was_scene_change:
                        output_lines.append("[场景切换]")
                        last_was_scene_change = False
                    output_lines.extend(processed_block)
                    output_lines.append('')  # 块后加空行
            
            # 开始新块
            current_block = [line]
            in_block = True
        
        # 如果在块中，继续收集行
        elif in_block:
            if line.strip() == '':  # 空行表示块结束
                if current_block:
                    processed_block = process_text_block(current_block)
                    if processed_block:
                        # 如果上次有场景切换，添加标记
                        if last_was_scene_change:
                            output_lines.append("[场景切换]")
                            last_was_scene_change = False
                        output_lines.extend(processed_block)
                        output_lines.append('')  # 块后加空行
                current_block = []
                in_block = False
            else:
                current_block.append(line)
                
                # 检查场景切换
                if line.startswith('^bg'):
                    last_was_scene_change = True
    
    # 处理文件末尾的最后一个块
    if current_block:
        processed_block = process_text_block(current_block)
        if processed_block:
            # 如果上次有场景切换，添加标记
            if last_was_scene_change:
                output_lines.append("[场景切换]")
            output_lines.extend(processed_block)
            output_lines.append('')  # 块后加空行
    
    # 将处理后的行追加到输出文件
    if output_lines:
        with open(output_file_path, 'a', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"已处理文件: {input_file_path}，添加了 {len([l for l in output_lines if l.strip()])} 行有效内容")
    else:
        print(f"文件 {input_file_path} 中没有找到匹配的内容")

def process_text_block(block):
    """
    处理单个文本块，提取需要的行并清理chara命令
    
    Args:
        block: 文本块列表
    
    Returns:
        处理后的文本块列表
    """
    processed = []
    
    for line in block:
        # 保留％v开头的行
        if line.startswith('％v'):
            processed.append(line)
        
        # 处理对话行（包含【或「的行）
        elif '【' in line or '「' in line:
            cleaned_line = clean_dialogue_line(line)
            if cleaned_line:
                processed.append(cleaned_line)
        
        # 处理chara行
        elif line.startswith('^chara'):
            cleaned_chara = clean_chara_line(line)
            if cleaned_chara:
                processed.append(cleaned_chara)
    
    return processed

def clean_dialogue_line(line):
    """
    清理对话行：删除[n]和[rb,...]等中括号内容，并截断到」字符
    
    Args:
        line: 原始对话行
    
    Returns:
        清理后的对话行
    """
    # 删除[n]
    line = line.replace('[n]', '')
    
    # 删除所有[rb,...]等中括号内容
    line = re.sub(r'\[.*?\]', '', line)
    
    # 截断到」字符
    end_index = line.find('」')
    if end_index != -1:
        line = line[:end_index + 1]  # 包括」字符
    
    return line

def clean_chara_line(chara_line):
    """
    清理chara行，只保留file参数
    
    Args:
        chara_line: 原始chara行
    
    Returns:
        只包含file参数的chara行
    """
    # 分割参数
    parts = chara_line.split(',')
    
    # 保留命令头和所有file参数
    cleaned_parts = [parts[0]]  # 命令头，如^chara01
    
    for part in parts[1:]:
        if part.startswith('file'):
            cleaned_parts.append(part)
    
    # 重新组合
    return ','.join(cleaned_parts)

def process_directory(directory_path, output_file_path):
    """
    处理目录中的所有.s文件
    
    Args:
        directory_path: 目录路径
        output_file_path: 输出文件路径
    """
    # 遍历目录及其所有子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.s'):
                file_path = os.path.join(root, file)
                print(f"发现脚本文件: {file_path}")
                process_script_file(file_path, output_file_path)

def main():
    """
    主函数：处理多个脚本文件或目录
    """
    print("游戏脚本处理工具")
    print("=" * 50)
    
    # 获取输出文件名
    output_file = input("请输入输出文件名（默认为 output.txt）: ").strip()
    if not output_file:
        output_file = "output.txt"
    
    # 检查输出文件是否存在，如果存在则询问是否追加
    if os.path.exists(output_file):
        choice = input(f"文件 {output_file} 已存在，是否追加内容？(y/n, 默认为y): ").strip().lower()
        if choice == 'n':
            # 清空文件
            open(output_file, 'w', encoding='utf-8').close()
            print("已清空输出文件")
    else:
        # 创建空文件
        open(output_file, 'w', encoding='utf-8').close()
    
    # 处理多个输入文件或目录
    while True:
        input_path = input("请输入要处理的脚本文件路径或目录路径（输入q退出）: ").strip()
        
        if input_path.lower() == 'q':
            break
        
        if not os.path.exists(input_path):
            print(f"路径不存在: {input_path}")
            continue
        
        if os.path.isfile(input_path):
            # 处理单个文件
            process_script_file(input_path, output_file)
        elif os.path.isdir(input_path):
            # 处理目录中的所有.s文件
            process_directory(input_path, output_file)
            print(f"已完成目录 {input_path} 的处理")
        else:
            print(f"无效的路径: {input_path}")
    
    print(f"处理完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main()