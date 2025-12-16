import re
import os
import sys

def convert_emotion_tag(file5_num):
    """根据file5:后面的数字转换情绪标签"""
    emotion_map = {
        0: "<E:smile>",
        1: "<E:serious>",
        2: "<E:thinking>",
        3: "<M:eyes_closed>|<E:serious>",
        4: "<E:worried>",
        5: "<E:embarrassed>",
        6: "<M:eyes_closed>|<E:smile>",
        7: "<E:smile>",
        8: "<M:eyes_closed>|<E:smirk>",
        9: "<E:serious>",
        10: "<E:angry>",
        11: "<E:sad>",
        12: "<E:sad>",
        13: "<E:serious>",
        14: "<E:resigned>",
        15: "<E:shocked>",
        16: "<E:shocked>",
        17: "<E:surprised>",
        18: "<E:smile>|<M:blush>",
        19: "<E:serious>|<M:blush>",
        20: "<E:thinking>|<M:blush>",
        # 21-35 为Disabled，不处理
    }
    
    # 检查是否为两位数，取个位数
    if isinstance(file5_num, str):
        # 如果是两位数，取最后一个数字
        if len(file5_num) == 2:
            num = int(file5_num[-1])
        else:
            num = int(file5_num)
    else:
        num = file5_num
    
    # 检查是否在0-20范围内且不是Disabled
    if 0 <= num <= 20:
        return emotion_map.get(num)
    return None

def process_text(input_text):
    """处理文本转换"""
    lines = input_text.split('\n')
    output_lines = []
    i = 0
    total_lines = len(lines)
    
    while i < total_lines:
        line = lines[i].strip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # 处理[场景切换]
        if line == "[场景切换]":
            output_lines.append("[场景切换]")
            i += 1
            continue
        
        # 处理角色行（以【开头】结尾）
        if line.startswith("【") and line.endswith("】"):
            # 检查是否是小倉朝日
            is_asahi = "小倉朝日" in line
            
            # 保存角色行
            output_lines.append(line)
            i += 1
            
            # 检查是否有下一行（台词行）
            if i < total_lines:
                next_line = lines[i].strip()
                # 如果是台词行（以「开头」结尾）
                if next_line.startswith("「") and next_line.endswith("」"):
                    output_lines.append(next_line)
                    i += 1
                    
                    # 检查是否是小倉朝日且有情绪行
                    if is_asahi and i < total_lines:
                        # 查找接下来的非空行（最多找两行）
                        next_non_empty_line = None
                        search_idx = i
                        
                        # 跳过空行，最多查看2行
                        for _ in range(2):
                            while search_idx < total_lines and not lines[search_idx].strip():
                                search_idx += 1
                            if search_idx < total_lines:
                                # 找到非空行
                                next_non_empty_line = lines[search_idx].strip()
                                # 检查是否包含file5:数字
                                match = re.search(r'file5:(\d{1,2})', next_non_empty_line)
                                if match:
                                    file5_num = match.group(1)
                                    emotion_tag = convert_emotion_tag(file5_num)
                                    if emotion_tag:
                                        output_lines.append(emotion_tag)
                                    # 跳过这个情绪标记行
                                    i = search_idx + 1
                                    break
                                else:
                                    # 这一行不是情绪标记行，继续检查下一行
                                    search_idx += 1
                            else:
                                break
                    continue
            continue
        
        # 如果行以%开头或以^chara开头，跳过这一行
        if line.startswith("%") or line.startswith("^chara"):
            i += 1
            continue
        
        # 如果不是以上情况，保留原行
        output_lines.append(line)
        i += 1
    
    return '\n'.join(output_lines)

def process_script_file(input_file, output_file, encoding='cp932'):
    """处理单个脚本文件"""
    try:
        # 读取输入文件（使用日本编码）
        with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
            input_text = f.read()
        
        # 处理文本
        output_text = process_text(input_text)
        
        # 追加到输出文件（使用相同的编码）
        with open(output_file, 'a', encoding=encoding) as f:
            # 添加文件名作为分隔
            f.write(f"=== 文件: {os.path.basename(input_file)} ===\n")
            f.write(output_text)
            f.write("\n\n")
        
        print(f"  已处理: {input_file}")
        return True
        
    except Exception as e:
        print(f"  处理文件 {input_file} 时发生错误: {e}")
        return False

def process_directory(directory_path, output_file, encoding='cp932'):
    """处理目录中的所有文本文件"""
    supported_extensions = ['.txt', '.s', '.script', '']
    processed_count = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            # 检查文件扩展名
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                if process_script_file(file_path, output_file, encoding):
                    processed_count += 1
            else:
                print(f"  跳过不支持的文件: {filename}")
    
    return processed_count

def try_different_encodings(file_path):
    """尝试不同的编码读取文件"""
    encodings_to_try = ['cp932', 'utf-8', 'euc-jp', 'shift_jis', 'iso-2022-jp']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return encoding, content
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    # 如果所有编码都失败，返回默认编码
    return 'cp932', ""

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
    
    # 选择编码
    print("\n请选择文件编码:")
    print("1. cp932 (日本Windows编码, 默认)")
    print("2. utf-8")
    print("3. euc-jp (日本Unix/Linux编码)")
    print("4. 自动检测编码")
    
    encoding_choice = input("请输入选择 (1-4, 默认为1): ").strip()
    if encoding_choice == '2':
        encoding = 'utf-8'
    elif encoding_choice == '3':
        encoding = 'euc-jp'
    elif encoding_choice == '4':
        encoding = 'auto'
    else:
        encoding = 'cp932'
    
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
    
    total_processed = 0
    
    # 处理多个输入文件或目录
    while True:
        print("\n" + "-" * 50)
        input_path = input("请输入要处理的脚本文件路径或目录路径（输入q退出）: ").strip()
        
        if input_path.lower() == 'q':
            break
        
        if not os.path.exists(input_path):
            print(f"路径不存在: {input_path}")
            continue
        
        if os.path.isfile(input_path):
            # 处理单个文件
            print(f"处理文件: {input_path}")
            
            if encoding == 'auto':
                # 自动检测编码
                actual_encoding, _ = try_different_encodings(input_path)
                print(f"  检测到编码: {actual_encoding}")
                if process_script_file(input_path, output_file, actual_encoding):
                    total_processed += 1
            else:
                if process_script_file(input_path, output_file, encoding):
                    total_processed += 1
                    
        elif os.path.isdir(input_path):
            # 处理目录中的所有文本文件
            print(f"处理目录: {input_path}")
            if encoding == 'auto':
                print("  自动检测编码模式，将尝试为每个文件检测编码")
            
            count = process_directory(input_path, output_file, encoding)
            total_processed += count
            print(f"  完成目录处理，共处理 {count} 个文件")
        else:
            print(f"无效的路径: {input_path}")
    
    print(f"\n处理完成！共处理 {total_processed} 个文件")
    print(f"结果已保存到 {os.path.abspath(output_file)}")
    
    # 询问是否要查看输出文件
    view_choice = input("是否要查看输出文件内容？(y/n, 默认为n): ").strip().lower()
    if view_choice == 'y':
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\n" + "=" * 50)
                print("输出文件内容:")
                print("=" * 50)
                print(content[:2000])  # 只显示前2000字符
                if len(content) > 2000:
                    print(f"... (仅显示前2000字符，文件总长度: {len(content)} 字符)")
        except Exception as e:
            print(f"读取输出文件时出错: {e}")

if __name__ == "__main__":
    main()