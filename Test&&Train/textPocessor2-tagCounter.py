import re
from collections import defaultdict

def extract_luna_dialogues_with_tags(file2_path):
    """
    从文件2中提取桜小路ルナ的带标签对话
    只提取有标签的对话，没有标签的跳过
    """
    dialogues_with_tags = []
    
    with open(file2_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是桜小路ルナ的对话
        if line == "【桜小路ルナ】":
            # 下一行应该是台词
            if i + 1 < len(lines):
                dialogue_line = lines[i + 1]
                # 验证是否是台词行（以「开头，以」结尾）
                if dialogue_line.startswith('「') and dialogue_line.endswith('」'):
                    # 检查下一行是否有标签
                    if i + 2 < len(lines):
                        next_line = lines[i + 2]
                        # 检查是否是标签行（以<开头，包含>）
                        if next_line.startswith('<') and '>' in next_line:
                            # 提取台词内容（去掉「和」）
                            dialogue_content = dialogue_line[1:-1]
                            
                            # 提取标签（可能有多个，用|分隔）
                            tags = []
                            tag_line = next_line
                            
                            # 提取所有<...>格式的标签
                            tag_pattern = r'<([^>]+)>'
                            tag_matches = re.findall(tag_pattern, tag_line)
                            
                            for tag_match in tag_matches:
                                # 处理用|分隔的标签
                                sub_tags = [t.strip() for t in tag_match.split('|') if t.strip()]
                                for tag in sub_tags:
                                    tags.append(f"<{tag}>")
                            
                            if tags:
                                dialogues_with_tags.append((dialogue_content, tags))
                            
                            i += 3  # 跳过角色行、台词行和标签行
                            continue
        
        i += 1
    
    return dialogues_with_tags

def find_file5_in_file1(file1_path, dialogue_content):
    """
    在文件1中精确查找对话内容，并返回下一行的file5数字
    如果下一行没有file5数字，返回None
    """
    with open(file1_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    # 构建完整的台词行进行搜索
    search_line = f"「{dialogue_content}」"
    
    for i in range(len(lines)):
        if lines[i] == search_line:
            # 验证前一行是否是【桜小路ルナ】
            if i > 0 and lines[i-1] == "【桜小路ルナ】":
                # 检查下一行是否有file5信息
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # 查找file5:数字
                    match = re.search(r'file5:(\d+)', next_line)
                    if match:
                        return match.group(1)
    
    return None

def analyze_luna_dialogues(file1_path, file2_path):
    """
    分析桜小路ルナ的对话并统计file5对应的标签
    """
    print("正在从文件2中提取桜小路ルナ的带标签对话...")
    dialogues_with_tags = extract_luna_dialogues_with_tags(file2_path)
    print(f"找到 {len(dialogues_with_tags)} 个带标签的对话")
    
    if not dialogues_with_tags:
        print("没有找到任何带标签的对话")
        return
    
    # 显示前几个示例
    print("\n前5个带标签的对话示例：")
    for i, (dialogue, tags) in enumerate(dialogues_with_tags[:5]):
        print(f"{i+1}. 「{dialogue[:50]}...」")
        print(f"   标签: {tags}")
    
    print("-" * 50)
    
    # 统计结果
    stats = defaultdict(lambda: defaultdict(int))
    matched_count = 0
    no_file5_count = 0
    not_found_count = 0
    
    print("开始在文件1中查找匹配...")
    
    for i, (dialogue_content, tags) in enumerate(dialogues_with_tags):
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i+1}/{len(dialogues_with_tags)}")
        
        # 在文件1中查找
        file5_num = find_file5_in_file1(file1_path, dialogue_content)
        
        if file5_num:
            matched_count += 1
            # 累加标签计数
            for tag in tags:
                stats[file5_num][tag] += 1
        elif file5_num is None:
            # 找到了对话但下一行没有file5
            no_file5_count += 1
        else:
            # 没找到对话
            not_found_count += 1
    
    print(f"\n匹配完成！")
    print(f"成功匹配并找到file5: {matched_count} 个对话")
    print(f"找到对话但下一行无file5: {no_file5_count} 个对话")
    print(f"未找到对话: {not_found_count} 个对话")
    
    if not stats:
        print("\n警告：没有找到任何匹配的file5数字！")
        return
    
    print("\n" + "="*50)
    print("统计结果：")
    
    # 按file5数字排序输出
    for file_num in sorted(stats.keys(), key=lambda x: int(x)):
        print(f"\nfile5:{file_num}对应的tag:")
        
        # 按标签计数降序排序
        sorted_tags = sorted(stats[file_num].items(), key=lambda x: x[1], reverse=True)
        
        for tag, count in sorted_tags:
            print(f"  {tag}: {count}")
    
    # 输出汇总信息
    print("\n" + "="*50)
    print("汇总信息：")
    total_tags = sum(sum(tags.values()) for tags in stats.values())
    print(f"总共统计了 {len(stats)} 个不同的file5数字")
    print(f"总共统计了 {total_tags} 个标签关联")

def debug_specific_dialogue(file1_path, file2_path, dialogue_index=0):
    """调试特定对话的匹配情况"""
    dialogues_with_tags = extract_luna_dialogues_with_tags(file2_path)
    
    if dialogue_index >= len(dialogues_with_tags):
        print(f"对话索引超出范围，最大为 {len(dialogues_with_tags)-1}")
        return
    
    dialogue_content, tags = dialogues_with_tags[dialogue_index]
    
    print(f"调试对话 #{dialogue_index}:")
    print(f"台词: 「{dialogue_content}」")
    print(f"标签: {tags}")
    
    with open(file1_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    search_line = f"「{dialogue_content}」"
    
    print(f"\n在文件1中搜索: {search_line}")
    
    found = False
    for i in range(len(lines)):
        if lines[i] == search_line:
            print(f"在第 {i+1} 行找到匹配！")
            
            # 检查前一行
            if i > 0:
                print(f"前一行: {lines[i-1]}")
                if lines[i-1] == "【桜小路ルナ】":
                    print("  前一行是【桜小路ルナ】，角色匹配！")
            
            # 检查后一行
            if i + 1 < len(lines):
                print(f"后一行: {lines[i+1]}")
                match = re.search(r'file5:(\d+)', lines[i+1])
                if match:
                    print(f"  找到file5数字: {match.group(1)}")
                else:
                    print("  下一行没有找到file5数字")
            else:
                print("  没有下一行")
            
            found = True
            break
    
    if not found:
        print("未找到匹配的对话")
        
        # 尝试显示文件1中的类似对话
        print("\n文件1中的类似桜小路ルナ对话（前5个）：")
        count = 0
        for i in range(len(lines)):
            if lines[i] == "【桜小路ルナ】" and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.startswith('「') and next_line.endswith('」'):
                    print(f"  「{next_line[1:51]}...」")
                    count += 1
                    if count >= 5:
                        break

def main():
    # 文件路径
    file1_path = input("请输入第一个文件的路径: ").strip() or "file1.txt"
    file2_path = input("请输入第二个文件的路径: ").strip() or "file2.txt"
    
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print("-" * 50)
    
    try:
        analyze_luna_dialogues(file1_path, file2_path)
        
        # 可选：调试特定对话
        debug_choice = input("\n是否要调试特定对话？(y/n): ").strip().lower()
        if debug_choice == 'y':
            dialogue_index = int(input("输入对话索引 (0为第一个): ").strip() or "0")
            debug_specific_dialogue(file1_path, file2_path, dialogue_index)
            
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()