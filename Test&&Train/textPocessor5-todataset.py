import re
import json
import os
from collections import deque

def parse_conversation_to_samples(text):
    """
    将对话文本转换为训练样本
    """
    samples = []
    
    # 按场景分割文本
    scenes = text.split('[场景切换]')
    
    for scene_idx, scene in enumerate(scenes):
        if not scene.strip():
            continue
            
        # 解析场景中的对话行
        lines = [line.strip() for line in scene.strip().split('\n') if line.strip()]
        dialogue_history = []  # 存储完整的对话历史
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检查是否为角色行
            role_match = re.match(r'^【(.+?)】$', line)
            
            if role_match:
                role = role_match.group(1)
                
                # 读取台词和情感标签（直到下一个角色行或文件结束）
                i += 1
                dialogue_lines = []
                emotion_tags = []
                
                while i < len(lines):
                    current_line = lines[i]
                    
                    # 如果遇到新的角色行，停止读取
                    if re.match(r'^【(.+?)】$', current_line):
                        break
                    
                    # 如果是情感标签（以<开头）
                    if current_line.startswith('<'):
                        emotion_tags.append(current_line)
                    else:
                        dialogue_lines.append(current_line)
                    
                    i += 1
                
                # 构建完整的对话块字符串
                dialogue_text = '\n'.join(dialogue_lines)
                full_dialogue_block = f"【{role}】\n{dialogue_text}"
                
                # 如果有情感标签，添加到对话块中
                if emotion_tags:
                    full_dialogue_block += '\n' + '\n'.join(emotion_tags)
                
                # 如果是小倉朝日的台词，创建样本
                if role == '小倉朝日':
                    # 构建对话历史（最多4个对话块）
                    history_blocks = []
                    # 从对话历史中取最后4个非朝日的对话块
                    # 注意：我们只取朝日说话之前的对话历史
                    for hist in dialogue_history:
                        # 检查历史对话块是否包含小倉朝日
                        if not hist.startswith('【小倉朝日】'):
                            history_blocks.append(hist)
                    
                    # 只保留最多4个对话块
                    history_blocks = history_blocks[-4:] if len(history_blocks) > 4 else history_blocks
                    
                    # 构建user content
                    history_text = '\n'.join(history_blocks)
                    user_content = f"【前後の会話】\n{history_text}\n【小倉朝日】\n\n（次の台詞を、最初の行に感情タグを出力し、その次の行に台詞を出力してください。）"
                    
                    # 构建assistant content
                    if emotion_tags:
                        assistant_content = '\n'.join(emotion_tags) + '\n' + dialogue_text
                    else:
                        assistant_content = dialogue_text
                    
                    # 创建样本
                    sample = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "あなたは【小倉朝日】として話してください。台詞は日本語で、原作の表記（「…」）を守ります。"
                            },
                            {
                                "role": "user",
                                "content": user_content
                            },
                            {
                                "role": "assistant",
                                "content": assistant_content
                            }
                        ],
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    }
                    
                    samples.append(sample)
                    
                    # 将当前朝日的对话添加到历史中（用于后续可能的样本）
                    # 但注意：如果后面还有朝日的对话，这次的对话会成为历史的一部分
                    dialogue_history.append(full_dialogue_block)
                else:
                    # 其他角色的对话，添加到历史中
                    dialogue_history.append(full_dialogue_block)
            else:
                # 如果不是角色行，跳过
                i += 1
    
    return samples

def process_file(input_file, output_file):
    """
    处理单个文件
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析为样本
    samples = parse_conversation_to_samples(content)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"转换完成！共生成 {len(samples)} 个样本")
    print(f"输出已保存到: {output_file}")
    
    # 显示前3个样本的预览
    if samples:
        print("\n前3个样本预览:")
        for i, sample in enumerate(samples[:3]):
            print(f"\n--- 样本 {i+1} ---")
            print(f"User Content预览: {sample['messages'][1]['content'][:200]}...")
            print(f"Assistant Content: {sample['messages'][2]['content']}")

def batch_process():
    """
    批量处理多个文件
    """
    print("=== 批量处理模式 ===")
    
    input_dir = input("请输入包含输入文件的目录路径: ").strip()
    output_dir = input("请输入输出目录路径 (默认: output): ").strip()
    
    if not output_dir:
        output_dir = "output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有文本文件
    import glob
    input_files = glob.glob(os.path.join(input_dir, "*.txt")) + \
                  glob.glob(os.path.join(input_dir, "*.text"))
    
    if not input_files:
        print(f"在目录 {input_dir} 中未找到.txt或.text文件，尝试查找所有文件...")
        input_files = [f for f in glob.glob(os.path.join(input_dir, "*")) 
                      if os.path.isfile(f) and not f.endswith('.jsonl')]
    
    if not input_files:
        print("在指定目录中未找到文件")
        return
    
    print(f"找到 {len(input_files)} 个文件")
    
    total_samples = 0
    processed_files = 0
    
    for input_file in input_files:
        try:
            # 生成输出文件名
            filename = os.path.basename(input_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{name_without_ext}.jsonl")
            
            print(f"处理: {filename}...")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            samples = parse_conversation_to_samples(content)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            total_samples += len(samples)
            processed_files += 1
            print(f"  → 生成 {len(samples)} 个样本")
            
        except UnicodeDecodeError:
            print(f"  → 跳过: 文件编码不是UTF-8")
        except Exception as e:
            print(f"  → 处理失败: {e}")
    
    print(f"\n批量处理完成！")
    print(f"处理文件数: {processed_files}/{len(input_files)}")
    print(f"总样本数: {total_samples}")
    print(f"输出目录: {output_dir}")

def test_with_example():
    """
    使用示例文本测试
    """
    print("=== 测试模式 ===")
    
    example_text = """【サーシャ】
「何だかんだ言って、私たちは朝日を中心に回ってる。瑞穂お嬢様を選ぶっていうのは、途中まで予想がつかなかったけれどね」
【小倉朝日】
「本当は……初めは、親友になろうって言われていたんです。でも、私はやっぱり忘れてても、遊星の部分が残ってた」
<M:eyes_closed>|<E:serious>
【サーシャ】
「それも意外なのよね……私、朝日はもう完全に女性化してると思ってたのに」
【サーシャ】
「ああ……もしかして、夏の間に反応させられちゃったとか？　なかなか勝てないものね、本能には」
【小倉朝日】
「うぅ……すみません。今でも反省してます、あの時のことは」
<E:worried>
【サーシャ】
「何、いろいろワケあり？　なーんだ、そういうことならもっと朝日の相談に乗ってあげたら良かった」
[场景切换]
【サーシャ】
「私が興味を持つ数少ない対象のひとつが、朝日だから。これからも楽しい話題を提供してちょうだい」
【小倉朝日】
「はい、よろしくお願いします。サーシャさん」
<E:smile>"""
    
    print("解析示例文本...")
    samples = parse_conversation_to_samples(example_text)
    
    print(f"\n生成 {len(samples)} 个样本:")
    
    for i, sample in enumerate(samples):
        print(f"\n=== 样本 {i+1} ===")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        
        # 显示对话历史内容
        user_content = sample['messages'][1]['content']
        print(f"\n对话历史预览:")
        history_start = user_content.find("【前後の会話】\n") + len("【前後の会話】\n")
        history_end = user_content.find("\n【小倉朝日】")
        if history_start > 0 and history_end > history_start:
            history = user_content[history_start:history_end]
            print(history)
    
    return samples

def main():
    """
    主函数，交互式处理
    """
    print("=== 对话文本转换工具 ===")
    print("将对话文本转换为训练样本格式")
    print("版本: 修复对话历史缺失问题")
    print()
    
    print("选择模式:")
    print("1. 处理单个文件")
    print("2. 批量处理多个文件")
    print("3. 测试示例文本")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "2":
        batch_process()
    elif choice == "3":
        test_with_example()
    else:
        # 获取输入文件路径
        while True:
            input_path = input("请输入输入文件路径: ").strip()
            if os.path.exists(input_path):
                break
            print(f"错误: 文件不存在 - {input_path}")
            print("请重新输入")
        
        # 获取输出文件路径
        default_output = os.path.splitext(input_path)[0] + ".jsonl"
        output_path = input(f"请输入输出文件路径 (默认: {default_output}): ").strip()
        if not output_path:
            output_path = default_output
        
        # 处理文件
        try:
            process_file(input_path, output_path)
            print()
            print("样本格式说明:")
            print("1. 每个样本包含一个完整的对话片段")
            print("2. system: 固定为小倉朝日的角色设定")
            print("3. user: 包含最多4句对话历史（含完整台词）")
            print("4. assistant: 小倉朝日的回复（情感标签 + 台词）")
            print("5. chat_template_kwargs: 固定设置")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()