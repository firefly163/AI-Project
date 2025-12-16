import os
import sys
import glob

def remove_percent_v_lines(text):
    """删除所有以%v或％v开头的行"""
    lines = text.split('\n')
    # 保留不以%v或％v开头的行
    filtered_lines = [line for line in lines if not line.strip().startswith(('%v', '％v'))]
    return '\n'.join(filtered_lines)

def process_file(input_file, output_file=None, encoding='cp932'):
    """处理单个文件"""
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
            input_text = f.read()
        
        # 处理文本
        output_text = remove_percent_v_lines(input_text)
        
        # 确定输出文件路径
        if output_file is None:
            # 在原文件名基础上添加_clean后缀
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_clean{ext}"
        
        # 写入输出文件
        with open(output_file, 'w', encoding=encoding) as f:
            f.write(output_text)
        
        print(f"  已处理: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"  处理文件 {input_file} 时发生错误: {e}")
        return False

def process_directory(directory_path, encoding='cp932'):
    """处理目录中的所有文本文件"""
    supported_extensions = ['.txt', '.s', '.script', '']
    processed_count = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            # 检查文件扩展名
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                if process_file(file_path, None, encoding):
                    processed_count += 1
            else:
                print(f"  跳过不支持的文件: {filename}")
    
    return processed_count

def main():
    """
    主函数：删除所有以%v或％v开头的行
    """
    print("删除以%v或％v开头的行处理工具")
    print("=" * 50)
    print("注意：此脚本会删除所有以%v或％v开头的行")
    print("=" * 50)
    
    # 选择编码
    print("\n请选择文件编码:")
    print("1. cp932 (日本Windows编码, 默认)")
    print("2. utf-8")
    print("3. euc-jp (日本Unix/Linux编码)")
    
    encoding_choice = input("请输入选择 (1-3, 默认为1): ").strip()
    if encoding_choice == '2':
        encoding = 'utf-8'
    elif encoding_choice == '3':
        encoding = 'euc-jp'
    else:
        encoding = 'cp932'
    
    total_processed = 0
    
    # 处理多个输入文件或目录
    while True:
        print("\n" + "-" * 50)
        print("请选择操作:")
        print("1. 处理单个文件")
        print("2. 处理整个目录")
        print("3. 使用通配符批量处理")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '4':
            break
        
        if choice == '1':
            # 处理单个文件
            input_path = input("请输入要处理的文件路径: ").strip()
            
            if not os.path.exists(input_path):
                print(f"文件不存在: {input_path}")
                continue
            
            if os.path.isfile(input_path):
                output_path = input("请输入输出文件路径（留空则自动生成）: ").strip()
                if not output_path:
                    output_path = None
                
                if process_file(input_path, output_path, encoding):
                    total_processed += 1
            else:
                print(f"不是文件: {input_path}")
                
        elif choice == '2':
            # 处理整个目录
            input_path = input("请输入要处理的目录路径: ").strip()
            
            if not os.path.exists(input_path):
                print(f"目录不存在: {input_path}")
                continue
            
            if os.path.isdir(input_path):
                count = process_directory(input_path, encoding)
                total_processed += count
                print(f"  完成目录处理，共处理 {count} 个文件")
            else:
                print(f"不是目录: {input_path}")
                
        elif choice == '3':
            # 使用通配符批量处理
            pattern = input("请输入通配符模式（例如: *.txt 或 scripts/*.s）: ").strip()
            
            try:
                files = glob.glob(pattern, recursive=True)
                if not files:
                    print(f"没有找到匹配的文件: {pattern}")
                    continue
                
                print(f"找到 {len(files)} 个文件:")
                for i, file_path in enumerate(files, 1):
                    print(f"  {i}. {file_path}")
                
                confirm = input("是否处理这些文件？(y/n, 默认为y): ").strip().lower()
                if confirm != 'n':
                    for file_path in files:
                        if os.path.isfile(file_path):
                            if process_file(file_path, None, encoding):
                                total_processed += 1
            except Exception as e:
                print(f"处理通配符时出错: {e}")
        
        else:
            print("无效的选择，请重新输入")
    
    print(f"\n处理完成！共处理 {total_processed} 个文件")
    
    # 显示示例
    print("\n" + "=" * 50)
    print("处理效果示例:")
    print("=" * 50)
    print("处理前:")
    print("------")
    print("%v_asa32169")
    print("【小倉朝日】")
    print("「遊ぶ……？　あの、私はお仕置きをされるんじゃなかったんですか？」")
    print("^chara01,file5:04")
    print()
    print("%v_nan32097")
    print("【名波七愛】")
    print("「口答えは許さない。心配しなくてもいい、態度次第では潰しはしない」")
    print()
    print("处理后:")
    print("------")
    print("【小倉朝日】")
    print("「遊ぶ……？　あの、私はお仕置きをされるんじゃなかったんですか？」")
    print("^chara01,file5:04")
    print()
    print("【名波七愛】")
    print("「口答えは許さない。心配しなくてもいい、態度次第では潰しはしない」")

def batch_mode():
    """
    批量模式：通过命令行参数批量处理
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='批量删除以%v或％v开头的行')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出文件路径（仅对单个文件有效）')
    parser.add_argument('-e', '--encoding', default='cp932', 
                       help='文件编码（默认：cp932）')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归处理目录')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # 处理单个文件
        output_file = args.output
        if process_file(args.input, output_file, args.encoding):
            print(f"处理完成: {args.input}")
    elif os.path.isdir(args.input):
        # 处理目录
        if args.recursive:
            # 递归处理
            for root, dirs, files in os.walk(args.input):
                for filename in files:
                    if filename.endswith(('.txt', '.s', '.script')) or '.' not in filename:
                        file_path = os.path.join(root, filename)
                        process_file(file_path, None, args.encoding)
        else:
            # 只处理当前目录
            process_directory(args.input, args.encoding)
    else:
        print(f"路径不存在: {args.input}")

if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        batch_mode()
    else:
        main()