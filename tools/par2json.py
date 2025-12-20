#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Parquet文件转换为JSON格式的工具

使用方法:
    python par2json.py input1.parquet [input2.parquet ...]

对于每个输入文件，将在同目录下生成同名的.json文件
"""

import os
import json
import argparse
import pandas as pd


def parquet_to_json(input_file, output_file=None):
    """
    将Parquet文件转换为JSON格式
    
    Args:
        input_file (str): 输入的Parquet文件路径
        output_file (str, optional): 输出的JSON文件路径。如果为None，则在输入文件同目录下生成同名的.json文件
    """
    # 如果未指定输出文件，自动生成
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.json"
    
    try:
        # 读取Parquet文件
        print(f"正在读取Parquet文件: {input_file}")
        df = pd.read_parquet(input_file)
        
        # 将DataFrame转换为JSON并写入文件
        print(f"正在转换并写入JSON文件: {output_file}")
        # 使用to_json方法，设置orient='records'以行为单位输出，indent=2格式化输出
        df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        
        print(f"转换完成！文件已保存至: {output_file}")
        print(f"共转换了 {len(df)} 条记录")
        
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        raise


def main():
    """
    主函数，处理命令行参数并执行转换
    """
    parser = argparse.ArgumentParser(description='将Parquet文件转换为JSON格式')
    parser.add_argument('input_files', nargs='+', help='输入的Parquet文件路径列表')
    
    args = parser.parse_args()
    
    # 统计信息
    total_files = len(args.input_files)
    success_count = 0
    error_count = 0
    
    print(f"开始处理 {total_files} 个Parquet文件...")
    print("=" * 70)
    
    # 遍历所有输入文件
    for i, input_file in enumerate(args.input_files, 1):
        print(f"处理文件 {i}/{total_files}: {input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在，跳过此文件")
            error_count += 1
            continue
        
        if not input_file.lower().endswith('.parquet'):
            print(f"警告: 输入文件 '{input_file}' 可能不是Parquet格式文件")
        
        try:
            # 执行转换（不指定输出文件，使用默认的同目录同名.json文件）
            parquet_to_json(input_file)
            success_count += 1
        except Exception:
            print(f"错误: 处理文件 '{input_file}' 失败，继续处理下一个文件")
            error_count += 1
        
        print("-" * 70)
    
    # 打印处理总结
    print("=" * 70)
    print(f"处理完成！总计 {total_files} 个文件")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")



if __name__ == "__main__":
    main()