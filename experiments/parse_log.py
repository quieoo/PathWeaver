import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
# 指定关键字，默认是 'loss'
parser.add_argument('--key', type=str, default='loss')
# 输出文件，默认是标准输出
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--cut', type=bool, default=False)

args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = f.readlines()

if args.cut:
    # 找到包含关键字的行，在该行将文件切分，并保留多个文件(文件名: output_0, output_1, ...)
    buffer_lines = []
    file_id=0
    for line in lines:
        if args.key in line:
            with open(args.output+'_'+str(file_id), 'w') as f:
                f.writelines(buffer_lines)
            file_id+=1
            buffer_lines = []
        else:
            buffer_lines.append(line)
    # 最后一个文件
    with open(args.output+'_'+str(file_id), 'w') as f:
        f.writelines(buffer_lines)
else:
    # 遍历每一行，查找包含关键字的行
    for line in lines:
        if args.key in line:
            if args.output is None:
                print(line)
            else:
                with open(args.output, 'a') as f:
                    f.write(line)
