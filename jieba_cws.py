#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

#进行分词操作
import jieba
import sys
def seg_list(string):
    str=jieba.cut(string.strip().decode('utf-8'))
    return u' '.join(str).encode('utf-8')

if __name__=='__main__':
    if len(sys.argv)!=2:
        sys.stderr.write('error %s' %__file__)
        exit(-1)
    file_name=sys.argv[1]
    with open(file_name,'r') as f:
        for line in f:
            new_line=seg_list(line)
            sys.stdout.write(new_line.strip())
            sys.stdout.write('\n')


