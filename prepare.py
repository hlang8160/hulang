#!/bin/env python
# -*- coding: UTF-8 -*- 

import re
import sys
# import os

# raw_data='./raw_data/valid.en-zh.en.sgm'
# tmp_data='./tmp_data/valid.en-zh.en.sgm'
#os.mknod('valid.en-zh.en.sgm')
def get_seg(line):
    pattern=re.compile(r'<seg id=.*>(.*)</seg>')
    if pattern.search(line):#如果找到就true
        new_line=pattern.search(line).group(1).strip()#group(1)返回第一个括号匹配的内容
        return new_line
    return False
if __name__=='__main__':
    if len(sys.argv)!=2:
        sys.stderr.write('error %s' % __file__)
        sys.exit(-1)
    file_name=sys.argv[1]
    with open(file_name,'r') as f:
        for line in f:
            new_line=get_seg(line)
            if new_line:
                sys.stdout.write(new_line.strip())
                sys.stdout.write('\n')
            # with open(tmp_data,'w') as g:
            #     g.write(new_line.strip())
            #     g.write('\n')
            #print line.strip()
            # print('\n')