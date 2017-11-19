#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

#第一行制定coding:utf-8是为了指定当前py文件中含有中文，对整个文件指定编码方式
# print __file__
import jieba
import sys
str="我是一个江西中国人"
seg_list=jieba.cut(str.strip().decode('utf-8'))#首先按照utf-8方式解析对象，转换成unicode模式
#读取文件将utf-8转化为unicode，decode()
#编辑之后将unicode转化为utf-8,encode()

#jieba.cut是将string划分为 单独的词语 ，之后连接起来join
str1=u' '.join(seg_list).encode('utf-8')
# print str1

#查看系统的编码方式
print sys.getdefaultencoding()
a="测试"
b=u"测试"
print a
print b
