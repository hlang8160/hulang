raw_data_en='./raw_data/valid.en-zh.en.sgm'
raw_data_zh='./raw_data/valid.en-zh.zh.sgm'
tmp_data_en='./tmp_data/valid.en-zh.en'
tmp_data_zh='./tmp_data/valid.en-zh.zh'

tmp_data_cw_en='./tmp_data/valid.en'
tmp_data_cw_zh='./tmp_data/valid.zh'

#从源文件中提取 英文和中文
python prepare.py $raw_data_en >$tmp_data_en
python prepare.py $raw_data_zh >$tmp_data_zh

#将中文进行分词操作
# python jieba_cws.py $tmp_data_en >$tmp_data_cw_en
python jieba_cws.py $tmp_data_zh >$tmp_data_cw_zh
