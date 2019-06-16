import jieba

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords
def init_jieba():
    jieba.enable_parallel(3)
    jieba.load_userdict("model/ingredients/userdict.txt")
def cutTrainData():
    input_filename = 'model/ingredients/train.data'
    output_filename = 'model/train.csv'
    print("加载原始数据...",end='')
    file = open(input_filename)
    sentence = file.read()
    print("完成")
    print("加载停用词表...", end='')
    stopwords = set(stopwordslist("model/ingredients/stopwords.txt"))
    print("完成")

    seg_list = jieba.cut(sentence)
    
    print("分词完成，正在写入文件 %s" %(output_filename))
    storefile = open(output_filename,mode='w')
    storetext = ''
    for x in seg_list:
        if x == '\n':
            storetext += x
        # 这里 if 的判断条件不可调换，因为对于 '\n', isspace 也会返回true
        elif x.isspace() or (x in stopwords) or x.isdigit() or x.isnumeric():
            continue
        else:
            storetext += x + ' '
    storefile.write(storetext)

    storefile.close()
    file.close()
    print("分词完成")
def cutTestData():
    input_filename = 'model/ingredients/test.data'
    output_filename = 'model/test.csv'
    print("加载原始数据...", end='')
    
    file = open(input_filename)
    sentence = ''
    # sentence = file.read()
    for line in file:
        sentence += line.split(maxsplit=1)[1]
    print("完成")

    print("加载停用词表...", end='')
    stopwords = stopwordslist("model/ingredients/stopwords.txt")
    print("完成")

    seg_list = jieba.cut(sentence)
    print("分词完成，正在写入文件 %s" %(output_filename))
    storefile = open(output_filename,mode='w')
    storetext = ''
    for x in seg_list:
        if x == '\n':
            storetext += x
        # 这里 if 的判断条件不可调换，因为对于 '\n', isspace 也会返回true
        elif x.isspace() or (x in stopwords) or x.isdigit() or x.isnumeric():
            continue
        else:
            storetext += x + ' '
            
    storefile.write(storetext)
    storefile.close()
    file.close()
    print("分词完成")
def main():
    # 加载用户 词语
    init_jieba()
    # 对训练集和测试集数据切分
    cutTestData()
    cutTrainData()
if __name__ == "__main__":
    main()