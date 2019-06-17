除去python代码文件之外:
---------------------
* corpus.csv为语料库文件，是train.csv和test.csv的混合
* fine-tune.txt 是部分实验记录
* test.cvs 和 train.csv 是预处理（分词）后的测试和训练数据

如何加载 model:
--------------
* 通过 执行 submit.py 来产生模型的最终结果
* 若有新的测试集(已分好词)，请现将其改名为 test.csv,然后存于与submit.py相同的目录下即可