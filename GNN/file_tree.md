├── IND-WhoIsWho/
    ├── ind_test_author_filter_public.json   --B榜测试作者
    ├── ind_test_author_submit.json  --B榜提交示例
    ├── ind_valid_author.json    --A榜测试作者
    ├── ind_valid_author_submit.json    --A榜提交示例
    ├── pid_to_info_all.json    --全部的论文数据
    ├── stopwords_English.txt    --英文停用词，用于Word2Vec
    ├── train_author.json  --训练集作者
├── model/    --存储训练好的模型
├── result/     --存储预测结果
├── sci_graph/    --使用scibert作为特征建立的图
├── scibert_model/    --scibert模型
    ├── config.json
    ├── pytorch_model.bin
├── scibert_tokenizer/    --scibert分词器
    ├── vocab.txt
├── w2v_graph/    --使用w2v作为特征建立的图
├── build_graph.sh    --建立所有图的指令集
├── build_sci_graph_test.py    --使用scibert作为作为特征建立测试集图集合
├── build_sci_graph_train.py    --使用scibert作为作为特征建立训练集图集合
├── build_w2v_graph_test.py    --使用w2v作为作为特征建立测试集图集合
├── build_w2v_graph_train.py    --使用w2v作为作为特征建立训练集图集合
├── install_requirements.sh    --安装依赖的指令集，主要是安装dgl-cu121需要单独安装
├── model.py    --模型定义文件
├── requirements.txt    --环境依赖文件
├── sci_model_test.py   --scibert作为特征训练的图模型进行推理
├── train_model.sh    --训练两个图模型的指令集
├── train_sci_model.py    --训练scibert作为特征建立的图模型
├── train_w2v_model.py    --训练w2v作为特征建立的图模型
├── w2v_model_test.py    --w2v作为特征训练的图模型进行推理
├── word2vec_model.py    --训练Word2vec模型并将word2vec特征保存到论文中
