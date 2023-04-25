目前的代码以CLINC为例，其他数据集类似,首先运行main.py，接着运行pobe.py
* main.py: finetune数据集，构建KNN索引，关键参数如下：
  1. pretrained_path: 预训练模型地址
  2. model_prefix: finetune后模型地址前缀
  3. database_prefix: KNN索引地址，构建后包含index_path, token_path.   
* pobe.py: Pobe OOD计算，将构建的KNN索引地址填入config/pobe.yaml的index_path, token_path即可,默认K=1024