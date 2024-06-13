## 尝试方法：
构造相关的字符集合得分、更多模型的训练及融合（RandomForest、CNN、SVM）、CAN后处理、10折交叉验证、参数搜索、特征筛选.....
## 方案一：
针对所给数据集，分析数据之间的关系构造特征工程，再将每篇论文的title，abstract，venue，org用空格连接，去脏去停之后，用word2vec训练词向量，与之前构造的特征进行拼接，用LightGBM、Xgboost、Catboost三个基学习器模型进行5折交叉验证训练之后，在通过LightGBM和Catboost作为元学习器将其上述三个模型的输出作为特征进行stacking得到的结果加权融合作为最终预测结果。
特征工程所用特征：

		len_title, title_cnt,  # 标题字符长度，词个数
 	  	len_abstract, abstract_cnt,  # 摘要字符长度，词个数
 	  	len_keywords, keywords_cnt,  # 关键词字符长度，词个数
 		len_venue, venue_cnt,  # 期刊字符长度，词个数
 		len_title_abstract, title_abstract_cnt,  # 标题+摘要字符长度，词个数
 		len_title_keywords, title_keywords_cnt,  # 标题+关键字字符长度，词个数
 		len_keywords_abstract, keywords_abstract_cnt,  # 关键字+摘要字符长度，词个数
 		len_all,  # 标题字符长度+摘要字符长度+关键词字符长度
 		all_cnt,  # 标题词个数+摘要词个数+关键词词个数
 		year, year_is_reanson,  # 发表年份  #年份是否合理
 		keywords_abstract_ratio,  # 关键词在摘要中的占比
   
 		org_is_null,  # 作者机构是否为空
 		co_org_ratio,  # 共同作者中与作者机构相同的作者个数占总作者个数比
 		co_org_cnt,  # 共同作者中与作者机构相同的作者个数
 		not_co_org_ratio,  # 共同作者中与作者机构不相同的作者个数占总作者个数比
 		not_co_org_cnt,  # 共同作者中与作者机构不相同的作者个数
 		org_dif_cnt,  # 不同机构的个数
 		co_author_cnt,  # 共同作者个数
 		author_org_len,  # 作者所在机构的字符长度
 		author_org_cnt,  # 作者所在机构的词个数
 		all_org_len,  # 共同作者机构字符总长度
 		all_name_len,  # 共同作者名字字符总长度
 		all_org_cnt,  # 共同作者机构词总个数
 		all_name_cnt,  # 共同作者名字词总个数
 		country_dif_cnt,  # 共同作者中所属不同国家的机构数
 		country_dif_ratio,  # 共同作者中所属不同国家的机构数占总的个数比
 		sum(country_cnt),  # 共同作者中国家总个数
   
 		max_jaro_venue,  # venue最大jaro
 		min_jaro_venue,  # venue最小jaro
 		mean_jaro_venue,  # venue平均jaro
 		max_min_jaro_venue,  # venue最大减最小jaro

 		max_jaro_year,  # year最大jaro
 		min_jaro_year,  # year最小jaro
 		mean_jaro_year,  # year平均jaro
 		max_min_jaro_year,  # year最大减最小jaro
 		max_jaro_org,  # org最大jaro
		min_jaro_org,  # org最小jaro
	 	mean_jaro_org,  # org平均jaro
	 	max_min_jaro_org,  # org最大减最小jaro
		country_cnt  # 作者机构所属国家个数
		country占比
  
	上述部分特征的‘mean', 'max', 'min', 'var', 'std', 'median’，‘max-min’，以及多项式特征

