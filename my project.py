import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


## 导入数据
data = pd.read_excel('data2.xlsx', sheet_name='Sheet1', encoding='utf-8')
X = data[['死亡率','治愈率','每百万人口确诊数','人口密度（公里²）','医疗质量指数(HQA)','医院床位（每千人）','老龄化65岁和65岁以上的人口（占总人口的百分比）','国际移徙者（占人口的百分比）']]
X = 1.0*(X - X.mean())/X.std()

estimator = KMeans(n_clusters=4) #构造聚类器，分为5类
estimator.fit(X) #聚类
labels = estimator.labels_ #获取聚类标签

data['cluster_db'] = labels
data.sort_values('cluster_db')
print(data.groupby('cluster_db').count()) ## 对各个簇数据分组汇总
DataFrame(data).to_excel('data2.xlsx', sheet_name='Sheet1') ## 聚类结果输出


tsne=TSNE()
data_zs = 1.0*(data - data.mean())/data.std()
tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index) # 转换数据格式

tsne['cluster_db'] = data['cluster_db']
d=tsne[data['cluster_db']==0]
plt.plot(d[0],d[1],'r+', label='0')

d=tsne[data['cluster_db']==1]
plt.plot(d[0],d[1],'go', label='1')

d=tsne[data['cluster_db']==2]
plt.plot(d[0],d[1],'b*', label='2')

d=tsne[data['cluster_db']==3]
plt.plot(d[0],d[1],'y^', label='3')

plt.legend()
plt.show()

