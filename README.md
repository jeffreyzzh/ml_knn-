# ml_knn_digit_recognition
kNN算法识别手写数字

---

#### KNN算法
K近邻(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。所谓K近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。[百度百科](https://baike.baidu.com/item/%E9%82%BB%E8%BF%91%E7%AE%97%E6%B3%95/1151153)  
这里我们使用Python实现KNN分类器并且把手写数字进行识别。

---

#### 数据集
数据集分为训练集和测试集。
本案例先用训练集1900张手写图片训练模型，再对900张图片测试。
![手写数据集](https://raw.githubusercontent.com/jeffreyzzh/ml_knn_digit_recognition/master/images/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%B1%95%E7%A4%BA.png)   
数据集已经全部转化为01文本格式 。 

---

#### 程序运行 
`python knn_digits.py`

#### 运行结果
![程序截图](https://raw.githubusercontent.com/jeffreyzzh/ml_knn_digit_recognition/master/images/knn_run.png)  
错误率为0.01，换句话说正确率99%左右。 