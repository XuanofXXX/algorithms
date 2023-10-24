# About
本项目实现了对于 kaggle 上的 [AI Cat and Dog Images DALL·E Mini](https://www.kaggle.com/datasets/mattop/ai-cat-and-dog-images-dalle-mini) 数据集的分类，使用了K-means算法，实现了对于猫狗图片的分类。

# 成果展示
![成果展示](/image/result_Kmeans.png)
`clu1` 是第一簇，`clu2`是第二簇，可以发现与原始数据集中的猫狗图片的类别是近乎一致的。
```
Cosine Similarities: clu1 in cats 0.9999942131634348
Cosine Similarities: clu2 in dogs 0.9999941384177341
```
# 实现细节

## 数据集
数据集来自于 [AI Cat and Dog Images DALL·E Mini](https://www.kaggle.com/datasets/mattop/ai-cat-and-dog-images-dalle-mini) ，其中包含了 54 张猫的图片和 54 张狗的图片。

## 算法
- 我们使用了文澜API进行图片到特征向量的转换。每张图片得到一个2048维的向量。
- 使用余弦相似度作为衡量两个向量之间距离的指标。
- 使用了K-means算法，进行聚类。
- 最后进行散点图表示。
