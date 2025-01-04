import pandas as pd

# 加载放射组学矩阵
file_path = 'C:\\Users\\dell\\Desktop\\radiomics_matrix.xlsx'
df = pd.read_excel(file_path)

# 查看前几行数据以确认加载正确
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# # 绘制某个特征的直方图
# sns.histplot(df['original_firstorder_Entropy'], kde=True)
# plt.title('Histogram of Entropy')
# plt.show()
#
# # 计算特征之间的相关性
correlation_matrix = df.corr()

# 绘制热力图
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# 小提琴图展示不同模态下的对比度
sns.violinplot(x='Modality', y='original_glcm_Contrast', data=df)
plt.title('Contrast across Different Modalities')
plt.show()

