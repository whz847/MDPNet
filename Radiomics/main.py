import os
import pandas as pd
from radiomics import featureextractor  # PyRadiomics

# 配置路径
data_dir = r"D:\dataset\BraTS2020T+V\BraTS2020_TrainingData\test"  # 数据集主目录
output_csv = "radiomics_matrix.csv"  # 输出的 Radiomics 矩阵文件名

# 配置 PyRadiomics 参数
params = {
    "setting": {
        "normalize": True,  # 是否归一化影像
        "resampledPixelSpacing": [1, 1, 1],  # 重新采样到一致的分辨率
        "interpolator": "sitkBSpline",  # 插值方法
        "binWidth": 25,  # 灰度分箱宽度
    },
    "featureClass": {
        "glcm": [],  # 仅提取灰度共生矩阵（GLCM）特征
        # "glcm": [],
        # "firstorder": [],
        # "shape": []
    },
}

# 创建特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# 初始化结果矩阵
radiomics_matrix = []

# 遍历所有病例文件夹
for case_folder in os.listdir(data_dir):
    case_path = os.path.join(data_dir, case_folder)
    if os.path.isdir(case_path):  # 确保是文件夹
        try:
            # 获取当前病例的分割文件
            seg_file = os.path.join(case_path, f"{case_folder}_seg.nii")
            if not os.path.exists(seg_file):
                print(f"Segmentation file not found for {case_folder}, skipping...")
                continue

            # 遍历当前病例的 MRI 模态文件
            for modality in ["flair", "t1", "t1ce", "t2"]:
                mri_file = os.path.join(case_path, f"{case_folder}_{modality}.nii")
                if not os.path.exists(mri_file):
                    print(f"{modality} file not found for {case_folder}, skipping...")
                    continue

                # 提取特征
                result = extractor.execute(mri_file, seg_file)

                # 转换为字典，附加文件名（病例 ID 和模态）
                result_dict = {key: value for key, value in result.items() if not key.startswith("diagnostics")}
                result_dict["PatientID"] = case_folder
                result_dict["Modality"] = modality  # 添加模态信息

                # 将特征保存到矩阵
                radiomics_matrix.append(result_dict)

                print(f"Features extracted for {case_folder} ({modality})")
        except Exception as e:
            print(f"Error processing {case_folder}: {e}")

# 将结果保存为 CSV
if radiomics_matrix:
    df = pd.DataFrame(radiomics_matrix)
    df.to_csv(output_csv, index=False)
    print(f"Radiomics matrix saved to {output_csv}")
else:
    print("No features extracted. Please check the input files.")
