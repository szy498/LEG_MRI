Python 3.13.5 (v3.13.5:6cb20a219a8, Jun 11 2025, 12:23:45) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import os
... import numpy as np
... import SimpleITK as sitk
... import warnings
... # 忽略无关警告，不影响运行
... warnings.simplefilter("ignore")
... 
... 
... 
... data_root = r"/home/user/desktop/data"
... # ================================================================================
... 
... # 定义：原始数据的上级文件夹路径
... src_dixon_path = os.path.join(data_root, "shuju", "Dixon")
... # 定义：5个输出文件夹的名称 + 严格对应series_0~4，顺序千万不要改！一一对应！
... target_folder_names = ["dixon_water", "dixon_fat", "dixon_ff", "series_3", "series_4"]
... # 拼接5个输出文件夹的完整路径（都在data根目录下，与shuju平行）
... target_folder_paths = [os.path.join(data_root, folder_name) for folder_name in target_folder_names]
... 
... # 自动创建所有输出文件夹，不存在则新建，存在则不操作，避免报错
... for folder_path in target_folder_paths:
...     if not os.path.exists(folder_path):
...         os.makedirs(folder_path)
...         print(f"已自动创建文件夹: {folder_path}")
...     else:
...         print(f"文件夹已存在: {folder_path}")
... 
... # 定义核心处理函数：读取单个患者的DCM文件夹，生成5个nii.gz并保存到对应目录
... def process_patient_dcm(patient_dcm_dir, patient_name, group_name):
...     """
...     :param patient_dcm_dir: 单个患者的DCM文件夹绝对路径
...     :param patient_name: 患者名字(文件夹名)
...     :param group_name: 分组名 NOR/SAR
...     """
...     try:
...         # 1. 读取DCM序列（核心：读取患者的所有DCM文件为3D影像）
...         reader = sitk.ImageSeriesReader()
...         dicom_file_list = reader.GetGDCMSeriesFileNames(patient_dcm_dir)
...         if len(dicom_file_list) == 0:
...             print(f"【警告】{group_name} - {patient_name} 文件夹内无有效DCM文件，跳过！")
...             return
...         
...         reader.SetFileNames(dicom_file_list)
...         image = reader.Execute()
...         image_array = sitk.GetArrayFromImage(image)  # 转为numpy数组，shape=(Z, H, W) Z=层数 H=高 W=宽
...         z_size = image_array.shape[0]
...         block_size = 71  # 固定切片层数，和你论文、原代码一致：Dixon序列71层
...         
...         # 2. 计算要分割的块数：固定分割为5块 → series_0 ~ series_4，刚好对应5个输出文件夹
...         num_blocks = 5
...         print(f"\n正在处理：{group_name} - {patient_name} | 总切片数: {z_size} | 分割为 {num_blocks} 块")
... 
...         # 3. 循环分割并保存，严格按顺序对应series_0~4 → 5个目标文件夹
...         for i in range(num_blocks):
...             # 计算当前块的起止索引，防止越界
...             start_idx = i * block_size
...             end_idx = min((i + 1) * block_size, z_size)
...             # 按Z轴切割影像数据
...             slice_block = image_array[start_idx:end_idx, :, :]
...             # 转回sitk影像格式 + 复制原始DCM的空间信息【重中之重，原代码缺失！】
...             slice_block_sitk = sitk.GetImageFromArray(slice_block)
...             slice_block_sitk.CopyInformation(image)  # 保留像素间距、坐标系、方向矩阵等医学信息
...             
...             # 4. 定义输出文件名：患者名_series_x.nii.gz ，方便溯源
...             output_nii_name = f"{patient_name}_series_{i}.nii.gz"
...             # 找到当前series对应的输出文件夹路径
            target_save_path = os.path.join(target_folder_paths[i], output_nii_name)
            # 保存nii.gz文件
            sitk.WriteImage(slice_block_sitk, target_save_path)
            print(f"✅ 保存成功: {target_folder_names[i]} / {output_nii_name}")
    
    except Exception as e:
        # 出错时打印信息，继续处理下一个患者，不会中断程序
        print(f"【错误】处理 {group_name} - {patient_name} 失败: {str(e)}")
        return

# ============================ 主程序：遍历所有数据 ============================
if __name__ == "__main__":
    # 遍历 NOR 和 SAR 两个平行文件夹
    for group in ["NOR", "SAR"]:
        group_dir = os.path.join(src_dixon_path, group)
        if not os.path.exists(group_dir):
            print(f"\n【提示】未找到文件夹: {group_dir}，跳过该分组！")
            continue
        
        print(f"\n==================================================")
        print(f"开始处理分组: {group}")
        print(f"==================================================")
        
        # 遍历当前分组下的所有患者名字文件夹
        patient_folders = os.listdir(group_dir)
        for patient_folder in patient_folders:
            patient_dir = os.path.join(group_dir, patient_folder)
            # 只处理文件夹，过滤掉可能的文件
            if os.path.isdir(patient_dir):
                process_patient_dcm(patient_dir, patient_folder, group)

    # 全部处理完成后的提示
    print("\n==================================================")
    print("🎉 所有患者数据处理完成！")
    print(f"输出文件位置：{data_root} 下的 {target_folder_names} 共5个文件夹")
