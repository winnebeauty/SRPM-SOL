import os
import json

def create_pdb_index_map(pdb_folder, output_file):
    """
    遍历指定的 pdb 文件夹，建立 id 和文件路径的映射关系。

    Args:
        pdb_folder (str): pdb 文件夹的绝对路径。
        output_file (str): 保存索引的文件名，默认保存为 "pdb_index.json"。
    
    Returns:
        dict: 返回生成的 id 和文件路径的映射字典。
    """
    index_map = {}
    current_id = 1  # 从 1 开始分配 id
    
    for subfolder in ["train", "valid", "test"]:
        folder_path = os.path.join(pdb_folder, subfolder)
        
        # 确保子文件夹存在
        if not os.path.exists(folder_path):
            print(f"Warning: {subfolder} 文件夹不存在，跳过...")
            continue
        
        # 遍历当前子文件夹中的所有文件
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdb"):  # 只处理 .pdb 文件
                    file_path = os.path.join(root, file)
                    index_map[str(current_id)] = file_path
                    current_id += 1

    # 保存到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(index_map, f, indent=4)
    print(f"索引映射保存到 {output_file}，共处理了 {len(index_map)} 个文件。")

    return index_map

