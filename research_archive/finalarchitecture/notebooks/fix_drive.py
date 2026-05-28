import os

def replace_in_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    for old_str, new_str in replacements.items():
        if old_str in content:
            content = content.replace(old_str, new_str)
            modified = True
            
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {os.path.basename(path)}")

replacements = {
    "'/kaggle/input/drive-retinal-vessel/training/images'": "'/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images'",
    "'/kaggle/input/drive-retinal-vessel/training/1st_manual'": "'/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual'",
    "/kaggle/input/drive-retinal-vessel/training/images/": "/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images/",
    "/kaggle/input/drive-retinal-vessel/training/1st_manual/": "/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual/",
    "/kaggle/input/drive-retinal-vessel/": "/kaggle/input/datasets/zionfuo/drive2004/DRIVE/"
}

base_dir = r"c:\Users\shive\projects\cv\retinal-disease-detection-cv\finalarchitecture\notebooks"
files = [
    "vessel_specialist.py",
    "README.md"
]

for file in files:
    replace_in_file(os.path.join(base_dir, file), replacements)

print("Done fixing DRIVE dataset paths.")
