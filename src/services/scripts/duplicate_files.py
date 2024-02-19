import os
import shutil


def duplicate_files_to_subfolder(subfolder_path, file_names):
    for file_name in file_names:
        src_file_path = os.path.join(path, file_name)
        dst_file_path = os.path.join(subfolder_path, file_name)
        shutil.copy2(src_file_path, dst_file_path)


def process(path: str, file_names: list[str]):
    """Insert the three missing files to each subfolder."""
    if not os.path.isdir(path):
        raise ValueError(f"Invalid path: {path}")

    for item in os.listdir(path):
        subfolder_path = os.path.join(path, item)

        if os.path.isdir(subfolder_path):
            existing_files = []

            for file_name in file_names:
                file_path = os.path.join(subfolder_path, file_name)

                if os.path.isfile(file_path):
                    existing_files.append(file_name)

            duplicate_files_to_subfolder(subfolder_path, file_names)

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        os.remove(file_path)


if __name__ == "__main__":
    file_names = ["adjacencyMatrix.txt", "aisleSubdivision.txt", "positionList.txt"]
    instance_names = []
    # instance_names = ['warehouse_A', 'warehouse_B', 'warehouse_C', 'warehouse_D']

    for instance_name in instance_names:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "data",
            instance_name,
        )
        process(path, file_names)
        print(f"Processed {instance_name}")

    print(f"DuplicateFiles | Instance processed {instance_names}")
