import os


def set_auto_in_file(file_path):
    text_list = []
    with open(file_path) as f:
        text_list = f.readlines()

    fp_num = 0
    for i, line in enumerate(text_list):
        if 'FP' in line:
            fp_num = fp_num+1
            text_list[i] = 'FP' + str(fp_num) + '=Auto\n'
        elif 'RAWDATA' in line and 0 < fp_num < 3:
            fp_num = fp_num + 1
            text_list.insert(i,"FP3=Auto\n")

    with open(file_path, 'w') as updated_file:
        updated_file.writelines(text_list)


def set_fp_to_auto(path):
    directory = os.fsencode(path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".Trial.enf"):
            set_auto_in_file(path + '\\' + filename)
            # print(os.path.join(directory, filename))
            continue