import os

def main():
    input_folder = "C:\Projects\VP2\MLdata\orthos_tiles\\1"
    # input_folder = "C:\Projects\VP2\MLdata\\test"

    if not os.path.exists(input_folder):
        print("Input folder {} doesn't exist".format(input_folder))
        exit(1)

    input_files = os.listdir(input_folder)
    input_size = len(input_files)
    print("Input folder {} contains {} files".format(input_folder, input_size))

    # rename ml_class -> lagis
    # for file_name in input_files:
    #     if file_name.startswith("ml_class"):
    #         new_name = "lagis"+file_name[8:]
    #         os.rename(os.path.join(input_folder, file_name), os.path.join(input_folder, new_name))

    # add underscore to the file name
    for file_name in input_files:
        name, ext = os.path.splitext(file_name)
        new_name = name + '_' + ext
        os.rename(os.path.join(input_folder, file_name), os.path.join(input_folder, new_name))


if __name__ == '__main__':
    main()
