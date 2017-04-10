import pickle

from shutil import copyfile

labels = pickle.load(open("../objects/labels.pkl", "rb"))
patient_and_path = pickle.load(open("../objects/patient_and_path.pkl", "rb"))

data_path = "../data/stage1/negs/"
for id, cancer in labels.items():
    print(cancer)
    if cancer == "0":
        print(id)
        if id in patient_and_path:
            print(id)
            for file_path in patient_and_path[id]:
                print(file_path)
                copyfile(file_path, data_path)
