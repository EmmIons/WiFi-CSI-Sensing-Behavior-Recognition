import os
import pandas as pd
import torch

if __name__ == "__main__":
    # dataset
    # path
    folder_path = './dataset'

    data_list = []
    labels = []
    len_of_everyClass = []
    for filename in os.listdir(folder_path):
        if filename.endswith('Amplitude.xlsx'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_excel(file_path, header=None)
            tensor = torch.tensor(df.values)
            len_of_everyClass.append(tensor.size(0)/30)
            data_list.append(tensor)
            # choose label
            # [0, 6]==[bow, fall, null, run, sit, stand, walk]
            if filename.startswith('bow'):
                labels.append(0)
            elif filename.startswith('fall'):
                labels.append(1)
            elif filename.startswith('null'):
                labels.append(2)
            elif filename.startswith('run'):
                labels.append(3)
            elif filename.startswith('sit'):
                labels.append(4)
            elif filename.startswith('stand'):
                labels.append(5)
            elif filename.startswith('walk'):
                labels.append(6)


    data = torch.cat(data_list, dim=0)
    data = torch.split(data, 30, dim=0)
    data = torch.stack(data, dim=0)
    # label, one-hot vectors
    label_onehot = torch.zeros(size=(int(sum(len_of_everyClass)), 7))
    l = 0
    for i in range(0, len(labels)):
        label_onehot[l:l+int(len_of_everyClass[i]), labels[i]] = 1
        l += int(len_of_everyClass[i])
    print(data.size(), labels, len_of_everyClass)
    torch.save(data, './dataset_processed/data.pt')
    torch.save(label_onehot, './dataset_processed/label.pt')



