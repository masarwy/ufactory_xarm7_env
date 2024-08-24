import os
import pickle

if __name__ == '__main__':
    folder_path = '../data/pose_estimation'
    all_data_folder = os.listdir(folder_path)
    for file in all_data_folder:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print('ok')
