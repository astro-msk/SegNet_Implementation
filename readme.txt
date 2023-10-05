1. Each variant has one unique .py file
2. Each variant's file will generate a .json file to import into run.py to train and execute the model

3. Link to dataset : https://drive.google.com/drive/folders/1E2KxpxLSwnyZxuhla9UXt2YvZQWndrOD?usp=sharing

4. Download the dataset and set the path to dataset in camvid_data_loader.py
5. camvid_data_loader.py will take the dataset as input and will generate .npy files
6. .npy files will be stored in a new created directory ./data/
7. .json file names should changed in run.py to execute a specific variant
