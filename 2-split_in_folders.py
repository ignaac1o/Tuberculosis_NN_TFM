from pathlib import Path
import shutil
import os

file_source ='/Users/ignacioalmodovarcardenas/Desktop/Msc in Estatistics for Data Science/TFM/Tuberculosis_NN_TFM/data'
file_destination_photos ='/Users/ignacioalmodovarcardenas/Desktop/Msc in Estatistics for Data Science/TFM/Tuberculosis_NN_TFM/photos'
file_destination_anotations ='/Users/ignacioalmodovarcardenas/Desktop/Msc in Estatistics for Data Science/TFM/Tuberculosis_NN_TFM/anotations'

for file in Path(file_source).glob('*.csv'):
    shutil.copy(os.path.join(file_source,file),file_destination_anotations)

for file in Path(file_source).glob('*.jpg'):
    shutil.copy(os.path.join(file_source,file),file_destination_photos)