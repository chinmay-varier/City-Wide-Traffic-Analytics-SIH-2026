
# City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics



## Installation

Installing Required Files

```bash
pip install -r requirements.txt
```

To Install the data set and sample video
Extract the zip file from this google drive
[Google Drive Link](https://drive.google.com/file/d/1-Qq7AWY4QVYRkVsBZj2RrdiPXxfgs_jB/view)

-> After extraction move both the folders into the alpr_project directory


<img width="193" height="323" alt="image" src="https://github.com/user-attachments/assets/233e0495-7088-41c7-9285-59e588689f41" />


## Seeing Trajectory and Heatmap

Based on some already loaded data, heatmap and trajectory can be 
generated

```bash
python .\Trajectory_Heatmap\main.py
```

## Read License Plates using OCR

Reading License plate on a finetuned 45mins) OCR Model

```bash
cd alpr_project
python -m src.main
```
