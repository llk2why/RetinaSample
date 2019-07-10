# RETINA 

## Brief Introduction  
This project is aimed to reconstruct images with mosaic

## Code Intepretation

### <span>config.py</span>
Configure is stored in this script

### <span>preprocess.py</span>
This python script chop original images into patches of resolution 128x128, gather relevant information of this task.

### <span>model.py</span>
Network is defined here

### <span>dataset.py</span>
Dataloader definition class

### <span>run.py</span>
Main console file

## Runing Instructions
```
python config.py
python prepreocess.py
python run.py
```