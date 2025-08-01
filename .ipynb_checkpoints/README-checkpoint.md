# Efficient and Adversarially Robust Object Detection

## Usage
### If something keeps failing?
Email me at [yyuan459@uwo.ca](mailto:yyuan459@uwo.ca) ideally including [OBJDETECTION2025] in the subject line, and 
a summary of the issue. In the email give me as many details as possible to reproduce the problem.
### Make sure to download:
- COCO2017 annotation JSONs into data/COCO2017/annotations

- Pretrained [yolov3 weights](https://www.kaggle.com/datasets/shivam316/yolov3-weights/data) into a folder called weights

### Instructions (for PC usage)
1. >pip install -r requirements.txt
2. Consult the "Install PyTorch" section of this page: https://pytorch.org and run the command
3. From there run the cells in Pipeline.ipynb in order (WIP as of 2025-07-21. Should work albeit images are shifted with 50+ epochs.)

### For SHARCNET usage
1. SSH to the cluster being used and run:
   1. >git clone https://github.com/maxinimize/Efficient-Robust-Object-Detection
   2. Enter the new directory and run 
      >python run.py
2. So far testing was done in the JupyterLab of clusters so connect to one such instance and:
   ### In terminal:
   1. > module load opencv
   2. Find the softwares tab on the left side of the JupyterLab (circled), search opencv (underlined), 
   load opencv/4.11.0 (hover over it and click on load) by the end your softwares should look like the picture below
   ![](instruction images\Screenshot 2025-07-21 150046.png)
   3. > . ENV/bin/activate
      
      > python -m ipykernel install --user --name=ENV --display-name "ENV"
   4. Go back into file browser on the left tab and click into Pipeline.ipynb and your middle/right screen should 
   look similar to above
   ### In Pipeline.ipynb
   5. Your screen should look similar to this image:
   ![](instruction images\Screenshot 2025-07-21 151736.png)
   6. Click the area circled in the image and select the ENV kernel
   7. Click on the first code cell (arrow), click the run cell button (underlined) and continue using code cells 
   linearly. If the import statements gives an error related to cv2 then try restarting the kernel which is the refresh
   button in the same area as the run button.
   8. Once you finish running everything, you should get results in data/results.
   
   #### Note1: Importing takes a longer time than what you may be used to
   #### Note2: If you already have ENV kernel set up from a previous session then still do the instructions and select kernel even if it says ENV is the current kernel
## Credits
Some of the code is from:

https://github.com/inspire-group/hydra

https://github.com/miladlink/TinyYoloV2

https://github.com/eriklindernoren/PyTorch-YOLOv3

https://github.com/trsvchn/coco-viewer

Dataset from:

https://cocodataset.org/#home
