#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# In[22]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[23]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[63]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[25]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[26]:


model


# In[27]:


img = 'https://image.cnbcfm.com/api/v1/image/106947859-GettyImages-167503543.jpg?v=1632760991'


# In[28]:


results = model(img)
results.print()


# ## testing....coco+ yolov5

# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[30]:


results.render()


# In[31]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[32]:


get_ipython().system('git clone https://github.com/heartexlabs/labelImg')


# In[33]:


get_ipython().system('pip install pyqt5 lxml --upgrade')
get_ipython().system('cd labelImg && pyrcc5 -o libs/resources.py resources.qrc')


# In[45]:


get_ipython().system('cd yolov5 && python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2')


# In[49]:


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/last.pt', force_reload=True)


# In[103]:


import os
import time
import uuid


# In[104]:


img = 'https://previews.123rf.com/images/ammentorp/ammentorp1611/ammentorp161100520/66380796-young-friends-having-fun-on-shopping-carts-multiracial-young-people-racing-on-shopping-cart.jpg'


# In[108]:


results = model(img)
class_ids = results.pred[0].detach().cpu().numpy()[:, -1].astype(int)
confidences = results.pred[0].detach().cpu().numpy()[:, 4]

print(class_ids)
print(confidences)


# In[109]:


results.print()


# In[110]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[98]:


image_links = {
    'https://previews.123rf.com/images/ammentorp/ammentorp1611/ammentorp161100520/66380796-young-friends-having-fun-on-shopping-carts-multiracial-young-people-racing-on-shopping-cart.jpg',
    'https://images.ctfassets.net/9l3tjzgyn9gr/aR8LRVbCLG8t3AvhK2LQo/3fca3c7325d478c9183a510726afa921/SmartCart_1224.jpg?fm=jpg&fl=progressive&q=50&w=1200',
    'https://www.cincinnati.com/gcdn/-mm-/b11ddaaebd6c5629dbc3a3c97a2349b520951aad/c=0-50-534-351/local/-/media/Cincinnati/Cincinnati/2014/06/25/1403697633000-downtowngrocery1.jpg?width=1200&disable=upscale&format=pjpg&auto=webp',
    'https://townsquare.media/site/84/files/2020/11/GettyImages-129472166.jpg',
    'https://www.drivingdynamics.com/hubfs/iStock-623710682.jpg',
    'https://philressler.com/wp-content/uploads/2022/01/shopping-cart.jpeg'
    
}

# Iterating...
for i in image_links:
    print(i)
    
    results = model(i)
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.imshow(np.squeeze(results.render()))
    plt.show()
    print("\n")


# ## real time capture - mall video or web cam

# In[ ]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# ## final test with timer

# In[ ]:


import cv2
import pyttsx3
import time

# Initialize the speech synthesis engine
engine = pyttsx3.init()

# Set the speech rate
engine.setProperty('rate', 150)

# Set the path to your YOLOv5 model weights and configuration files
model_weights = 'path/to/your/model/weights.pt'
model_config = 'path/to/your/model/yolov5s.yaml'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights, yaml_path=model_config)

# Initialize variables
start_time = time.time()
stranded_time = 0
stranded_flag = False

# Open the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    # Get the class labels and confidences
    class_ids = results.pred[0].detach().cpu().numpy()[:, -1].astype(int)
    confidences = results.pred[0].detach().cpu().numpy()[:, 4]
    
    # Check if "no owner" class is detected
    if 0 in class_ids:
        stranded_flag = True
        stranded_time = time.time() - start_time
    else:
        stranded_flag = False
        stranded_time = 0
        start_time = time.time()
    
    # Display the frame
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    # Check if the "no owner" is detected for more than 5 minutes
    if stranded_flag and stranded_time > 300:
        engine.say("Shopping cart stranded for 5 minutes")
        engine.runAndWait()
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

