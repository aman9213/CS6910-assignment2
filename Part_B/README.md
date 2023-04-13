TO run tese file first upload iNaturalist.zip file in google drive in MyDrive, then mount it in google colab.\
zip file will exract with this code-
```python
import zipfile
import os
zip_ref = zipfile.ZipFile('/content/drive/MyDrive/nature_12K.zip', 'r')
zip_ref.extractall('/nature') #Extracts the files into the /nature folder
zip_ref.close()
```
### DL2_part_B file

This file will do transfer learning with ResNet50 model without fixed feature extraction,find  val_accuracy and plot the accuracy\
**Methods->**\
**Method name**: `train()`\
**Description**: It will train and validate the model with test data the model.\
**Arguments**: model,loss,optimizer,scheduler,num_epoch/
model: model which you want to tarin (here ResNet50).\
loss: Loss function\
optimizer: optimizer which you want to use (here sgd)\
scheduler: for decay of learning rate\
num_epoch: Number of epochs\
**Returns**: It return a trained model

**below code is for defining the model-**
```python
model=models.resnet50(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,10)  ## redefining the last layers for 10 classes
model.fc.requires_grad=True
model=model.to(device)
loss_fun=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
```
**call the train function with this code**
`model=train(model,loss_fun,optimizer,lr_scheduler,num_epoch)`

### DL2_part_B_final file
In this I added more code on above file for transfer learining of model with features extraction, and printed accuracy and plotted.\
**Here I am defining model with this code**
```python
model=models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc=nn.Linear(model.fc.in_features,10)
model=model.to(device)
loss_fun=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.fc.parameters(),lr=0.001,momentum=0.9)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```
**Methods:** same as in file DL2_part_B\
**call the train function**
`model = train(model,loss_fun, optimizer,lr_scheduler, num_epoch)`

**Note- Complete code is available in DL2_part_b_final,for runing/evalution please check this file only.**



