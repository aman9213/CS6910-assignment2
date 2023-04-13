## DL2_file

In this file I have trained the model with random hyperparameter and checked the accuracy.\
To run this file first Upload iNaturalist to Google drive in My drive section and then mount your drive in google colab.\
After that you can run each cell one by one.
      
### Important Functions or class
`Class ConvNet(nn.Module):`  
**Methods:**  
**Method name:** `__init__()` 

**Description**: resonsible for initializing the cnn object. 

**Arguments**: self,K,S,factor,activation,batch_norm,drop_out,nodes,num_class=10\
K: is number of filters\
S: is size of filters\
factor: is filter organization factor [1=same ,2=double,0.5=half]\
activation: is type of activation function\
batch_norm: is for batch normalization\
drop_out: is for drop out at fully connected layer\
nodes: is number of nodes in fully connected layer\
num_class: is number of output classes which is 10

**Returns**:Initialize the CNN 

**Methods:**

**Method name**: `forward()`\
**Description**: dothe forward propagation in CNN\
**Arguments**: self,x\
x: is the input to the CNN

**Returns**: It will returns the predicted  outputs

`model=ConvNet(K,S,factor,activation,batch_norm,drop_out,nodes,num_class=10)` will call ConvNet class and intialize th model.
& `output= model(image)` give output of forward prop.

## DL2_part_A(2) File

In this file I am doing wandb configuration with different Hyperparameter.\
**Followings are the sweep configuration**

```python
sweep_configuration={'name':'EE22s037','method':'bayes',
'metric':{'name':'val_acc','goal':'maximize'},
'parameters':{
'filter_n':{'values':[32,64]},

'filter_org':{'values':[1,2,0.5]},
'activation':{'values':['Relu','Gelu','Silu','Mish']},
'batch_norm':{'values':['Yes','No']},
'drop_out':{'values':[0.2,0.3]}

}
}
```

**Methods**

**Method name**: `train()`\
**Description**: It will train on train data and vaalidate on validate data,and print minimum loss and maximum accuaracy and plot it.\
and also log the plot on wandb.\
**Arguments**:None\

`sweep_id=wandb.sweep(sweep=sweep_configuration,entity="amanvb-9213",project='DL-assignment2')`

`wandb.agent(sweep_id,function=train,count=20)`.\This code will gnerate sweep id and start the sweeping.

## DL2_part_A_final File

After done with sweeping ,take the best hyperparameter and now train the complete model with these and evaluate on test data.\
**Methods**:

**Methode name**: `train_best_model()`\
**Description**: This function train with the best hyperparameter and print min losses and maximum accuracies and plot it.\
**Arguments**:None\
**Returns**: `best trained model.[best_model=train_best_model()]`

The last cell of this file will plot the 3 images from each classes and predict its label.

              
              
       

       
    
          
         
