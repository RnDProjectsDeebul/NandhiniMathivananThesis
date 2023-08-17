
from dataloader import *
from model import *
from loss import *
import numpy as np
import scipy.stats as stats
from metrics import *
import torch.optim as optim
from config import *
from sklearn import preprocessing


def validate(model, dataloader, data, epoch):
    print('Validating')
  
    valid_running_loss = []
    val_difference = []
    var = []
    
    
    
  

    with torch.no_grad():
            

            for i, (X_val_batch, y_val_batch) in enumerate(valid_loader):
            
            
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_batch = y_val_batch[:, :, 0]



                #y_val_pred, alpha,beta = model(X_val_batch)
                y_val_pred, var = model(X_val_batch)
                std = np.sqrt(var)
            


                val_loss = criterion(y_val_pred, y_val_batch,var)
                #val_loss = GeneralGaussianNLLLoss(y_val_pred, y_val_batch.unsqueeze(1), alpha,beta)

                valid_running_loss.extend(val_loss.tolist())
        
                val_difference.extend((y_val_batch - y_val_pred).tolist())

 
               

    diff = np.average(val_difference)    
    #print(valid_running_loss)

    valid_loss = np.average(valid_running_loss)
    avg_var  = np.average(var) 
    
    return valid_loss,diff







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraspKeypointModel().to(device)
#model = Evidential_model(n_feature=8, n_hidden=2, n_output=1)

optimizer = optim.Adam(model.parameters(), lr=LR)
#validation_loss = validate(model, valid_loader, test_dataset, 1)


    


