from dataloader import *
from model import *
from loss import *
from validate import *
import torch.optim as optim
import matplotlib.pyplot as plt
from config import *
import pandas as pd
import albumentations as A



def train_func(params, train_loader, valid_loader, valid_dataset, model, optimizer, logger = None):
    last_loss = 10000
    patience = 5
    trigger_times = 0
    model_save_loc = "/home/nandhini/thesis/Grasp/Gaussian_norm.pth"
    print('Training')
    Train_epoch_loss = []
    Val_epoch_loss = []
   
    for e in range(params[EPOCHS]):
    
        model.train()
        train_running_loss = []

       
        for i, (X_train_batch, y_train_batch) in enumerate(train_loader):


            X_train_batch, y_train_batch = X_train_batch.to(params['device']), y_train_batch.to(params['device'])
    
            optimizer.zero_grad()
            outputs, var = model(X_train_batch)

            train_loss = criterion(y_train_batch[:, :, 0], outputs, var)
           

            train_running_loss.extend(train_loss.tolist())


            train_loss.mean().backward()
            optimizer.step()
 
        train_loss = np.average(train_running_loss)
        Train_epoch_loss.append(train_loss)

        model.eval()
        
        with torch.no_grad():
            
            validation_loss = validate(model, valid_loader, valid_dataset, e)
            val_loss = validation_loss[0]
           
            pred = validation_loss[1]
          
            Val_epoch_loss.append(val_loss)
            
        print("Epoch: {}/{}.. ".format(e+1, EPOCHS),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Validation Loss: {:.3f}.. ".format(val_loss))
        
        if logger!= None:
            logger['plots/training/train_loss'].log(train_loss)
            logger['plots/training/validation_loss'].log(val_loss)

        current_loss = val_loss
        print('The Current Loss:', current_loss)
        
        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times, '\n')
            
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
            
        else:
            print('trigger times: 0', '\n')
            trigger_times = 0
            torch.save(model, model_save_loc)
            
        last_loss = current_loss
        model.train()             
        
    plt.figure(figsize=(10, 7))
    plt.plot(Train_epoch_loss, color='orange', label='train loss')
    plt.plot(Val_epoch_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('"/home/nandhini/thesis/Grasp/Gaussian_norm.png')
    plt.show()

    print('DONE TRAINING')
                                     
    return model, Train_epoch_loss, Val_epoch_loss




