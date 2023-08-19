from dataloader import *
from model import *
from loss import *
from ev_validate import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm






def train_func():
    last_loss = 10000
    patience = 5
    triggertimes = 0
    model_save_loc = "/home/nandhini/thesis/kin8nm/Evidential/kin8nm_evidential_model.pth"
    print('Training')
    Train_epoch_loss = []
    Val_epoch_loss = []
    for e in range(EPOCHS):
    
        model.train()
        train_running_loss = []

        for X_train_batch, y_train_batch in  (train_loader):

            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            optimizer.zero_grad()

            mu,alpha,beta,lamda = model(X_train_batch)
            loss = EvidentialLoss(mu,alpha,beta,lamda, y_train_batch.unsqueeze(1))
            train_loss = loss(mu,alpha,beta,lamda, y_train_batch.unsqueeze(1))
            train_running_loss.append(train_loss.tolist())
            
            train_loss.backward()
            optimizer.step()

            #train_running_loss.append(train_loss)
           
            

      
        train_loss = np.average(train_running_loss)
        Train_epoch_loss.append(train_loss)

        model.eval()
        
        with torch.no_grad():
            
            validation_loss = ev_validate(model, valid_loader, val_data, e)
            val_loss = validation_loss[0]
            pred = validation_loss[1]
            Val_epoch_loss.append(val_loss)
            
        print("Epoch: {}/{}.. ".format(e+1, EPOCHS),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Validation Loss: {:.3f}.. ".format(val_loss))

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
    plt.savefig('/home/nandhini/thesis/kin8nm/Evidential/kin8nm_evidential_loss.png')
    plt.show()

    print('DONE TRAINING')
                                     
    return model,Val_epoch_loss, Train_epoch_loss


model = Evidential_model(n_feature=8, n_hidden=50, n_output=1)
optimizer = optim.Adam(model.parameters(), lr=LR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kin8nm_model,valid_loss, training_loss = train_func()
                                     