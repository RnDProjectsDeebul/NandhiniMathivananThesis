import neptune

from dataloader import ImageCoordinateDataset
import albumentations as A
import albumentations.augmentations.transforms as A_transform
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from train import train_func
from model import *

params = {"learning_rate": 0.001, 
          "optimizer": "Adam", 
          "EPOCHS": 100,
          "LR" :  0.001,
          "batch_size" : 64,
          "device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          }

train_transform = A.Compose([
    A_transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Rotate(p=0.5), 
    A.VerticalFlip(p=0.2),
    A.PixelDropout(dropout_prob=0.01),
    A.CenterCrop(height=480, width=640, p=0.2),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy'))

valid_transform = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

image_files = [f for f in os.listdir('/home/nandhini/thesis/Grasp/archive/Data') if f.endswith('r.png')]
train_size = int(0.8 * len(image_files))
test_size = len(image_files) - train_size
tra_dataset, val_dataset = random_split(image_files, [train_size, test_size])
train_dataset = ImageCoordinateDataset('/home/nandhini/thesis/Grasp/archive/Data',files = tra_dataset, train_transform=train_transform, valid_transform=None)
valid_dataset = ImageCoordinateDataset('/home/nandhini/thesis/Grasp/archive/Data',files = val_dataset, train_transform=None, valid_transform=valid_transform)


train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

model = GraspKeypointModel().to(params['device'])
optimizer = optim.Adam(model.parameters(), lr=params['LR'])

run = neptune.init_run(
    project="m.nandhinishree/Thesis",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDU2NTNhZC05ZDJmLTQ4MDctOThiNS1lNWJkZGI1ZDM3YWMifQ==",
) 

logger = False 
if logger:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="provide the name of the project",
    api_token="provide the api key",
    tags = [],
    name= [],
    )
    # Logging hyperparameters.
    run['config/hyperparameters'] = params
  
else:
    run = None

kin8nm_model, training_loss, val_loss = train_func(params, train_loader, valid_loader, valid_dataset, model, optimizer, logger = run)


run.stop()