import torch
import torch.optim as optim
import multiprocessing
import time
import csv 
import preprocess as prep
import models
import utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0
    train_entanglement = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss, entanglement = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_entanglement += entanglement

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}\tEntanglement: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(), entanglement))

    train_loss /= len(train_loader)
    train_entanglement /= len(train_loader)
    print('Train set Average loss:', train_loss)
    print('Train set Avg Entanglement:', train_entanglement)
    return train_entanglement


def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0
    test_entanglement = 0

    # two np arrays of images
    original_images = []
    rect_images = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss, entanglement = model.loss(output, data, mu, logvar)
            test_loss += loss.item()
            test_entanglement += entanglement
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tEntanglement: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item(), entanglement))

    test_loss /= len(test_loader)
    test_entanglement /= len(test_loader)
    print('Test set Average loss:', test_loss)
    print('Test set Average entanglement:', test_entanglement)
    return test_entanglement


# parameters
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EPOCHS = 25

LATENT_SIZE = 32
LEARNING_RATE = 1e-3

USE_CUDA = True
PRINT_INTERVAL = 1
LOG_PATH = './logs/log.pkl'
MODEL_PATH = './checkpoints/'
COMPARE_PATH = './comparisons/'

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

# training code

train_ids, test_ids = prep.split_dataset()
print('num train_images:', len(train_ids))
print('num test_images:', len(test_ids))

data_train = prep.ImageDiskLoader(train_ids)
data_test = prep.ImageDiskLoader(test_ids)

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

print('latent size:', LATENT_SIZE)

model = models.BetaVAE(latent_size=LATENT_SIZE, beta=250).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":

    start_epoch = model.load_last_model(MODEL_PATH) + 1
    train_entanglements = []
    test_entanglements = []

    for epoch in range(start_epoch, EPOCHS + 1):
        train_entanglement = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
        test_entanglement = test(model, device, test_loader, return_images=5)

        train_entanglements.append([epoch, float(train_entanglement)])
        test_entanglements.append([epoch, float(test_entanglement)])

    # writing the data into the file 
    with open('train_entanglement_hdarts.csv', 'w+', newline ='') as file:
        write = csv.writer(file) 
        write.writerows(train_entanglements) 
    with open('test_entanglement_hdarts.csv', 'w+', newline ='') as file:
        write = csv.writer(file) 
        write.writerows(test_entanglements) 

