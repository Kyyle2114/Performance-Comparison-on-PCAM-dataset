import torch
import matplotlib.pyplot as plt 
import numpy as np 

from tqdm import tqdm 

"""
---
Training code example 
---

model = SomeModel(...).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
es = EarlyStopping(patience=10, delta=0, mode='min', verbose=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)

history = {'train_loss' : [],
           'val_loss': [],
           'train_accuracy': [],
           'val_accuracy': []}
           
EPOCHS = 100
max_loss = np.inf           

for epoch in range(EPOCHS):
    train_loss, train_acc = model_train(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, scheduler=scheduler, tqdm_disable=False)
    val_loss, val_acc = model_evaluate(model=model, data_loader=val_loader, criterion=criterion, device=device)
    
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    
    es(val_acc)
    # Early Stop Check
    if es.early_stop:
        break

    if val_loss < max_loss:
        print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
        max_loss = val_loss
        torch.save(model.state_dict(), 'Best_Model.pth')

    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, accuracy: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f} \n')

---
"""



def model_train(model, data_loader, criterion, optimizer, device, scheduler=None, tqdm_disable=False):
    """
    Model train

    Args:
        model (torch Model)
        data_loader (torch DataLoader)
        criterion (torch loss)
        optimizer (torch optimizer)
        device (devide): cpu / cuda / mps 
        scheduler (torch scheduler, optional): lr scheduler. Defaults to None.
        tqdm_disable (bool, optional): if True, tqdm progress bars will be removed. Defaults to False.

    Returns:
        loss, accuracy: Avg loss, acc for 1 epoch  
    """
    model.train()
    
    running_loss = 0
    correct = 0

    for X, y in tqdm(data_loader, disable=tqdm_disable):
        X, y = X.to(device), y.float().to(device)
        
        optimizer.zero_grad()
        
        output = model(X)
        output = torch.sigmoid(output).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # binary classification
        pred = output >= torch.FloatTensor([0.5]).to(device)
        correct += pred.eq(y).sum().item()  
        running_loss += loss.item() * X.size(0)
    
    if scheduler:
        scheduler.step() 
        
    accuracy = correct / len(data_loader.dataset) # Avg acc
    loss = running_loss / len(data_loader.dataset) # Avg loss
 
    return loss, accuracy


def model_evaluate(model, data_loader, criterion, device):
    """
    Model train

    Args:
        model (torch Model)
        data_loader (torch DataLoader)
        criterion (torch loss)
        device (devide): cpu / cuda / mps 

    Returns:
        loss, accuracy: Avg loss, acc for 1 epoch  
    """
    model.eval()
    
    with torch.no_grad(): 
        running_loss = 0
        correct = 0

        for X, y in data_loader:
            X, y = X.to(device), y.float().to(device)   
            
            output = model(X)
            output = torch.sigmoid(output).squeeze()
            
            # binary classification
            pred = output >= torch.FloatTensor([0.5]).to(device)
            correct += torch.sum(pred.eq(y)).item()
            running_loss += criterion(output, y).item() * X.size(0)
            
        accuracy = correct / len(data_loader.dataset)
        loss = running_loss / len(data_loader.dataset)  

        return loss, accuracy
    
def plot_loss(history):
    """
    Plot training loss

    Args:
        history (dict): training history
    """
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

def plot_acc(history):
    """
    Plot training accuracy

    Args:
        history (dict): training history
    """
    plt.plot(history['train_accuracy'], label='train', marker='o')
    plt.plot(history['val_accuracy'], label='val',  marker='o')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()
    
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        Pytorch EarlyStopping

        Args:
            patience (int, optional): patience. Defaults to 10.
            delta (float, optional): threshold to update best score. Defaults to 0.0.
            mode (str, optional): 'min' or 'max'. Defaults to 'min'(comparing loss -> lower is better).
            verbose (bool, optional): verbose. Defaults to True.
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False
