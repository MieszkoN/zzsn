import torch
import torch.optim as optim
import copy, numpy as np
import matplotlib.pyplot as plt
import sys


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    train_losses = []
    train_accuracies = [ 0.0 ]
    validation_losses = []
    validation_accuracies = [0.0]
    
    
    for epoch in range(num_epochs):
        print('-' * 40)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in ['train', 'val']:
            current_loss = 0.0
            current_corrections = 0

            for i,(images, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                current_loss += loss.item() * images.size(0)
                current_corrections += torch.sum(predictions == labels.data)

                sys.stdout.flush()
                
                
            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_accuracy = current_corrections.double() / dataset_sizes[phase]
            
            if phase == 'train':
                avg_loss = epoch_loss
                train_accuracy = epoch_accuracy
            else:
                validation_loss = epoch_loss
                validation_accuracy = epoch_accuracy

            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy.item())
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy.item())
                
        print('Train Loss: {:.4f} Train Accuracy: {:.4f}'.format(avg_loss, train_accuracy))
        print( 'Validation Loss: {:.4f} Validation Accuracy: {:.4f}'.format(validation_loss, validation_accuracy))
    
    print('Best validation accuracy: {:4f}'.format(best_accuracy))

    model.load_state_dict(best_model_weights)
    return model, train_losses, train_accuracies, validation_losses, validation_accuracies




def test_model(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_loss = 0.0
    current_corrections = 0

    with torch.no_grad():
        for images, labels in dataloader['test']:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            current_loss += loss.item() * images.size(0)
            current_corrections += torch.sum(predictions == labels.data)

    dataset_size = len(dataloader['test'])
    average_loss = (current_loss / dataset_size) /100
    accuracy = current_corrections.double() / dataset_size

    print('Test Loss: {:.4f} Test Accuracy: {:.4f}'.format(average_loss, accuracy))
    return average_loss, accuracy


def draw_figure(epochs, y1, y2, title, y_label, start):
    x = list(range(start, epochs))  

    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.show()
    
    
    
def generateAdamOptimizer(model):
    return optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001)

def generateSgdOptimizer(model):
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)