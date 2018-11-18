"""Useful functions for training the model
"""
import copy
from collections import defaultdict
import numpy as np
import torch
from ...utils import auto_tqdm
from ...utils import timelogger


@timelogger
def train_model(model, dataloader, criterion, optimizer, metrics=None, valid_dataloader=None, num_epochs=6):
    """Train model on a dataloader
    :param model: A PyTorch Neural Network model (subclass of `nn.Module`)
    :param dataloader: DataLoader for train set
    :param criterion: a PyTorch Loss Class, e.g. `torch.nn.modules.loss.CrossEntropyLoss`
    :param optimizer: a PyTorch optimizer to update the model parameters
    :param metrics: a sequence of functions to calculate the metrics of the model, e.g. accuracy
    :param valid_dataloader: DataLoader for validation set
    :param num_epochs: number of epoches to train, default 6
    :return: Trained model with train_history
    """
    if hasattr(model, 'train_history'):
        print("Continue training for {} epoches on an already trained model...".format(num_epochs))
        train_history = model.train_history
        train_history.epoches = train_history.epoches + num_epochs
        if train_history.has_validation:
            assert valid_dataloader is not None, "valid_dataloader has to be provided."
        else:
            assert valid_dataloader is None, "This model has been trained without validation data."
    else:
        print("Training for {} epoches on a new untrained model...".format(num_epochs))
        train_history = TrainHistory(epoches=num_epochs)
        train_history.has_validation = True if valid_dataloader else False

    if hasattr(model, 'best_loss'):
        best_loss = model.best_loss
    else:
        best_loss = np.Inf

    if metrics is not None:
        pass

    best_model_weights = copy.deepcopy(model.state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training the model on:", device)
    print("=" * 99)
    model = model.to(device)
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase if valid_dataloader is not None
        if valid_dataloader is None:
            phases = ['train']
            dataloaders = {'train': dataloader}
        else:
            phases = ['train', 'valid']
            dataloaders = {'train': dataloader, 'valid': valid_dataloader}

        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            batch_num = len(dataloaders[phase].batch_sampler)
            batch_size = dataloaders[phase].batch_size

            if phase == 'train':
                tqdm = auto_tqdm("batches")
                dataloader = tqdm(dataloaders[phase], total=batch_num)
                dataloader.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
            else:
                dataloader = dataloaders[phase]

            # Iterate over data.
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward the network and track grad train_history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # backward + optimize only in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (batch_num * batch_size)
            loss_key = 'loss' if phase == 'train' else 'val_loss'
            train_history.history[loss_key].append(epoch_loss)

            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        # print loss
        lastest_train_loss = train_history.history['loss'][-1]
        print('Train loss: {:.6f}'.format(lastest_train_loss), end='\t')
        if valid_dataloader:
            lastest_val_loss = train_history.history['val_loss'][-1]
            print('Validation loss: {:.6f}'.format(lastest_val_loss))

    assert train_history.epoches == len(train_history.history['loss'])
    model.train_history = train_history
    model.best_loss = best_loss
    print("=" * 99)
    if valid_dataloader: print('Best validation loss: {:6f}'.format(best_loss))

    # load best model weights and return the model
    model.load_state_dict(best_model_weights)
    return model


class TrainHistory(object):
    """PyTorch train train_history
    """
    def __init__(self, epoches):
        assert isinstance(epoches, int) and (epoches > 0), "epoches has to be a positive integer."
        self._epoches = epoches
        self.history = defaultdict(list)

    @property
    def epoches(self):
        return self._epoches

    @epoches.setter
    def epoches(self, val):
        current_epoches = self.epoches
        assert val > current_epoches, "Epoches can only increase"
        self._epoches = val

    def is_empty(self):
        return not bool(self.history)

    def __repr__(self):
        if 'loss' in self.history:
            return 'TrainHistory(epoches={})'.format(self.epoches)
        else:
            return '<Empty TrainHistory object>'
