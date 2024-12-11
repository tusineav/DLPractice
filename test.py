import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from generator import BatchGenerator
import json

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(6, 64, 1, 1)
        self.conv2 = torch.nn.Conv2d(128, 64, 1, 1)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, 1)
        self.conv4 = torch.nn.Conv2d(256, 256, 1, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1, 1)

        self.dp1 = torch.nn.Dropout(0.4)
        self.dp2 = torch.nn.Dropout(0.4)
        self.dp3 = torch.nn.Dropout(0.4)
        self.dp4 = torch.nn.Dropout(0.6)
        
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        self.bn6 = torch.nn.BatchNorm1d(512)
        self.bn7 = torch.nn.BatchNorm1d(256)
        
        self.lin1 = torch.nn.Linear(2048, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 40)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    
    train_instances = json.load(open("datasets/mscoco/annotations/instances_train2017.json", "r"))
    val_instances = json.load(open("datasets/mscoco/annotations/instances_val2017.json", "r"))
    
    print(train_instances.keys())
    print(train_instances['images'][0])
    print(train_instances['annotations'][0])
    exit()
    
    #fig, ax = plt.subplots(1, 2)
    
    
    
    
    train_X, train_y = load_data('train')
    test_X, test_y = load_data('test')
    
    X = np.concatenate([train_X, test_X], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)
    
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]
    
    #Reserve 10% of the original data for testing
    test_X = X[int(0.9 * len(X)):]
    test_y = y[int(0.9 * len(y)):]
    
    test = {'X': test_X, 'y': test_y}
    pickle.dump(test, open("test_data.pickle", "wb"))
    
    X = X[:int(0.9 * len(X))]
    y = y[:int(0.9 * len(y))]
    
    
    kfolds = 5
    ratio = 1 / kfolds
    X_folds = []
    y_folds = []
    for i in range(0, kfolds):
        X_folds.append(list(X[int(ratio * i * len(X)): int(ratio * (i + 1) * len(X))]))
        y_folds.append(list(y[int(ratio * i * len(y)): int(ratio * (i + 1) * len(y))]))
    
    model_name = "standard_cnn1d_no_pool"
    
    os.makedirs("trained_models/", exist_ok=True)
    os.makedirs("trained_models/%s/" % model_name, exist_ok=True)
    for fold in range(0, kfolds):
        os.makedirs("trained_models/%s/fold_%d/" % (model_name, fold + 1),  exist_ok=True)
        train_X = []
        train_y = []
        for i in range(0, kfolds):
            if i != fold:
                train_X.extend(X_folds[i])
                train_y.extend(y_folds[i])
        
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        val_X = np.array(X_folds[fold])
        val_y = np.array(y_folds[fold])
        
        exit()
        train_loader = BatchGenerator((train_X, train_y), num_workers=8, batch_size=32, shuffle=True)
        val_loader = BatchGenerator((val_X, val_y), num_workers=8, batch_size=16, shuffle=True)
         

        model = Model().to(device)

        optimizer = torch.optim.Adam(model.parameters())

        train_losses = []
        val_losses = []
        permutated_val_losses = []
        train_accuracies = []
        val_accuracies = []
        permutated_val_accuracies = []
        
        epochs = 100
        for epoch in range(1, epochs + 1):
            print("Fold %d, epoch %d" % (fold + 1, epoch))
            
            model.train()
            
            losses = []
            preds = []
            labels = []
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                print("\ttrain batch", batch_idx + 1, "/", len(train_loader), end="\r")
                
                batch_X = batch_X.to(device)
                batch_X = torch.transpose(batch_X, 1, 2)
                
                optimizer.zero_grad()
                
                predictions = model.forward(batch_X)
                
                gt = np.zeros((len(batch_y), 40), dtype="float")
                for index, label in enumerate(batch_y):
                    gt[index][label] = 1
                    
                preds.append(predictions.max(dim=1)[1].cpu().detach().numpy())
                labels.append(batch_y)

                loss = torch.nn.BCELoss()(predictions, torch.tensor(gt, dtype=torch.float, device='cuda'))
                loss.backward()
                
                losses.append(loss.cpu().detach().numpy())
                optimizer.step()

            print()
            
            labels = np.concatenate(labels).reshape(-1)
            preds = np.concatenate(preds)
            train_loss = np.average(losses)
            train_accuracy = accuracy_score(labels, preds)
            print("\ttrain loss: " + str(round(train_loss, 5)) + ", train accuracy: " + str(round(train_accuracy * 100, 2)))
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            print("\tValidating on original test samples:")
            model.eval()

            losses = []
            preds = []
            labels = []
            with torch.no_grad():
                for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
                    print("\t\tvalidation batch", batch_idx + 1, "/", len(val_loader), end="\r")
                    
                    batch_X = batch_X.to(device)
                    batch_X = torch.transpose(batch_X, 1, 2)
                    
                    predictions = model.forward(batch_X)
                    
                    gt = np.zeros((len(batch_y), 40), dtype="float")
                    for index, label in enumerate(batch_y):
                        gt[index][label] = 1
                        
                    preds.append(predictions.max(dim=1)[1].cpu().detach().numpy())
                    labels.append(batch_y)

                    loss = torch.nn.BCELoss()(predictions, torch.tensor(gt, dtype=torch.float, device='cuda'))
                    losses.append(loss.cpu().detach().numpy())
            print()
            
            labels = np.concatenate(labels).reshape(-1)
            preds = np.concatenate(preds)
            val_loss = np.average(losses)
            val_accuracy = accuracy_score(labels, preds)

            print("\t\tval loss: " + str(round(val_loss, 5)) + ", val accuracy: " + str(round(val_accuracy * 100, 2)))

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print("\tValidating on permutated test samples:")
            labels = []
            preds = []
            losses = []
            with torch.no_grad():
                for batch_idx, (batch_X, batch_y) in enumerate(val_loader):
                    print("\t\tvalidation batch", batch_idx + 1, "/", len(val_loader), end="\r")
                    
                    batch_X = batch_X.to(device)
                    batch_X = torch.transpose(batch_X, 1, 2)
                    
                    predictions = model.forward(batch_X)
                    
                    gt = np.zeros((len(batch_y), 40), dtype="float")
                    for index, label in enumerate(batch_y):
                        gt[index][label] = 1
                        
                    preds.append(predictions.max(dim=1)[1].cpu().detach().numpy())
                    labels.append(batch_y)

                    loss = torch.nn.BCELoss()(predictions, torch.tensor(gt, dtype=torch.float, device='cuda'))
                    losses.append(loss.cpu().detach().numpy())
                    
            print()
            
            labels = np.concatenate(labels).reshape(-1)
            preds = np.concatenate(preds)
            permutated_val_loss = np.average(losses)
            permutated_val_accuracy = accuracy_score(labels, preds)

            print("\t\tpermutated val loss: " + str(round(permutated_val_loss, 5)) + ", permutated val accuracy: " + str(round(permutated_val_accuracy * 100, 2)))
            
            permutated_val_losses.append(permutated_val_loss)
            permutated_val_accuracies.append(permutated_val_accuracy)
            
            print()
            print()
    
        print(train_losses, val_losses, permutated_val_losses, train_accuracies, val_accuracies, permutated_val_accuracies)
        
        torch.save(model, "trained_models/%s/fold_%d/model_fold_%d.torch" % (model_name, fold + 1, fold + 1))
                
        ax[0].plot(range(1, epochs + 1), train_losses, 'b', label='train loss')
        ax[0].plot(range(1, epochs + 1), val_losses, 'r', label='val loss')
        ax[0].plot(range(1, epochs + 1), permutated_val_losses, 'g', label='permutated val loss')
        ax[1].plot(range(1, epochs + 1), train_accuracies, 'c', label='training accuracy')
        ax[1].plot(range(1, epochs + 1), val_accuracies, 'm', label='validation accuracy')
        ax[1].plot(range(1, epochs + 1), permutated_val_accuracies, 'y', label='permutated validation accuracy')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("BCE Loss")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("Accuracy")
        plt.savefig("trained_models/%s/fold_%d/fold_%d.png" % (model_name, fold + 1, fold + 1))
        plt.clf()
