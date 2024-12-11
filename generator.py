import torch
import numpy as np

class BatchGenerator(torch.utils.data.Dataset):
    def __init__(self, instances, labels, batch_size=1):
        self.instances = np.array(instances)
        self.labels = np.array(labels)

        self.shuffle()
        
        self.batch_size = min(batch_size, len(instances))

    def shuffle(self):
        p = np.random.permutation(len(self.instances))
        self.instances = np.array(self.instances)[p]
        self.labels = np.array(self.labels)[p]

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        X = []
        y = []
        for i in range(0, self.batch_size):
            gt_X = self.instances[idx * self.batch_size + i]
            X.append(gt_X)
            
            #gt_y = np.zeros(len(targets))
            #gt_y[int(self.labels[idx * self.batch_size + i])] = 1
            gt_y = int(self.labels[idx * self.batch_size + i])
            y.append(gt_y)
        
        #return torch.tensor(np.array(X, dtype=np.float32)), torch.tensor(np.array(y, dtype=np.float32))
        return torch.tensor(np.array(X, dtype=np.float32)), y
        
    def on_epoch_end(self):
        self.shuffle()