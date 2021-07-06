import torch
from torch.utils.data._utils.collate import default_collate

from torchvision import datasets, transforms
def my_collate(batch):
    modified_batch = []
    for item in batch:
        image, label = item
        if label == 7:
            modified_batch.append(item)
    # print(modified_batch.__sizeof__())
    return default_collate(modified_batch)
def get_loader(dataset, batch_size):
    if dataset == 'mnist':
        # MNIST dataset.
        train_dataset = datasets.MNIST(root='./data/mnist',
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())
        # idx = train_dataset.train_labels == 7
        # train_dataset.data = train_dataset.train_labels[idx]
        # train_dataset.targets = train_dataset.train_data[idx]


        test_dataset = datasets.MNIST(root='./data/mnist',
                                      train=False,
                                      transform=transforms.ToTensor())

        # idx2 = test_dataset.train_labels == 7
        # test_dataset.data = test_dataset.test_labels[idx2]
        # test_dataset.targets = test_dataset.test_data[idx2]



        # Data loader.
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   drop_last=True, collate_fn = my_collate)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=True, collate_fn = my_collate)




        return train_loader, test_loader
def get_loader_oneclass(dataset, batch_size):
    if dataset == 'mnist':
        # MNIST dataset.


        oneclass_dataset = datasets.MNIST(root='./data/mnist',
                                      train=False,
                                      transform=transforms.ToTensor())



        # Data loader.

        oneclass_loader=torch.utils.data.DataLoader(dataset=oneclass_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2,
                                                  drop_last=True, collate_fn = my_collate)



        return oneclass_loader