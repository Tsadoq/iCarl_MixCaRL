import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image

from resnet import resnet32



class iCarlNet(nn.Module):
    def __init__(self, feature_size, n_classes,optimizer):
        self.num_classes =n_classes
        self.known_classes = 0
        self.exemplar_sets=[]
        self.flag_mean = True
        self.exemplar_means = []

        #We take a standard ResNet and Extend it
        super(iCarlNet,self).__init__()
        self.extractor = resnet32()
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features.feature_size)
        self.batch_normal = nn.BatchNorm1d(feature_size,momentum=0.01)
        self.relu = nn.ReLU()
        self.fully_connected = nn.Linear(feature_size,n_classes,bias=False)


        self.classication_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.BCELoss()
        self.optimizer = optimizer
        


    def forward(self, x):
        # X: input data
        x = self.extractor(x)
        x = self.batch_normal(x)
        x = self.relu(x)
        x = self.fully_connected(x)
        return x

        
    def increment_classes(self, n_classes_to_add):
        weight=self.fully_connected.weight.data
        feature_size=self.fully_connected.in_features
        old_num_classes=self.fully_connected.out_features
        self.fully_connected=nn.Linear(feature_size,old_num_classes+n_classes_to_add,bias=False)
        self.fully_connected.weight.data[:old_num_classes]=weight
        
        
    def classify(self, input_image_batch, transform):
        # input_image_batch: input batch of 10 classes

        batch_size = input_image_batch.size(0)
        
        if self.flag_mean:
            exemplar_means = []
            for exemplars in self.exemplar_sets:
                features = []
                for exemplar in exemplars:
                    feature = self.extractor(exemplar)
                    feature.data = feature.data / feature.data.norm()
                    features.append(feature)
                features = torch.stack(features)
                exemplar_mean = features.mean(0)
                exemplar_means.append(exemplar_mean)
            self.exemplar_means = exemplar_means
            self.flag_mean = False
        
        means = torch.stack(self.exemplar_means)
        
        feature = self.extractor(exemplar)
        for n in range(feature.size(0)):
            feature.data[n] = feature.data[n] / feature.data[n].norm()
        feature = feature.expand_as(mean)
        
        # Predict label by nearest exemplar mean
        distances = torch.sum(torch.pow((feature - mean), 2), dim = 1)
        _, predictions = torch.min(distances, 1)
        
        return predictions
                    

    def construct_exemplar_set(self, images, exemplars_per_class, transform):
        #MIO USCITE DA QUI MERDEEEEE
        example_features =  []
        exemplar_set = []
        exemplar_features = []
        for image in images:
            feature = self.extractor(image)
            feature.data = feature.data / feature.data.norm()
        example_features.append(feature)
        example_features = torch.stack(dim=0,example_features)
        class_mean = torch.mean(0,example_features) #should do the mean on the dimension of the append
        class_mean = class_mean / class_mean.norm()
        
        for i in range(exemplars_per_class):
            sum = np.sum(exemplar_features,axis=0)
            extracted_features = features
            mean = class_mean
            average_feature_vector = float(1)/(k+1)*(extracted_features + sum)
            average_feature_vector = average_feature_vector /average_feature_vector.norm()
            j = torch.argmin(torch.sqrt(np.sum(mean-average_feature_vector)**2,axis=1))
            exemplar_set.append(images[j])
            exemplar_features.append(feature[j])

        self.exemplar_sets.append(torch.stack(exemplar_set))
        
        
        

        
                

    def reduce_exemplar_sets(self, m):
        for i in range(len(self.exemplar_sets)):
            self.exemplar_sets[i]=self.exemplar_sets[i][:m]
            
 


    def combine_dataset_with_exemplars(self, dataset):
        

 


    def update_representation(self, dataset):
        self.compute_means=True
        classes=list(set(dataset.labels))
        new_classes=[cls for cls in classes if cls>self.num_claasses-1]
        self.increment_classes(len(new_classes))
        self.combine_dataset_with_exemplar(dataset)
        train_data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2)
        labels_old_net=torch.zeros(len(dataset), self.n_classes).cuda()
        for indices,images,labels in train_data_loader:
            images=images.to(device)
            indices=indices.to(device)
            g=
            
            
        

