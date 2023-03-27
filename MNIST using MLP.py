#!/usr/bin/env python
# coding: utf-8

# In[164]:


import matplotlib.pyplot as plt
import torch
import torchvision


# In[165]:


#hyper-parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 20
LR = 0.01


# In[166]:


torch.cuda.is_available(),torch.cuda.get_device_name(),torch.cuda.device_count()


# In[167]:


train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, 
                           transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
print(len(train_loader)) #the size of dataset, 60000 of images


# In[168]:


#load test data
test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, 
                           transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
print(len(test_loader)) #size of test data , 10000 images


# In[169]:


#have a look on data

examples = enumerate(test_loader)
id, (image,label) = next(examples)
#examples is a batch of images, 
#when BATCH_SIZE == 1:
#example[0] is a order signed by enumerate(), 
#example[1][0] is the tensor of image,
#example[1][1] is the label of the image 0~9

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(image[0][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(label[0]))
  plt.xticks([])
  plt.yticks([])
  id, (image,label) = next(examples)


# In[170]:


# model
model = torch.nn.Sequential(
    
    torch.nn.Linear(784,800),
    torch.nn.ReLU(),
    torch.nn.Linear(800,50),
    torch.nn.ReLU(),
    torch.nn.Linear(50,10),
    torch.nn.Softmax(),
)


# In[174]:


# loss function & optimizater
loss_fn = torch.nn.NLLLoss() #nll_loss for classification
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# In[ ]:


# Accuracy check round 1(before training)

counter = 0 # count the number of right guess

for image,label in test_loader:
    input = image[0][0].reshape(1,784)
    output = model(input)
    pred_label = int(torch.argmax(output))
    label = int(label)
    if pred_label == label:
        counter += 1
accuracy = counter/len(test_loader)

accuracy
        


# In[175]:


#Trainning process
for image,label in train_loader:
    
    optimizer.zero_grad()
    
    input = image[0][0].reshape(1,784)
    output = model(input)
    loss = loss_fn(output,label)
    
    loss.backward()
    optimizer.step()
    
    print('loss {}'.format(loss.item()))
    
print("finish!!")


# In[176]:


# Accuracy check round 2(after training)

counter = 0 # count the number of right guess

for image,label in test_loader:
    input = image[0][0].reshape(1,784)
    output = model(input)
    pred_label = int(torch.argmax(output))
    label = int(label)
    if pred_label == label:
        counter += 1
accuracy = counter/len(test_loader)

accuracy
        


# In[177]:


# test & presentation
result = enumerate(test_loader)
id, (image,label) = next(result)

fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
    
  input = image.reshape(1,784)
  output = model(input)
  pred_label = int(torch.argmax(output))
    
  plt.imshow(image[0][0], cmap='gray', interpolation='none')
  plt.title("Truth:{},pred:{}".format(label[0],pred_label))
  
  plt.xticks([])
  plt.yticks([])
  id, (image,label) = next(result)


# In[ ]:




