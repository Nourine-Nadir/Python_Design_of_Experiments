from models import FC_net
from utils import *
from sklearn.model_selection import  train_test_split
import numpy as np
import torch as T


engine = Engine()
df = engine.read_sheet('Step1.xlsx',)
data = np.array(df)

features = np.array(data[:,1:-2]).astype(np.float32)
outputs = np.array(data[:,-2:-1]).astype(np.float32)

train_features, test_features, train_outputs, test_outputs = train_test_split(
    features, outputs, test_size=0.2, random_state=23
)

data_dict = {
    'features' : features,
    'outputs' : outputs,
    'train_features' : train_features,
    'train_outputs' : train_outputs,
    'test_features' : test_features,
    'test_outputs' : test_outputs,

}

print(f'features {features.shape}')
print(f'output {outputs.shape}')

NN_model = FC_net(lr=0.001,
                  input_shape=3,
                  fc1_dims=32,
                  fc2_dims=16,
                  n_output=outputs.shape[-1])

features = T.tensor(np.array(train_features), device=NN_model.device)
labels = T.tensor(np.array(train_outputs), device=NN_model.device)

for i in range(10000):
    predictions = NN_model(features)

    loss = NN_model.loss(predictions, labels)
    # Backward pass
    NN_model.optimizer.zero_grad()
    loss.backward()
    NN_model.optimizer.step()

    print(loss)

predictions = NN_model(features)

print(f'predictions : {predictions} \noutputs : {outputs}')