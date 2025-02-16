import torch 
import torch.nn as nn
import torch.optim.sgd as optim
from torch.autograd import Variable
x = input("what input to DOUBLE ")
input = float(x)

## Training data for 2X
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
 
## Model stuff
class LinearRegressionModel(nn.Module):
 
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 
our_model = LinearRegressionModel()
 
criterion = nn.MSELoss(size_average = False)
optimizer = optim.SGD(our_model.parameters(), lr = 0.01)
## 1000 iterations training
for epoch in range(1000):

    pred_y = our_model(x_data)
    ## loss calcualtion and auto weight
    loss = criterion(pred_y, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ## Show loss every 100 iterations
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, '%.50f' % loss.item()))

new_var = Variable(torch.Tensor([[input]]))
pred_y = our_model(new_var)
print("Prediction after training for", input, "is", our_model(new_var).item())