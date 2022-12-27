from hyperparameters import *



class CNN_Net(nn.Module):
  def __init__(self):
    self.loss_graphT = []
    super(CNN_Net, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.Conv2d(in_channels=32, out_channels=100, kernel_size=8, stride=1, padding=0),
      nn.LeakyReLU(negative_slope=0.01, inplace=False)
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels = 100, out_channels = 32, kernel_size=8, stride=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.01, inplace=False),
      nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid()
    )


  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def Train_net(self, TrainData, learning_rate, epoch_number):
    self.train()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    EPOCHS = epoch_number
    for epoch in range(EPOCHS):
      for i, (inputs) in enumerate(TrainData):
        batch_X = inputs
        optimizer.zero_grad()
        output = self.forward(batch_X)
        loss = loss_function(output, batch_X)
        self.loss_graphT.append(loss.item())
        loss.backward()
        optimizer.step()
        print("epoch: ", epoch, " loss for batch: ", loss)

  def Test_single(self, number, Current_DataSet):
    self.eval()
    with torch.no_grad():
      result =  self.forward(Current_DataSet[number:number+1])
    plt.imshow(result.view(128, 128, 3).cpu().detach().numpy())
    plt.show()
    plt.imshow(Current_DataSet[number].view(128, 128, 3).cpu().detach().numpy())
    plt.show()
    return result

  def plot_res(self):
    plt.plot(np.arange(1,len(self.loss_graphT)+1), self.loss_graphT)
    plt.ylabel('loss of Train')
    plt.xlabel("number steps")
    plt.grid(True)
    plt.show()