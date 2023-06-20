import torch.optim as optim
opt = optim.SGD(mynet.parameters(), lr=0.03)
print(opt)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
	for X, y in data_iter:
		output = mynet(X)	# 获得前向传播结果
		l = loss(..., ...)	# 计算损失值

		# 以下三步计算梯度并使用SGD算法进行梯度下降
  
        # 梯度归零。Pytorch不会自动进行梯度归零，如果不手动清零，则下一次迭代时计算得到的梯度值会与之前留下的梯度值进行叠加   
		opt.zero_grad() 
        # 计算梯度
		l.backward()
        # 执行梯度下降，对学习的参数进行一次更新
		opt.step()
