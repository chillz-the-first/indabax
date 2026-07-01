def train(model, X, y, loss_fn, optimizer, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    return model

def train_cnn(model, loader, loss_fn, optimizer, epochs=1000):
    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(loader): # batch = the batch index
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            if batch % 200 == 0:
                print(f'Epoch {epoch:4d}, loss {loss.item():.4f}')

    return model