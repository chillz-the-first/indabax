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
        for images, labels in loader: # X = images, y = labels
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                print(f'Epoch {epoch:4d}, loss {loss.item():.4f}')

    return model