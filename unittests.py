from pypearl import ArrayD2, ArrayD1, ArrayI2, Layer, ReLU, Softmax, CCE, SGD, Model, copy_model, breed_models
import time

model = Model()

model.add_layer(4, 4)

model.add_relu()

model.add_layer(4, 2)

model.add_softmax()

x = ArrayD1(4)

x[0] = 4.0
x[1] = 3.0
x[2] = 2.0
x[3] = 1.0

y = model.forwardGA(x)

print("First Model: ", y)


model2 = Model()

model2.add_layer(4, 4)

model2.add_relu()

model2.add_layer(4, 2)

model2.add_softmax()

y2 = model2.forwardGA(x)

print("Second Model: ", y2)


modelClone = copy_model(model)

y3 = modelClone.forwardGA(x)

print("Copied First Model: ", y3)

model11 = breed_models(model, modelClone, 0.5)

y4 = model11.forwardGA(x)

print("Bred Clones: ", y4)

model12 = breed_models(model, model2, 0.5)

y5 = model12.forwardGA(x)

print("Bred 1,2: ", y5)

model.randomize(0.1)

y6 = model.forwardGA(x)

print("1 Randomized: ", y6)

y4 = model11.forwardGA(x)

print("Bred Clones: ", y4)

y3 = modelClone.forwardGA(x)

print("Copied First Model: ", y3)

'''
x = ArrayD2(3, 2)
for i in range(3):
    for j in range(2):
        x[i, j] = (i*j+3+j+i)/11

y = ArrayI2(3, 4)
y[0, 3] = 1
y[1, 1] = 1
y[2, 0] = 1

l1 = Layer(2, 4)

r = ReLU()

l2 = Layer(4, 4)

s = Softmax()

c = CCE()

o = SGD()

epochs = 1000000
t0 = time.perf_counter()
for i in range(epochs):
    v1 = l1.forward(x)
    v2 = r.forward(v1)
    v3 = l2.forward(v2)
    v4 = s.forward(v3)

    loss = c.forward(v4, y)
    #print("Loss")
    #print(loss)
    if i == 0:
        initloss = loss
    dval = c.backward(v4, y)
    dval2 = s.backward(dval)
    dval3 = l2.backward(dval2)
    dval4 = r.backward(dval3)
    l1.backward(dval4)
    o.optimize(l1)
    o.optimize(l2)
t1 = time.perf_counter()
elapsed = t1 - t0

# ---------------- results --------------------------------------------------------------
print("\ntraining finished")
print(f"initial loss : {initloss:.6f}")
print(f"final loss   : {loss:.6f}")
print(f"elapsed time : {elapsed:.2f} s "
      f"(≈ {elapsed/epochs*1e6:.1f} µs / epoch)")

#12.97s
#10.23s

'''