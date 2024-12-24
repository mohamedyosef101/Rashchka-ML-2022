from tools import split_iris
from models import Perceptron
import matplotlib.pyplot as plt 

X, y = split_iris('setosa', 'versicolor', [0, 2])
ppn = Perceptron(eta=0.1, epochs=10, store_results=True)
ppn.fit(X, y)

def plot(lst, label):
  plt.plot(range(1, len(lst) + 1),
         lst, marker='o', label=label)

plot(ppn.errors, "Errors")
# plt.title("Errors over epochs")
# plt.xlabel('Epochs')
# plt.show()

for i in range(len(ppn.weights[0])):
  plot([w[i] for w in ppn.weights], f"w{i+1}")
plot(ppn.biases, "biases")
plt.xlabel('Epochs')
plt.legend()
plt.show()