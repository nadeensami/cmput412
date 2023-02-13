import matplotlib.pyplot as plt

file = open("odometry-4.txt", "r")

x = []
y = []
theta = []

for line in file:
  remove_parenthesis = line[1:len(line)-2]
  data = remove_parenthesis.split(", ")
  x.append(float(data[0]))
  y.append(float(data[1]))
  theta.append(float(data[2]))

plt.plot(x,y)
plt.show()