from alne import *
import matplotlib.pyplot as plt


alone = Alone(n_iter=15, e=0.01, random_state=1)
alone.fit(x_std, y)

plot_decision_regions(x_std, y, classifier=alone)
plt.title('Adaline - Spadek stochastyczny wzdluz gradientu')
plt.xlabel('Dlugosc dzialki')
plt.ylabel('Dlugosc platka')
plt.legend(loc='upper right')
plt.show()
plt.plot(range(1, len(alone.cost_) + 1), alone.cost_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Sredni koszt')
plt.show()
