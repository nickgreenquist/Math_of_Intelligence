from numpy import *
import matplotlib.pyplot as plt

def computer_error(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return float(totalError) / len(points)

def step_gradient(b, m, points, a):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += -(2/N) * x * (y - ((m * x) + b))

    b = b - (a * b_gradient)
    m = m - (a * m_gradient)
    return [b, m]

def plot_current(plt, points, m, b):
    plt.clf()
    for j in range(len(points)):
        x = points[j, 0]
        y = points[j, 1]

        plt.scatter(x, y)
    x0 = 0
    y0 = (m * x0) + b
    x1 = 70
    y1 = (m * x1) + b

    plt.plot([x0, x1], [y0, y1], 'k-', lw=2)
    plt.pause(0.005)


def gradient_descent_runner(points, b, m, a, num_iterations, convergence):  
    
    # Real time update plot
    plt.axis([0, 70, 0, 120])
    plt.ion()

    last_b = b
    last_m = m
    for i in range(num_iterations):

        # Get new slope and y-intercept
        b,m = step_gradient(b, m, array(points), a)

        # Plot the updated line
        plot_current(plt, points, m, b)

        # Break if convergence
        m_diff = abs(last_m - m)
        b_diff = abs(last_b - b)
        if m_diff <= convergence and b_diff <= convergence:
            print("Converged at %s iterations" % (i))
            break

        # update last m and b
        last_b = b
        last_m = m

    return [b,m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    convergence = 0.001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000

    initial_error = computer_error(initial_b, initial_m, points)
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, initial_error))
    
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, convergence)

    final_error = computer_error(b, m, points)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, final_error))

if __name__ == '__main__':
    run()