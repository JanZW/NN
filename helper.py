import matplotlib.pyplot as plt
import os
from decay import StaticDecay, LinearDecay, ExponentialDecay
from neighborhood import gaussian, bubble

DEFAULTS_EXP = ('rgbg_beergardens', #dataset
                        2,          #number of neurons per city
                        5000,        #iterations
                        1000,         #k
                        1000,         #plot_k
                        bubble,     #neighborhood
                        ExponentialDecay(0.7, 0.9999),  #learning rate
                        ExponentialDecay(30, 0.999)) #radius
DEFAULTS_LIN = ('rgbg_beergardens', 8, 40000, 5000, 5000, gaussian,
                        LinearDecay(0.9, 0.0000089),
                        LinearDecay(20, 0.0005))

DEFAULT_STA = ('rgbg_beergardens', 2, 50000, 5000, 5000, gaussian,
                        StaticDecay(0.5),
                        StaticDecay(1))



plt.figure()


def plot_map(cities, neurons, iteration):
    """
    Generates the required map of cities and neurons at a given moment and
    stores the result in a png image. The map contains all the cities
    represented as red dots and all the neurons as green, crossed by a line
    dots. The method plots the iteration in which the snapshot was taken.
    :param cities: the cities to be plotted, passed as a list of (x, y)
    coordinates
    :param neurons: the cities to be plotted, passed as a list of (x, y)
    coordinates
    :param iteration: the iterations when the snapshot is taken
    :return: returns nothing
    """
    plt.scatter(*zip(*cities), color='red', s=20)
    plt.scatter(*zip(*neurons), color='green', s=2)

    plt.plot(*zip(*(neurons+[neurons[0]])), color='darkgreen')

    # Invert x axis to match representation at
    # http://www.math.uwaterloo.ca/tsp/world/countries.html
    plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal', adjustable='datalim')

    plt.title('Iteration #{:06d}'.format(iteration))
    plt.savefig('results/{}.png'.format(iteration))
    plt.clf()


def read_data(filename):
    """
    Reads and parses data from a txt file with a map data. The format that the
    function expects is the one followed in the uwaterloo TSP web archive
    (math.uwaterloo.ca/tsp/world/countries.html), ignoring the first
    description lines that have to be manually removed. The method searches for
    the filename in the assets folder.
    :param filename: the path to the file to be parsed
    :return: the cities as a list of (x, y) coordinates
    """
    cities = []

    path = 'rgbg_beergardens.txt'
    with open(path, 'r') as f:
        for line in f:
            city = list(map(float, line.split()[1:]))
            cities.append((city[1], city[0]))

    return cities


def get_input():
    """
    Gets the input from the user line or launches the default values
    :return data_set: list of cities as (x,y) coordinates
    :return n_neurons: number of neurons per city in the data_set
    :return iterations: number of iterations to be executed
    :return learning_rate: learning rate to be used
    :return radius: radius of neurons to be used
    """
    data_set='rgbg_beergardens'

    
    use_defaults = input('Do you want to use default parameters? (y/n) ') == 'y'

    if use_defaults:
        decay = input('What kind of decay? [s/l/e]') or 'e'
        if decay == 'e':
            return DEFAULTS_EXP
        if decay == 'l':
            return DEFAULTS_LIN
        if decay == 's':
            return DEFAULT_STA
    # Comprehensive input
    n_neurons = int(input('How many neurons per city? (8)') or 8)
    iterations = int(input('How many iterations (500)?') or 500)
    learning_rate = get_input_decay('learning rate')
    radius = get_input_decay('radius')
    k = int(input('After how many iterations print current TSP distance? (1000)') or 1000)
    plot_k = int(input('After how many iterations generate plot? (200)') or 200)

    neighborhood = input('Choose neighborhood function: [g/b]') or 'b'
    if neighborhood == 'b':
        neighborhood = bubble
    elif neighborhood == 'g':
        neighborhood = gaussian
    else:
        exit('Not a valid neighborhood!')

    return data_set, n_neurons, iterations, k, plot_k, neighborhood, learning_rate, radius

def get_input_decay(name):
    """
    Generates the appropriate decay function for a given variable. The decays
    that can be generated are static (no decay over time), linear and
    exponential.
    :param name: name of the variable that will be generated
    :return: decay function for the variable with appropriate parameters
    """
    decay = input('What kind of decay for {}? [s/l/e]'.format(name)) or 'e'
    if decay == 's':
        value = float(input('Static value: '))
        return StaticDecay(value)
    if decay == 'l':
        value = float(input('Start value: '))
        rate = float(input('Rate:'))
        return LinearDecay(value, rate)
    if decay == 'e':
        value = float(input('Start value: '))
        rate = float(input('Rate:'))
        return ExponentialDecay(value, rate)
    exit('Not a valid option!')
