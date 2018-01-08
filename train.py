import datetime
import glob
import sys
from fann2 import libfann

if __name__ == '__main__':
    if len(sys.argv) < 2:
        files = []
        for file in glob.glob("data/*.data"):
            files.append(file)
        files.sort(reverse=True)
        filename = files[0]
    else:
        filename = sys.argv[1]

    learning_rate = 0.1
    desired_error = 0.001
    max_iterations = 60
    iterations_between_reports = 1
    layers = [784, 260, 10]

    ann = libfann.neural_net()
    ann.create_standard_array(layers)
    ann.set_learning_rate(learning_rate)
    ann.set_learning_rate(learning_rate)
    ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
    print('Training on data: ' + filename)
    ann.train_on_file(filename, max_iterations, iterations_between_reports, desired_error)

    filename = 'net/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.net'
    print('Saving net to: ' + filename)
    ann.save(filename)
