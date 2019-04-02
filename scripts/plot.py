import sys
import json
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def load_results(filename):
    with open(filename, 'r') as file:
        results = json.loads(file.read())
        return results


def load_baseline(filename):
    with open(filename, 'r') as file:
        baselines = {}
        for line in file:
            values = line.split(',')
            epochs = values[0]
            avg = values[1]
            stddev = values[2]
            baselines[epochs] = (avg, stddev)
        return baselines


def loss_fn(training, uptime, epochs):
    x = []
    y = []

    i = 0
    for step in range(epochs):
        time = (step + 1) * (uptime / epochs)
        x.append(time)
        if i < len(training):
            step_, reward_ = training[i]
            if step_ == step:
                reward = reward_
                i += 1
        y.append(reward)

    return x, y


def avg_fn(trainings, uptime, epochs):
    ys = []
    for training in trainings:
        if len(training) < 10:
            continue
        x, y = loss_fn(training, uptime, epochs)
        ys.append(np.array(y))
    ys = np.stack(ys)
    ys_mean = np.mean(ys, axis=0)
    ys_stddev = np.std(ys, axis=0)
    return ys_mean, ys_stddev


def time_schedule(uptime, epochs):
    xs = [(step + 1) * (uptime / epochs) for step in range(epochs)]
    return xs


def plot_result_curve(model, xs, ys_mean, ys_stddev, epochs, learning_rate, batch_size):
    plt.plot(xs, -ys_mean, label='tf-mdp ({})'.format(model))
    plt.fill_between(xs, -(ys_mean - ys_stddev), -(ys_mean + ys_stddev),
        alpha=0.2)
    plt.title('Training (epochs = {}, {}, {})'.format(epochs, batch_size, learning_rate),
        fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)',
        fontsize=14)
    plt.ylabel('Average Total Cost',
        fontsize=14)
    plt.draw()


def plot_results(results, epochs):
    for model, result in results.items():
        layers, batch_size, learning_rate = model.split('_')
        uptime = result['time']
        trainings = result['trainings']
        mean, stddev = avg_fn(trainings, uptime, epochs)
        xs = time_schedule(uptime, epochs)
        plot_result_curve(layers, xs, mean, stddev, epochs, learning_rate, batch_size)


def plot_baselines(xs, baselines):
    for epochs, (avg, stddev) in baselines.items():
        avg = -np.array([float(avg)] * len(xs))
        stddev = np.array([float(stddev)] * len(xs))
        plt.plot(xs, avg, '--', label='tf-plan (epochs/step={})'.format(epochs))
        # plt.fill_between(xs, avg - stddev, avg + stddev,
        #     alpha=0.2)


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    output = sys.argv[3]

    results = load_results(filename1)
    baselines = load_baseline(filename2)

    epochs = 200

    f = plt.figure(figsize=(7, 3))

    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.useafm'] = True
    # matplotlib.rcParams['pdf.use14corefonts'] = True
    # matplotlib.rcParams['text.usetex'] = True

    plot_results(results, epochs)

    max_uptime = max(result['time'] for result in results.values())
    xs = time_schedule(max_uptime, epochs)
    plot_baselines(xs, baselines)

    plt.legend(loc='upper right', fontsize='x-large')
    plt.grid()
    # plt.show()
    
    f.savefig('plot-{}.pdf'.format(output), bbox_inches='tight')
