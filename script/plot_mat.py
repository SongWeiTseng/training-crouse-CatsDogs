import matplotlib.pyplot as plt


def plot(path):
    acc_file = ['train_acc.txt', 'val_acc.txt']
    loss_file = ['train_loss.txt', 'val_loss.txt']

    plt.figure()
    acc_list = [[], []]
    for i, data in enumerate(acc_file):
        with open(path + acc_file[i], 'r') as f:
            acc_list[i] = f.readlines()

    plot_list = [[], []]
    for train_acc, val_acc in zip(acc_list[0], acc_list[1]):
        plot_list[0].append(float(train_acc.split(' ')[1]))
        plot_list[1].append(float(val_acc.split(' ')[1]))
    plt.plot(plot_list[0], label='train acc')
    plt.plot(plot_list[1], label='val acc')
    plt.legend()
    plt.title("accuracy curve")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig(path + "accuracy.png", dpi=300)
    plt.show()

    loss_list = [[], []]
    for i, data in enumerate(loss_file):
        with open(path + loss_file[i], 'r') as f:
            loss_list[i] = f.readlines()

    plt.figure()
    plot_list = [[], []]
    for train_loss, val_loss in zip(loss_list[0], loss_list[1]):
        plot_list[0].append(float(train_loss.split(' ')[1]))
        plot_list[1].append(float(val_loss.split(' ')[1]))
    plt.plot(plot_list[0], label='train loss')
    plt.plot(plot_list[1], label='val loss')
    plt.legend()
    plt.title("loss curve")
    plt.xlabel("epochs")
    plt.ylabel("BCE Loss")
    plt.savefig(path + "Loss.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    plot('../log/Adam-nofreeze/')
