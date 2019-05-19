import numpy as np
import visdom

class VisdomLogger:
    def __init__(self, run_name):
        self.vis = visdom.Visdom(env=run_name)
        if not self.vis.check_connection():
            raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

        self.train_loss_window = self.create_plot_window('#Iterations', 'Loss', 'Training Loss')
        self.avg_loss_window = self.create_plot_window('#Iterations', 'Loss', 'Average Loss')
        self.avg_accuracy_window = self.create_plot_window('#Iterations', 'Accuracy', 'Average Accuracy')

    def create_plot_window(self, xlabel, ylabel, title):
        return self.vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

    def add_measure(self, window, x, y, name):
        if name not in ['train', 'validation', 'test']:
            print('Error invalid measure name')
        self.vis.line(X=np.array([x]), Y=np.array([y]), win=window, name=name, update='append')

    def add_acc_measure(self, x, y, name): 
        self.add_measure(self.avg_accuracy_window, x, y, name)

    def add_loss_measure(self, x, y, name):
        self.add_measure(self.avg_loss_window, x, y, name)

    def add_train_loss_measure(self, x, y):
        self.add_measure(self.train_loss_window, x, y, 'train')

