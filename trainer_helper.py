import datetime
import adabound
import os
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.handlers import EarlyStopping, ModelCheckpoint

from visdom_helper import VisdomLogger
from helper import plot_train_history
from ignite_helper import create_supervised_rnn_trainer, create_supervised_rnn_evaluator

def score_function(engine):
    metrics = engine.state.metrics
    try:
        return 100.0*metrics['accuracy']
    except:
        print("Warning engine state doesn't contain accuracy")
        return 0.0

class Trainer_Helper:
    def __init__(self, run_name, device, log_interval=50, scheduler=None, use_visdom=True, use_checkpoints=True):
        self.run_name = run_name
        self.log_interval = log_interval
        self.device = device
        self.epochs_trained = 0
        self.scheduler = scheduler

        self.accuracy_train_hist = []
        self.loss_train_hist = []
        self.accuracy_val_hist = []
        self.loss_val_hist = []

        self.vis_logger = None
        self.use_visdom = use_visdom

        if use_visdom:
            self.vis_logger = VisdomLogger(run_name)
        
        self.checkpointer = None
        if use_checkpoints:
            self.checkpointer = ModelCheckpoint(
                'models_checkpoints', 
                run_name,
                score_name='val_accuracy',
                n_saved=3, 
                score_function=score_function,
                save_as_state_dict=True,
                atomic=True
            )
    def setup_dataloader(self, train_dataloader, val_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
    
    def add_scheduler(self, scheduler_handler):
        self.scheduler = scheduler_handler

    def train_rnn(self, model, optimizer, criterion, epochs, print_stdout = True, log_interval = 50):
        self.trainer = create_supervised_rnn_trainer(model, optimizer, criterion, device=self.device)
        self.train_evaluator = create_supervised_rnn_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=self.device)
        self.val_evaluator = create_supervised_rnn_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=self.device) 
        self._train(model, optimizer, criterion, epochs, print_stdout = print_stdout, log_interval = log_interval)
    
    def train(self, model, optimizer, criterion, epochs, print_stdout = True, log_interval = 50):
        self.trainer = create_supervised_trainer(model, optimizer, criterion, device=self.device)
        self.train_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=self.device)
        self.val_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=self.device)        
        self._train(model, optimizer, criterion, epochs, print_stdout = print_stdout, log_interval = log_interval)

    def _train(self, model, optimizer, criterion, epochs, print_stdout, log_interval):
        trainer = self.trainer
        train_evaluator = self.train_evaluator
        val_evaluator = self.val_evaluator

        if self.checkpointer is not None:
            val_evaluator.add_event_handler(Events.COMPLETED, self.checkpointer, {'model': model})
            
        if self.scheduler is not None:
            trainer.add_event_handler(Events.ITERATION_COMPLETED, self.scheduler)

        use_visdom = self.use_visdom
        vis_logger = self.vis_logger

        self.epochs_trained += epochs

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train_dataloader) + 1
            if use_visdom and iter % log_interval == 0:
                vis_logger.add_train_loss_measure(engine.state.iteration, engine.state.output)


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(self.train_dataloader)
            metrics = train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.accuracy_train_hist.append(avg_accuracy)
            self.loss_train_hist.append(avg_loss)

            if print_stdout:
                print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                      .format(engine.state.epoch, avg_accuracy, avg_loss))

            if use_visdom:
                vis_logger.add_acc_measure(engine.state.epoch, avg_accuracy, 'train')
                vis_logger.add_loss_measure(engine.state.epoch, avg_loss, 'train')


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            val_evaluator.run(self.val_dataloader)
            metrics = val_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.accuracy_val_hist.append(avg_accuracy)
            self.loss_val_hist.append(avg_loss)

            if print_stdout:
                print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                      .format(engine.state.epoch, avg_accuracy, avg_loss))

            if use_visdom:
                vis_logger.add_acc_measure(engine.state.epoch, avg_accuracy, 'validation')
                vis_logger.add_loss_measure(engine.state.epoch, avg_loss, 'validation')

        @trainer.on(Events.COMPLETED)
        def plot_end_result(engine):
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plot_train_history(self.epochs_trained, 'plots/' + self.run_name, self.loss_train_hist, self.loss_val_hist, self.accuracy_train_hist, self.accuracy_val_hist)

        trainer.run(self.train_dataloader, max_epochs=epochs)
