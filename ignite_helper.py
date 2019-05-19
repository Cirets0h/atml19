import torch

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor


def _prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (x.to(device, dtype=torch.float),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_supervised_rnn_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, hidden = model(x, None)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def create_supervised_rnn_evaluator(model, metrics={},
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, hidden = model(x, None)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
