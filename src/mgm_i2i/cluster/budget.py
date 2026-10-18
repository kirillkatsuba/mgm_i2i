"""Resume-aware stopping by sampled patches or legacy training wall time."""


class TrainingBudget:
    def __init__(self, config, dataset_size):
        self.seconds = config['seconds']
        self.epochs = config.get('epochs')
        self.epoch_size = config.get('samples_per_epoch', dataset_size)
        self.target_samples = None
        if self.epochs is not None:
            if not isinstance(self.epochs, int) or self.epochs < 1:
                raise ValueError('epochs must be a positive integer')
            if self.epoch_size != dataset_size or dataset_size < 1:
                raise ValueError('samples_per_epoch must match the training image count times crops per image')
            self.target_samples = self.epochs * self.epoch_size
            batch = config['batch_size'] * config.get('grad_accumulation', 1)
            if self.target_samples % batch:
                raise ValueError('Epoch target must be divisible by effective batch size; partial batches are unsupported')

    def reached(self, consumed, elapsed):
        if self.target_samples is not None:
            return consumed >= self.target_samples
        return elapsed >= self.seconds

    def metadata(self, consumed):
        return dict(training_stop='epochs' if self.epochs is not None else 'time',
                    target_epochs=self.epochs, target_samples=self.target_samples,
                    samples_per_epoch=self.epoch_size,
                    epochs_completed=consumed / self.epoch_size)
