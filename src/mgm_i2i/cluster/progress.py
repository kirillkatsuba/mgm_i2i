"""Time-budget progress with resume-aware clocks and readable Slurm log lines."""
import math
import sys

from tqdm import tqdm


class LineStream:
    """Render each tqdm refresh as a line when stdout is redirected by Slurm."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        line = text.replace('\r', '').rstrip()
        if line:
            self.stream.write(line + '\n')
        return len(text)

    def flush(self):
        self.stream.flush()


def clock(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class BudgetProgress:
    def __init__(self, total, description, elapsed=0., step=0, stream=None):
        stream = sys.stdout if stream is None else stream
        interactive = stream.isatty()
        self.total = float(total)
        self.losses = {}
        self.closed = False
        self.bar = tqdm(total=self.total, initial=min(max(0., elapsed), self.total),
                        desc=description, unit='s', ascii=True, disable=False,
                        file=stream if interactive else LineStream(stream),
                        mininterval=1. if interactive else 30., miniters=0,
                        dynamic_ncols=interactive, leave=True,
                        postfix=self._postfix(elapsed, step),
                        bar_format='{desc} {percentage:6.2f}%|{bar:20}|{postfix}')

    def _postfix(self, elapsed, step):
        remaining = math.ceil(max(0., self.total-elapsed))
        loss_text = ' '.join(f'{key}={value:.5g}' for key, value in self.losses.items()
                             if value is not None)
        return (f'elapsed={clock(elapsed)}/{clock(self.total)} '
                f'remaining={clock(remaining)} step={step} {loss_text}').rstrip()

    def update(self, elapsed, step, losses=None):
        if self.closed:
            return
        if losses is not None:
            self.losses = losses
        self.bar.set_postfix_str(self._postfix(elapsed, step), refresh=False)
        self.bar.update(max(0., min(elapsed, self.total)-self.bar.n))

    def close(self):
        if not self.closed:
            self.bar.close()
            self.closed = True
