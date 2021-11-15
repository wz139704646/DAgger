import logging


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()

    def logs(self, s, step, tag='log'):
        self.logger.info("{}/Step {}: {}".format(tag, step, s))

    def logkv(self, d, step, tag='log'):
        for k,v in d.items():
            self.logger.info("{}/Step {} {}: {}".format(tag, step, k, v))


class TensorboardLogger(Logger):
    def __init__(self, writer):
        super().__init__()

        self.writer = writer

    def logs(self, s, step, tag='log'):
        self.writer.add_text(tag, s, global_step=step)

    def logkv(self, d, step, tag='log'):
        for k,v in d.items():
            self.writer.add_scalar(tag+"/"+k, v, global_step=step)

    def logm(self, m, m_in):
        """log model
        :param m: model (nn)
        :param m_in: input to model
        """
        self.writer.add_graph(m, m_in)
