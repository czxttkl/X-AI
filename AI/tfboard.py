"""
A convinient writer to Tensorboard
"""
import tensorflow as tf


class TensorboardWriter:

    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.summary_writer = tf.summary.FileWriter(self.folder_name)

    def write(self, tags, values, step):
        """
        write tags and values for the step
        :param tags: a list of tags
        :param values: a list of values
        :param step: the training step to be associated with
        """
        summary = tf.Summary()
        for tag, val in zip(tags, values):
            summary.value.add(tag=tag, simple_value=val)
        self.summary_writer.add_summary(summary, global_step=step)
        self.summary_writer.flush()
