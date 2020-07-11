import os
import logging
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

"""
    Call back function for pytorch-lighting to store results during training.
    
"""


class CallbackLogger(pl.Callback):

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.output_file = "test_results.txt"

    def on_validation_end(self, trainer, pl_module):
        logger.info(" ---- Validation Ended and results are ----- ")

        if pl_module.is_logger():
            metrics = trainer.Callback_metrics

            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info(" ------ Testing Done and results are ----- ")

        if pl_module.is_logger():
            metrics = trainer.Callback_metrics

        output_test_results_file = os.path.join(
            self.output_dir, self.output_file)

        with open(output_test_results_file, "w") as test_fp:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    test_fp.write("{} = {}\n".format(key, str(metrics[key])))
