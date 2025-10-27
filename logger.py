import os
import logging
import datetime

def get_local_time():
    return datetime.datetime.now().strftime("(%Y-%m-%d_%H'%M'%S)")


class Logger(object):
    def __init__(self, log_configs=True, args=None):
        self.args = args
        log_dir_path = './log/{}'.format(args.data)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        dataset_name = args.data
        self.log_file_handler = logging.FileHandler("{}/{}_{}_{}.log".format(
                log_dir_path, args.modelName, dataset_name, get_local_time()), mode="a", encoding="utf-8")
        self.formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s \n%(message)s")
        self.formatter_train_eval_test = logging.Formatter("%(message)s")
        self.log_file_handler.setFormatter(self.formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_file_handler)
        if log_configs:
            self.log(args)

    def log(self, message, save_to_file=True, print_to_console=True):
        self.log_file_handler.setFormatter(self.formatter)
        if save_to_file:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_train(self, message, save_to_file=True, print_to_console=True):
        self.log_file_handler.setFormatter(self.formatter_train_eval_test)
        if save_to_file:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        epoch = self.args.epoch
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += "{}: {:.4f} ".format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = "-"*30 + "Epoch {:<3d} ".format(epoch_idx) + data_type + "-"*30
        else:
            message = ""
        message_metric = ""
        message_value = ""
        metric_HR, metric_NDCG = eval_result.keys()
        # for metric in eval_result:
        for i in range(len(k)):
            message_metric += "{:<10s}".format(metric_HR+"@" + str(k[i])) + "{:<10s}".format(metric_NDCG+"@" + str(k[i]))
            message_value += "{:<10.6f}".format(eval_result[metric_HR][i]) + "{:<10.6f}".format(eval_result[metric_NDCG][i])
        message += "\n" + message_metric + "\n" + message_value
        self.log(message, save_to_log, print_to_console)



