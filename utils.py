from imgaug import augmenters as iaa
import keras
from keras.callbacks import Callback
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

class SaveWeights(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.ep = 0

    def on_epoch_end(self, epoch, logs={}):
        self.i += 1
        self.ep = epoch + 1
        # print("Self I: ", self.i, " Epoch: ", self.ep)
        if (self.i % 10) == 0:
            model.save_weights(f"dmth_results/weights/Assignment5_RESNET__{self.ep}.hdf5")
            print("Saved the Model after Epoch: ", self.i)

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []

        self.val_loss = []

        self.gender_acc = []
        self.val_gender_acc = []

        self.image_acc = []
        self.val_image_acc = []

        self.age_acc = []
        self.val_age_acc = []

        self.weight_acc = []
        self.val_weight_acc = []

        self.bag_acc = []
        self.val_bag_acc = []

        self.footwear_acc = []
        self.val_footwear_acc = []

        self.pose_acc = []
        self.val_pose_acc = []

        self.emotion_acc = []
        self.val_emotion_acc = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)

        self.val_loss.append(logs.get('val_loss'))

        self.gender_acc.append(logs.get('gender_output_acc'))
        self.val_gender_acc.append(logs.get('val_gender_output_acc'))

        self.image_acc.append(logs.get('image_quality_output_acc'))
        self.val_image_acc.append(logs.get('val_image_quality_output_acc'))

        self.age_acc.append(logs.get('age_output_acc'))
        self.val_age_acc.append(logs.get('val_age_output_acc'))

        self.weight_acc.append(logs.get('weight_output_acc'))
        self.val_weight_acc.append(logs.get('val_weight_output_acc'))

        self.bag_acc.append(logs.get('bag_output_acc'))
        self.val_bag_acc.append(logs.get('val_bag_output_acc'))

        self.footwear_acc.append(logs.get('footwear_output_acc'))
        self.val_footwear_acc.append(logs.get('val_footwear_output_acc'))

        self.pose_acc.append(logs.get('pose_output_acc'))
        self.val_pose_acc.append(logs.get('val_pose_output_acc'))

        self.emotion_acc.append(logs.get('emotion_output_acc'))
        self.val_emotion_acc.append(logs.get('val_emotion_output_acc'))

        self.i += 1

        if (self.i % 10) == 0:
            # print("===========Here after 10=========")

            f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=1, ncols=9, figsize=(30, 4),
                                                                            sharex=True)

            ax1.plot(self.x, self.val_loss, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.gender_acc, label="gender_accuracy")
            ax2.plot(self.x, self.val_gender_acc, label="gender_val_accuracy")
            ax2.legend()

            ax3.plot(self.x, self.image_acc, label="image_accuracy")
            ax3.plot(self.x, self.val_image_acc, label="image_val_accuracy")
            ax3.legend()

            ax4.plot(self.x, self.age_acc, label="age_accuracy")
            ax4.plot(self.x, self.val_age_acc, label="age_val_accuracy")
            ax4.legend()

            ax5.plot(self.x, self.weight_acc, label="weight_accuracy")
            ax5.plot(self.x, self.val_weight_acc, label="weight_val_accuracy")
            ax5.legend()

            ax6.plot(self.x, self.bag_acc, label="bag_accuracy")
            ax6.plot(self.x, self.val_bag_acc, label="bag_val_accuracy")
            ax6.legend()

            ax7.plot(self.x, self.footwear_acc, label="footwear_accuracy")
            ax7.plot(self.x, self.val_footwear_acc, label="footwear_val_accuracy")
            ax7.legend()

            ax8.plot(self.x, self.emotion_acc, label="emotion_accuracy")
            ax8.plot(self.x, self.val_emotion_acc, label="emotion_val_accuracy")
            ax8.legend()

            ax9.plot(self.x, self.pose_acc, label="pose_accuracy")
            ax9.plot(self.x, self.val_pose_acc, label="pose_val_accuracy")
            ax9.legend()

            plt.show();



# Keras-Contib Implementation
class CyclicLR(Callback):

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        print("Learning Rate: ", float(K.get_value(self.model.optimizer.lr)))
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def additional_augmenation(image):
    aug1 = iaa.CoarseDropout(p=0.10, size_percent=0.05)
    image = aug1.augment_image(image)
    return image


def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1

    print('Learning Rate: ', lr)
    return lr
