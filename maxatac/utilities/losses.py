import pandas as pd

from maxatac.utilities.system_tools import Mute
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

with Mute():
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        Lambda,
        BatchNormalization,
        Dense,
        Flatten
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError

    from maxatac.utilities.constants import KERNEL_INITIALIZER, INPUT_LENGTH, INPUT_CHANNELS, INPUT_FILTERS, \
        INPUT_KERNEL_SIZE, INPUT_ACTIVATION, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, FILTERS_SCALING_FACTOR, DILATION_RATE, \
        OUTPUT_LENGTH, CONV_BLOCKS, PADDING, POOL_SIZE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
        DEFAULT_ADAM_DECAY


# maxATAC v1 Loss Function
'''def maxatac_loss_function(
        y_true,
        y_pred,
        y_pred_min=0.0000001,  # 1e-7
        y_pred_max=0.9999999,  # 1 - 1e-7
        y_true_min=-0.5
):
    y_true = K.flatten(y_true)
    y_pred = tf.clip_by_value(
        K.flatten(y_pred),
        y_pred_min,
        y_pred_max
    )
    losses = tf.boolean_mask(
        tensor=-y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred),
        mask=K.greater_equal(y_true, y_true_min)
    )
    return tf.reduce_mean(input_tensor=losses)'''

class cross_entropy(tf.keras.losses.Loss):
    """
    Cross Entropy Loss function is used in maxATAC v1
    """
    def __init__(self, name="cross_entropy", **kwargs):
        super().__init__(name=name)
        #self.y_true = y_true
        #self.y_pred = y_pred
        self.y_pred_min = 0.0000001,  # 1e-7
        self.y_pred_max = 0.9999999,  # 1 - 1e-7
        self.y_true_min = -0.5

    def call(self, y_true, y_pred):
        """
        Calculate the loss according to maxATAC paper
        """

        yt = K.flatten(y_true)
        yp = tf.clip_by_value(
            K.flatten(y_pred),
            self.y_pred_min,
            self.y_pred_max
        )
        losses = tf.boolean_mask(
            tensor=-yt * K.log(yp) - (1 - yt) * K.log(1 - yp),
            mask=K.greater_equal(yt, self.y_true_min)
        )
        return tf.reduce_mean(input_tensor=losses)



'''
mse = tf.keras.losses.MeanSquaredError(reduction="auto",
name="mean_squared_error")  
# May want to change Reduction methods possibly
'''

class pearsonr_mse(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_mse", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.001 #best
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        mse_loss = mse()
        mse_raw = mse_loss(y_true, y_pred)
        #sum with weight
        total_loss = pr_loss(y_true, y_pred) + self.alpha*mse_raw
        return total_loss

class pearsonr_poisson(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_poisson", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.1 ###TODO: SET TO 0.001
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        pr = pr_loss(y_true, y_pred)
        #poisson part
        poiss_loss = poisson()
        poiss = poiss_loss(y_true, y_pred)
        #sum with weight
        total_loss = (2*pr*poiss)/(pr+poiss)
        return total_loss

class poisson(tf.keras.losses.Loss):
    def __init__(self, name="poisson", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.keras.losses.poisson(y_true, y_pred)

class mse(tf.keras.losses.Loss):
    def __init__(self, name="mse", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        print("value: ", tf.keras.losses.MSE(y_true,y_pred))
        return tf.keras.losses.MSE(y_true,y_pred)

class multinomialnll(tf.keras.losses.Loss):
    def __init__(self, name="multinomialnll", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        logits_perm = y_pred
        true_counts_perm = y_true

        #import numpy
        #np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/true_counts_perm.tsv", true_counts_perm, delimiter='\t')
        #np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/logits_perm.tsv", logits_perm, delimiter='\t')
        
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true)[0],dtype=tf.float32)

        return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen
        #tf.print("seqlen: ", seqlen, "true_counts_perm_shape: ", tf.shape(true_counts_perm), "counts_per_example: ", tf.shape(counts_per_example), "loss: ", loss)


class multinomialnll_mse(tf.keras.losses.Loss):
    def __init__(self, name="multinomialnll_mse", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.0000001
            self.counter=1
    def call(self, y_true, y_pred):

        # GOPHER implementation of loss

        '''
        #multinomial part of loss function

        # y_pred = tf.clip_by_value(y_pred, -0.9999, 1e10)
        logits_perm = y_pred
        true_counts_perm = y_true


        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        #seqlen = tf.cast(tf.shape(y_true[0])[0],dtype=tf.float32)
        seqlen = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)  # so this is intended to get the batch size

        true_counts_perm = y_true  # this should be Batch x Output_dim

        mult_loss = -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen # dist.log_prob(true_counts_perm) will be a vector with batch size

        #MSE part of loss function
        #mse_loss = tf.keras.losses.MSE(y_true[1], y_pred[1])

        log_true = tf.math.log(y_true + 1)
        log_pred = tf.math.log(y_pred + 1)

        mse_loss = tf.keras.losses.MSE(log_true, log_pred)

        #mse_loss = tf.keras.losses.MSE(y_true, y_pred)

        tf.print("mse_loss: ", mse_loss)
        mse_loss = tf.reduce_mean(mse_loss)

        #sum with weight
        total_loss = mult_loss + self.alpha*mse_loss


        #tf.print("seqlen: ", seqlen, "true_counts_perm_shape: ", tf.shape(true_counts_perm), "counts_per_example: ",
        #         tf.shape(counts_per_example), "mult_loss: ", mult_loss, "mse_loss: ", mse_loss, "total_loss: ", total_loss)
        '''


        ### BPNET implementation

        import keras.backend as K

        probs = y_pred / K.sum(y_pred, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = multinomialnll()(y_true, logits)

        '''
        np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/y_true.tsv", y_true,
                   delimiter='\t')
        np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/y_pred.tsv", y_pred,
                   delimiter='\t')
        '''

        MSE_loss = tf.keras.losses.MSE([K.log(1 + K.sum(y_true, axis=(-2, -1)))],
                                       [K.log(1 + K.sum(y_pred, axis=(-2, -1)))])


        bpnet_loss = multinomial_loss + self.alpha * MSE_loss

        total_loss = bpnet_loss

        '''
        if self.counter in range(0,40):
            epoch = 1
        elif self.counter in range(40,80):
            epoch = 2
        elif self.counter in range(80,120):
            epoch = 3
        elif self.counter in range(120,160):
            epoch = 4
        else:
            epoch = 5

        tf.print("epoch: ", epoch, "multinomialnll_GOPHER: ", total_loss, "multinomialnll_BPnet: ", bpnet_loss)
        self.counter = self.counter +1

        if self.counter == 39 or self.counter ==79 or self.counter == 119 or self.counter == 159 or self.counter==199:

            df=pd.DataFrame([[total_loss.numpy(), bpnet_loss.numpy()]], columns=['GOPHER_multinomialnll', 'BPnet_multinomialnll'])

            df.to_csv("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/Epoch_"+str(epoch)+"_loss_comp.tsv", sep = '\t')'''

        return total_loss

class multinomialnll_mse_bpnet(tf.keras.losses.Loss):
    def __init__(self, name="multinomialnll_mse_bpnet", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.0000001

    def call(self, y_true, y_pred):
        import keras.backend as K

        probs = y_pred / K.sum(y_pred, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = multinomialnll()(y_true, logits)

        np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/y_true.tsv", y_true,
                   delimiter='\t')
        np.savetxt("/Users/war9qi/Project_Data/maxATAC_sample/ELK1_quantitative_output/y_pred.tsv", y_pred,
                   delimiter='\t')

        mse_loss = tf.keras.losses.MSE([K.log(1 + K.sum(y_true, axis=(-2, -1)))],
                             [K.log(1 + K.sum(y_pred, axis=(-2, -1)))])

        return multinomial_loss + self.alpha * mse_loss



class multinomialnll_mse_reg(tf.keras.losses.Loss):
    def __init__(self, name="multinomialnll_mse_reg", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.0000001
        # self.alpha=0.001
    def call(self, y_true, y_pred):
        mult_loss = multinomialnll()(y_true, y_pred)

        #MSE part of loss function
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)

        #sum with weight
        total_loss = self.alpha*mult_loss + mse_loss

        return total_loss

class basenjipearsonr (tf.keras.losses.Loss):
    def __init__(self, name="basenjipearsonr", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
        true_sum = tf.reduce_sum(y_true, axis=[0,1])
        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
        pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=[0,1])
        true_mean = tf.divide(true_sum, count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(pred_sum, count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = product
        term2 = -tf.multiply(true_mean, pred_sum)
        term3 = -tf.multiply(pred_mean, true_sum)
        term4 = tf.multiply(count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = true_sumsq - tf.multiply(count, true_mean2)
        pred_var = pred_sumsq - tf.multiply(count, pred_mean2)
        tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
        correlation = tf.divide(covariance, tp_var)


        return -tf.reduce_mean(correlation)


class r2 (tf.keras.losses.Loss):
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        print(y_true)
        print(y_pred)
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
        print("y_true shape")
        print(y_true.shape)
        print(y_true)
        print("y_pred shape")
        print(y_pred.shape)
        print(y_pred)

        shape = y_true.shape[-1]
        true_sum = tf.reduce_sum(y_true, axis=[0,1])
        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=[0,1])

        true_mean = tf.divide(true_sum, count)
        print(true_mean)
        true_mean2 = tf.math.square(true_mean)
        print(true_mean2)

        total = true_sumsq - tf.multiply(count, true_mean2)
        print(total)

        resid1 = pred_sumsq
        resid2 = -2*product
        resid3 = true_sumsq
        resid = resid1 + resid2 + resid3
        print(resid)

        r2 = tf.ones_like(shape, dtype=tf.float32) - tf.divide(resid, total)
        return -tf.reduce_mean(r2)

class poissonnll(tf.keras.losses.Loss):
    def __init__(self, name="poissonnll", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        y_pred = tf.clip_by_value(y_pred, -0.9999, 1e10)
        logInput = tf.math.log(y_pred + 1)

        Target = y_true + 1

        loss = tf.nn.log_poisson_loss(log_input=logInput,
                                   targets=Target,
                                   compute_full_loss=True)

        return loss


class kl_divergence(tf.keras.losses.Loss):
    def __init__(self, name="kl_divergence", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        from sklearn.preprocessing import normalize

        # KLD call

        '''loss  = tf.keras.losses.KLDivergence().call(y_true=tf.linalg.normalize(y_true, ord=1, axis=1)[0],
                                                    y_pred=tf.linalg.normalize(y_pred, ord=1, axis=1)[0])'''

        epsilon = 1e-8
        y_true = tf.where(tf.math.is_nan(y_true), epsilon * tf.ones_like(y_true), y_true)
        y_pred = tf.where(tf.math.is_nan(y_pred), epsilon * tf.ones_like(y_pred), y_pred)

        y_true = tf.where(tf.math.is_inf(y_true), epsilon * tf.ones_like(y_true), y_true)
        y_pred = tf.where(tf.math.is_inf(y_pred), epsilon * tf.ones_like(y_pred), y_pred)

        y_true = tf.clip_by_value(y_true, epsilon, 1.0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        y_true_normalized = tf.linalg.normalize(y_true, ord=1, axis=1)[0]
        y_pred_normalized = tf.linalg.normalize(y_pred, ord=1, axis=1)[0]


        log_y_pred = tf.math.log(y_pred_normalized)

        # Compute the KL divergence manually
        loss = tf.reduce_sum(y_true_normalized * log_y_pred, axis=1)


        return loss

class cauchy_lf(tf.keras.losses.Loss):
    def __init__(self, name="cauchy_lf", **kwargs):
        super().__init__(name=name)
        self.gamma = 0.9

    def call(self, y_true, y_pred):
        """
        Compute the Cauchy loss function.

        Arguments:
        y_true -- tensor of true values
        y_pred -- tensor of predicted values
        gamma -- scale parameter, controls the robustness to outliers, 0.1 < gamma < 10

        Returns:
        computed Cauchy loss
        
        Thamsanqa Mlotshwa et al., Cauchy Loss Function: Robustness Under Gaussian and Cauchy Noise?
        https://arxiv.org/pdf/2302.07238
        """
        # Compute squared error
        squared_error = tf.square(y_true - y_pred)

        # Compute the Cauchy loss
        loss = ((self.gamma ** 2)/2) * tf.math.log(1 + squared_error / (self.gamma ** 2))

        return tf.reduce_mean(loss)  # Return the mean loss
