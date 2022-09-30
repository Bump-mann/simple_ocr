import os

import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm


#----------------------#
#   防止bug
#----------------------#
def get_train_step_fn(strategy):
    @tf.function
    def train_step(imgs1, imgs2, targets, net, optimizer):
        with tf.GradientTape() as tape:
            prediction = net([imgs1, imgs2], training=True)
            loss_value = tf.reduce_mean(K.binary_crossentropy(targets, prediction))

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        equal       = tf.equal(tf.round(prediction),targets)
        accuracy    = tf.reduce_mean(tf.cast(equal,tf.float32))
        return loss_value, accuracy

    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_train_step(imgs1, imgs2, targets, net, optimizer):
            per_replica_losses, per_replica_acc = strategy.run(train_step, args=(imgs1, imgs2, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_acc, axis=None)
        return distributed_train_step

#----------------------#
#   防止bug
#----------------------#
def get_val_step_fn(strategy):
    @tf.function
    def val_step(imgs1, imgs2, targets, net, optimizer):
        prediction = net([imgs1, imgs2], training=False)
        loss_value = tf.reduce_mean(K.binary_crossentropy(targets, prediction))
        return loss_value
    if strategy == None:
        return val_step
    else:
        #----------------------#
        #   多gpu验证
        #----------------------#
        @tf.function
        def distributed_val_step(imgs1, imgs2, targets, net, optimizer):
            per_replica_losses = strategy.run(val_step, args=(imgs1, imgs2, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return distributed_val_step
    
def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, save_period, save_dir, strategy):
    train_step  = get_train_step_fn(strategy)
    val_step    = get_val_step_fn(strategy)
    
    total_loss      = 0
    total_accuracy  = 0
    val_loss        = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images0, images1, targets = batch[0], batch[1], batch[2]
            
            loss_value, accuracy = train_step(images0, images1, targets, net, optimizer)
            total_loss      += loss_value.numpy()
            total_accuracy  += accuracy.numpy()

            pbar.set_postfix(**{'Total Loss'        : total_loss / (iteration + 1), 
                                'Total accuracy'    : total_accuracy / (iteration + 1),
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')
        
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_step_val:
                break
            images0, images1, targets = batch[0], batch[1], batch[2]

            loss_value = val_step(images0, images1, targets, net, optimizer)
            val_loss = val_loss + loss_value.numpy()
            
            pbar.set_postfix(**{'Val Loss'  : val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': total_loss / epoch_step, 'val_loss': val_loss / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
        
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        net.save_weights(os.path.join(save_dir, "best_epoch_weights.h5"))
            
    net.save_weights(os.path.join(save_dir, "last_epoch_weights.h5"))