#!/usr/bin/env python3
"""Model"""
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """ function that builds, trains, and saves neural network """
    X_train, Y_train = *Data_train
    X_valid, Y_valid = *Data_valid
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    saver = tf.train.Saver()
    initg = tf.global_variables_initializer()
    with tf.Session() as sess:
        loader.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        for i in range(epochs + 1):
            X_shu, Y_shu = shuffle_data(X_train, Y_train)
            accu_train = sess.run(accuracy, feed_dict={x: X_shu, y: Y_shu})
            loss_train = sess.run(loss, feed_dict={x: X_shu, y: Y_shu})
            accu_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            loss_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accu_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(accu_valid))
            if i < epochs:
                counter = 0
                j = 0
                z = batch_size
                while (z <= m):
                    sess.run(train_op, feed_dict={x: X_shu[j: z],
                                                  y: Y_shu[j: z]})
                    if (counter + 1) % 100 == 0 and counter != 0:
                        step_accu = sess.run(accuracy,
                                             feed_dict={x: X_shu[j: z],
                                                        y: Y_shu[j: z]})
                        step_cost = sess.run(loss, feed_dict={x: X_shu[j: z],
                                                              y: Y_shu[j: z]})
                        print("\tStep {}:".format(counter + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accu))
                    if (z + batch_size <= m):
                        j += batch_size
                        z += batch_size
                    else:
                        j += m % batch_size
                        z += m % batch_size
                    counter += 1

        return saver.save(sess, save_path)
