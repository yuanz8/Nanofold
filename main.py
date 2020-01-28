'''
Nano Fold project
The whole project will revolve around having a satisfaction score

    Database:
        - The database will be used to train the NN
            - supervised learning
    Neural Networks:
    The neural networks will possibly using reinforce learning
        - Angle prediction:
            - Ramachandran plot
        - Distance prediction:
            - Distance plot
    CRUCIAL NOTES:
     Needs:
        - To have variable output and inputs
            - Distance Plot [Depends on the number of residues]
                - N residues x N residues
            - Ramachandran plot [The convolutional output will be constant]
                - The coordinates will be
     Obtaining the data
        - Need to find 'quality' data for this to work
            - Needs to have answers to the sequences
            - The data has to be a very high resolution
    PURPOSE:
     - Determining the 3-D shape of a protein through its genetic sequence
'''
import tensorflow as tf
import tensorflow.keras.optimizers as opt
import tensorflow.keras.losses as loss
import lib.FindDevices as fd
from lib.NeuralNet import NeuralNet # FIXME: modify changes on this file related to import

# This will be a string input statement
seq = "."
proteinSequence = [] # FIXME: replace the '[]' with the output
resNet = NeuralNet()


def final_output(distance, angle):
    with tf.device(fd.getChosenDevice("GPU", 0)):
        grad_opti = tf.train.GradientDescentOptimizer(0.01)  # FIXME: Change the learning rate
        mod_dist_pred = model_dist.predict(seq, [])
        mod_angle_pred = model_angle.predict(seq, [])

        # FIXME: use gradient descent to optimize the 3d structure
        dist_min = grad_opti.minimise(loss=tf.convert_to_tensor(mod_dist_pred))
        angle_min = grad_opti.minimise(loss=tf.convert_to_tensor(mod_angle_pred))

    return dist_min, angle_min # Returns the output

def trainner(distance, angle):
    # Trains both the angle and distance neural network
    fin_trainning = False
    with tf.device(fd.getChosenDevice("GPU", 0)):
        train_mod_dist = distance.fit(seq, [], batch_size=128, epochs=30)
        print("\nhistory dict:", train_mod_dist.history)
        result_dist = distance.evaluate(seq, [], batch_size=128)
        print("test loss, test acc:", result_dist)
    print("I've finished trainning!!!")
    fin_trainning = True
    return result_dist, train_mod_dist.history, fin_trainning

with tf.device(fd.getChosenDevice("GPU", 0)):
    print("Im running on a GPU!!!")
    # FIXME change the decay and lr rate
    optimizer = opt.Adagrad(lr=0.1, decay=0.01)
    losses = loss.mean_squared_logarithmic_error()
    model_dist = resNet.residualNeuralNetworkDistance(seq)  # Distance prediction
    model_angle = resNet.residualNeuralNetworkAngle(seq)  # Angle Prediction

    dist_model = model_dist.compile(optimizer=optimizer, loss=losses)
    angle_model = model_angle.compile(optimizer=optimizer, loss=losses)

if __name__ == "main":
    # FIXME: Add the trainning here
    result, hist, fin = trainner(dist_model, angle_model)
    if fin == True:
        print("Type Y or y to continue.")
        enter = input("Do you want to make a prediction?  ")
        if enter == 'Y' or enter == 'y':
            final_output(dist_model, angle_model)
        else:
            print("Exiting...")
