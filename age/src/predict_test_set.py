from keras.models import load_model
import numpy as np

emotion_table = {0: 'neutral',
                 1: 'happiness',
                 2: 'surprise',
                 3: 'sadness',
                 4: 'anger',
                 5: 'disgust',
                 6: 'fear',
                 7: 'contempt'}




model = load_model('../logs/model-epoch-000176-acc-0.84578.hdf5')

def compute_accuracy(X_test, Y_test, file):
    Y_test_pred = model.predict(X_test, verbose=1)
    Y_test_pred_arg = np.argmax(Y_test_pred, axis=1)

    count = 0
    for i in range(Y_test.shape[0]):
        if (Y_test[i][Y_test_pred_arg[i]] == np.max(Y_test[i])):
            count += 1

    accuracy_score_test = count / Y_test.shape[0]
    print(file, accuracy_score_test)


X_test_path = '../data/FER2013Test/final_data.npy'
Y_test_path = '../data/FER2013Test/final_labels_data.npy'

X_test = np.load(X_test_path)
Y_test = np.load(Y_test_path)

img = np.reshape(X_test[12], newshape=(1,64,64,1))
print(emotion_table[np.argmax(model.predict(img, verbose=1))])






# Accuracy: 0.8223552894211577
# Accuracy: 0.8283433133732535
# Accuracy: 0.82178500142572
