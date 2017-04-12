import numpy as np
#np.random.seed(123)

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

import dicom, os, argparse, pickle
from random import randint

from keras import backend as K
K.set_image_dim_ordering('th')

from datetime import datetime
import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['X_train', 'Y_train'], loc='upper left')
    fig = plt.figure()
    fig.savefig('accuracyPlot.png')

def gen_patient_and_path(data_path):
    patient_and_path = {}
    for f in os.listdir(data_path):
        path = os.path.join(data_path, f)
        if os.path.isfile(path):
            print(path)
            image_file = dicom.read_file(path)
            patient_id = image_file.PatientID
            #print(patient_id)

                #Break this up to return a cancer dictionary and non-cancer dictionary


            if patient_id not in patient_and_path:
                patient_and_path[patient_id] = [path]
            else:
                patient_and_path[patient_id].append(path)
    print(len(patient_and_path))
    return patient_and_path

def gen_submission_and_path(patient_and_path):
    submissions = gen_submissions()
    print(len(submissions))
    submissions_and_path = {}
    for id, file in patient_and_path.items():
         print(id)
    for id in submissions:
         if id in patient_and_path:
             print(id)
             if id not in submissions_and_path:
                 print(id)
                 submissions_and_path[id] = patient_and_path[id]
    print("size of submission_path dictionary: " + str(len(submissions_and_path)))
    return submissions_and_path

def gen_submissions():
    tests = []
    sub_path = "../data/sample_submission.csv"
    with open(sub_path, 'r') as submission_file:
        for line in submission_file:
            id = line.strip().split(',')[0]
            print(id)
            tests.append(id)
    print("end submission file")
    return tests


def load_image_paths(data_path, num_images):
    i = 0
    image_paths = []
    for f in os.listdir(data_path):
        path = os.path.join(data_path, f)
        if os.path.isfile(path):
            image_paths.append(path)
            i += 1
        if i > num_images:
            break

    label_path = '../data/stage1_labels.csv'
    labels = {}
    with open(label_path, 'r') as label_file:
        for line in label_file:
            id, cancer = line.strip().split(',')
            labels[id] = cancer

    # shuffle image_paths
    np.random.shuffle(image_paths)
    return image_paths, labels

def get_pixel_array(image_path):
    image = dicom.read_file(image_path)
    image = image.pixel_array.astype(np.float)
    image /= np.max(image)
    return image

def load_images(image_paths):
    pixel_images = []
    for path in image_paths:
       pixel_images.append(get_pixel_array(path))

    X = np.ndarray(shape=(len(pixel_images), 1, 512, 512), dtype=np.float)
    for i in range(len(pixel_images)):
        X[i] = pixel_images[i]

    return X

def gen_random_labels(num_images):
    Y = np.ndarray(shape=(num_images, 2), dtype=np.int)
    for i in range(num_images):
        case = randint(0, 1)
        if case == 0:
            Y[i] = [1, 0]
        else:
            Y[i] = [0, 1]
    return Y

def gen_labels(image_paths, labels):
    Y = np.ndarray(shape=(len(image_paths), 2), dtype=np.int)
    for i in range(len(image_paths)):
        image_file = dicom.read_file(image_paths[i])
        patient_id = image_file.PatientID
        if patient_id in labels: # a label for this patient exists
            case = labels[patient_id]
        else: # assign a random label
            case = randint(0, 1)
        if case == 0:
            Y[i] = [1, 0]
        else:
            Y[i] = [0, 1]
    return Y

def gen_tests(image_paths, labels):
    tests = []
    matching_tests = []
    for i in range(len(image_paths)):
        image_file = dicom.read_file(image_paths[i])
        patient_id = image_file.PatientID
        if patient_id not in labels:
            tests.append(patient_id)
    sub_path = "../data/sample_submission.csv"
    with open(sub_path, 'r') as submission_file:
        for line in submission_file:
            id = line.strip().split(',')[0]
            if id in tests:
                matching_tests.append(id)
    return tests

def gen_patient_images(image_paths):
    patient_images = {}
    for i in range(len(image_paths)):
        image_file = dicom.read_file(image_paths[i])
        patient_id = image_file.PatientID
        if patient_id in patient_images:
            patient_images[patient_id].append(image_paths[i])
        else:
            patient_images[patient_id] = [image_paths[i]]
    return patient_images

def load_data_arrays(image_paths, labels):
    num_images = len(image_paths)
    print('Number of images to train:', num_images)
    num_train = int(num_images * 0.8)

    X_train_path = '../data/X_train_' + str(num_images) + '.npy'
    X_test_path = '../data/X_test_' + str(num_images) + '.npy'

    # load X_train and X_test files if created before, otherwise create them
    if not os.path.isfile(X_train_path):
        #Find patient/path dictionary

        X_train = load_images(image_paths[0:num_train])
        np.save(X_train_path, X_train)
        print('Completed X_train generation')


    else:
        X_train = np.load(X_train_path)

    if not os.path.isfile(X_test_path):
        X_test = load_images(image_paths[num_train:num_images])
        np.save(X_test_path, X_test)
        print('Completed X_test generation')
    else:
        X_test = np.load(X_test_path)

    Y = gen_labels(image_paths, labels)
    Y_train = Y[0:num_train]
    Y_test = Y[num_train:num_images]

    return X_train, Y_train, X_test, Y_test

def train_model(num_epochs, batch_size, data):
    X_train = data[0][0]
    Y_train = data[0][1]

    X_test = data[1][0]
    Y_test = data[1][1]

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation = 'relu', input_shape = (1, 512, 512)))

    model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    checkpoint_path = 'weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, mode='max', period=1)
    callbacks_list = [checkpoint]

    history = model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=num_epochs, callbacks=callbacks_list, verbose=1)
    plot_accuracy(history)
    model.save('model.hdf5')



    score = model.evaluate(X_test, Y_test, verbose=1)
    print(score)

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def make_prediction(model, patient_imagepath_dict):
    result = {}
    for patient, imagepaths in patient_imagepath_dict.items():
        imageArray = load_images(imagepaths)
        score = model.predict(imageArray, verbose = 0)
        average = 0
        for i in range(len(score)):
            if score[i][0] == 0:
                average += 1
        average = average/len(score)
        if average >= 0.5:
            result[patient] = 1
        else:
            result[patient] = 0
    return result

def output_results(path):
    with open(path, 'w') as submission_file:
        submission_file.write("id, cancer \n")
        for id, result in prediction.items():
            submission_file.write(str(id) + ', ' + str(result) + '\n')

def output_args(args):
    print('Number of epochs:', args.num_epochs)
    print('Batch size:', args.batch_size)
    print('Percentage of training set:', args.perc_train)
    print('Data path used:', args.data_path, flush=True)
    if args.predict:
        print('Using model to make predictions...')

if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--perc_train', help='The percentage of the data to include in the training set.', type=float, default=0.8)
    parser.add_argument('--data_path', help='The path to the parent directory of the data image files.', type=str, default='../data/stage1/')
    parser.add_argument('--predict', type=bool, default=False)
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--output_path', default='./results/results.csv')

    args = parser.parse_args()

    output_args(args)
    if args.predict:
        print('Loading image paths', flush=True)
        patient_and_path = {}

        if not os.path.isfile('./data/patient_and_path.p'):
            patient_and_path = gen_patient_and_path(args.data_path)

            with open('./data/patient_and_path.p', 'wb') as pat_path:
                pickle.dump(patient_and_path, pat_path)
        else:
            with open('./data/patient_and_path.p', 'rb') as pat_path:
                patient_and_path = pickle.load(pat_path)


        submission_and_path = {}
        if not os.path.isfile('./data/submission_and_path.p'):
            submission_and_path = gen_submission_and_path(patient_and_path)
            with open('./data/submission_and_path.p', 'wb') as sub_path:
                pickle.dump(submission_and_path, sub_path)
        else:
            with open('./data/submission_and_path.p', 'rb') as sub_path:
                submission_and_path = pickle.load(sub_path)
        print('Loading the model', flush=True)
        model_path = './model.hdf5'
        trained_model = load_trained_model(model_path)
        print ('Model loaded')
        print('Making predictions')
        prediction = make_prediction(trained_model, submission_and_path)
        with open(args.output_path, 'w') as submission_file:
            submission_file.write("id,cancer \n")
            for id, result in prediction.items():
                submission_file.write(str(id) + ',' + str(result) + '\n')

    else:
        print('Loading image paths', flush=True)
        image_paths, labels = load_image_paths(args.data_path, args.num_images)
        print('Loading data arrays', flush=True)
        X_train, Y_train, X_test, Y_test = load_data_arrays(image_paths, labels)
        data = ((X_train, Y_train), (X_test, Y_test))
        print('Training model', flush=True)
        train_model(args.num_epochs, args.batch_size, data)

    print('Total runtime: ' + str(datetime.now() - start))
