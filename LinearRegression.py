# Linear regression algorithm
import numpy as np
import random
import matplotlib.pyplot as plt
from pickle import load,dump

# Functions to save files
def save_structure(structure,filename):
    file = open(filename,'wb')
    dump(structure,file)
    file.close()

def load_structure(filename):
    file = open(filename,'rb')
    structure = load(file)
    file.close()
    return structure

# Getting data
def read_data(file = 'kc_house_data.csv'):
    f = open(file)
    lines = f.read().split('\n')
    # Preprocessing data
    X,y = preprocess_data(lines[1:])
    save_structure(X,'X.pkl')
    save_structure(y,'y.pkl')
    return X,y

def feature_scaling(X):
    features = []
    s = []
    m = []
    new_X = []
    for i in range(18):
        features.append([])
    for x in X:
        for i in range(18):
            print(x,len(x))
            features[i].append(x[i])
    for f in features:
        f_a = np.array(f)
        s.append(np.std(f_a))
        m.append(np.mean(f_a))
    s = np.array(s)
    m = np.array(m)
    # Scaling
    for x in X:
        x_aux = (x - m) / s
        new_X.append(x_aux)
    return np.array(new_X)

def convert_numbers(line):
    l = []
    for e in line:
        if len(e) > 0:
            if '.' in e:
                l.append(float(e))
            else:
                l.append(int(e))
        else:
            l.append(0)
    return l

def preprocess_data(lines):
    X = []
    y = []
    for line in lines:
        l = line.split(',')
        if len(l) == 21:
            y.append(l[2]) # Price
            # Drop date and price
            l = l[3:]
            l = convert_numbers(l)
            # Convert in numpy array
            l = np.array(l)
            # Feature scaling
            X.append(l)
    X = np.array(X)
    X = feature_scaling(X)
    y = np.array(y)
    return X,y

def split_dataset(X,y,limit):
    training = []
    y_tr = []
    y_te = []
    testing = []
    aux_indices = [i for i in range(len(X))]
    print(type(aux_indices))
    random.shuffle(aux_indices)
    for i in aux_indices[:limit]:
        training.append(X[i])
        y_tr.append(y[i])
    for i in aux_indices[limit:]:
        testing.append(X[i])
        y_te.append(y[i])
    return np.array(training), np.array(y_tr), np.array(testing), np.array(y_te)

def predict(X,params,b):
    predictions = []
    for x in X:
        predictions.append(np.sum(np.transpose(params) * x) + b)
    #for x in X:
    #    predictions = np.sum(params * x) +
    predictions = np.array(predictions)
    #print(predictions)
    return predictions

def loss_function(y,predictions):
    loss = 0.5 * np.mean((y - predictions) ** 2)
    return loss,0.5 * ((y - predictions) ** 2)

def gradient_descent(X,y,params, b, alpha = 0.1):
    partial_derivatives , partial_b = loss_theta_der(X,y,params,b) , loss_b_der(X,y,params,b)
    params = params - alpha * partial_derivatives
    b = b - alpha * partial_b
    return params,b

def loss_theta_der(X,y,params,b):
    final_pd = []
    pd = [0 for i in range(18)]
    m = y.size
    for x in X:
        for i in range(len(x)):
            error = (np.sum(params*x) + b) - y[i]
            pd[i] += (1/m) * (error * x[i])
    pd = np.array(pd)
    pd = pd / len(X)
    return pd

def loss_b_der(X,y,params,b):
    pd = 0
    i = 0
    m = y.size
    for x in X:
        error = (np.sum(params * x) + b) - y[i]
        pd += (1/m) * error
        i += 1
    pd = pd / len(X)
    return pd

def train():
    total_loss = []
    # Create random parameters
    params = np.random.rand(1,18)
    b = random.random()
    j = 0
    # Load data
    #X,y = read_data()
    X,y = load_structure('X.pkl'), load_structure('y.pkl')
    # Split data
    training, y_tr, testing, y_te = split_dataset(X,y,int(len(X)*0.7))
    # Train model
    while j <= 1000:
        predictions = predict(training,params,b)
        loss = loss_function(y_tr, predictions)
        total_loss.append(loss)
        params, b = gradient_descent(X,y,params,b)
        if j % 50 == 0:
            print('Valor función de costo en la iteración: ',j,' : ',loss)
        j += 1
    save_structure(params,'params.pkl')
    save_structure(b,'b.pkl')
    plt.plot(range(1001), total_loss)
    plt.show()

def test():
    total_loss = []
    params = load_structure('params.pkl')
    b = load_structure('b.pkl')
    X,y = load_structure('X.pkl'), load_structure('y.pkl')
    # Split data
    training, y_tr, testing, y_te = split_dataset(X,y,int(len(X)*0.7))
    predictions = predict(testing,params,b)
    loss,lo = loss_function(y_te, predictions)
    total_loss.append(loss)
    for l in total_loss:
        print('Error promedio de todas las predicciones en la prediccion ',l)
    for e in range(len(list(lo))):
        print('Valor esperado:',y_te[e],' Valor obtenido: ', predictions[e],' Funcion de coste: ',lo[e])

test()
