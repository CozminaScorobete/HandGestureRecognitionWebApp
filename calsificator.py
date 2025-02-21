#Importam Librariile


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Setam un radome_seed pentru reproductibilitate ------------------
SEED = 62


#Definim path-urile pentru--------------------------

#DATASET
dataset = 'x_y_values.csv'

#UNDE VA FI SALVAT MODELUL
model_save_path = 'hand_recognition.hdf5'
#UNDE SE VA SALVA MODELUL CU EXTENSIA tflite  
''' Un fisier cu aceasta extensie a trecut printr-un proces de quantizare care reduce dimensiunile si mareste eficienta pentru deployment '''
tflite_save_path = 'hand_recognition.tflite'

#DEFINIM NUMARUL DE CLASE PE CARE MODELUL LE V-A GASI---------------------
NUM_CLASSES = 3

#definim variabilele unde vom salva seturile de date pe care le vom folosi----------------------------------
#X_dataset de la 1 la 42 (sunt 21 de pcte cu coordonatele x si y)
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
#Y_dataset pentru primacoloana
Y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

#Splituim setul de date in date de antrenament si date de test----------------------------------------
#am folosit functia train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, train_size=0.75, random_state=SEED)



#Modelam reteaua reuronala-----------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),  # Stratul de intrare are lungimea 42
    tf.keras.layers.Dropout(0.2),  # Strat de regularizare pentru combaterea overfitting-ului
    tf.keras.layers.Dense(30, activation='relu'),  # Stratul dens cu 30 de neuroni și activarea ReLU
    tf.keras.layers.Dropout(0.5),  # Al doilea strat de dropout pentru regularizare
    tf.keras.layers.Dense(15, activation='relu'),  # Al doilea strat dens cu 10 neuroni și activarea ReLU
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Stratul dens de ieșire cu activarea softmax
])


#Monitorizareaz si contolare antrenamentului retelei neuronale------------------------------------
# ModelCheckpoint Callback --> folosit pentru a salba modelulul cu cea mai buna performanta
modelCheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
#EarlyStopping Callback  --> folosit pentru a opri antrenamentul prematur daca performanta nu se imbunatateste
earlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


#cream optimizator personalizat și rata de învățare adaptativă-----------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  #un learning rate mai mic este mai incet dar aduce rezultate mai bune, pe cand un learning rate mai mare poate duce la instabilitate

# Compilare model cu optimizator personalizat, funcție de pierdere  și metrice suplimentare
model.compile(
    optimizer=optimizer,  #folosim optimizatorul ADAM
    loss='sparse_categorical_crossentropy',  # functia de loss masoara cat de bune sunt predictiile modelului, aceasta functie  sparse_categorical_crossentropy e folosita pentru modele cu mai multe clase
    metrics=['accuracy']  # masoara performanta in timpul trainingului
)


#Amtrenam modelul --------------------------------------------------------------------
history =model.fit(
    X_train,
    Y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, Y_test),
    callbacks=[modelCheckpoint_callback, earlyStopping_callback]
)


epochs = history.epoch
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot training & validation loss
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics.png')
plt.show()

# Evaluate the model
val_loss, val_acc = model.evaluate(X_test, Y_test, batch_size=128)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# Calculate additional metrics
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Classification report
report = classification_report(Y_test, Y_pred_classes, labels=[0, 1], target_names=[f'Class {i}' for i in range(NUM_CLASSES)])
print(report)

#Teste de inferenta--------------------------------------------------------------------------
predict_result = model.predict(np.array([X_test[0]]))
#print(np.squeeze(predict_result))
#print(np.argmax(np.squeeze(predict_result)))

# Salvaza modelul la path-ul indicat-----------------------------------------------------
model.save(model_save_path, include_optimizer=False)


# Transformarea modelului (quantization)-------------------------------------------------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model) #cream un ternsreflow lite convertor object
converter.optimizations = [tf.lite.Optimize.DEFAULT] #pornim optimizarile default 
tflite_quantized_model = converter.convert() #convertim modelul 

#Salvammodelul quantizat-----------------------------------------------------------
open(tflite_save_path, 'wb').write(tflite_quantized_model)


