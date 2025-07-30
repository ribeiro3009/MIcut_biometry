import os
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# Instale o TensorFlow e o Keras se ainda não o fez:
# pip install tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam


def load_data(annotations_path, images_dir):
    """Carrega imagens e máscaras com base nas anotações."""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    images = []
    masks = []

    for filename, data in annotations.items():
        img_path = os.path.join(images_dir, filename)
        mask_path = data['mask_path']

        # Carrega a imagem e a máscara
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is not None and mask is not None:
            # Redimensiona para um tamanho fixo (ex: 256x256)
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)

def unet_model(input_size=(256, 256, 3)):
    """Cria o modelo de arquitetura U-Net."""
    inputs = Input(input_size)

    # Codificador
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # Camada intermediária
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)

    # Decodificador
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(16, (3, 3), activation='relu', padding='same')(u5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    """Função principal para treinar o modelo."""
    # Caminhos para os dados
    annotations_path = 'ml_segmentation/annotations.json'
    #images_dir = 'ml_segmentation/merged_columns_Ml_Sample'
    images_dir = 'ml_segmentation/filtered_columns'

    # Carrega os dados
    images, masks = load_data(annotations_path, images_dir)

    # Normaliza as imagens e máscaras
    images = images / 255.0
    masks = masks / 255.0
    masks = np.expand_dims(masks, axis=-1)  # Adiciona dimensão para o canal

    # Divide os dados em conjuntos de treinamento e validação
    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Cria e treina o modelo
    model = unet_model()
    model.summary()

    print("\nIniciando o treinamento do modelo...")
    history = model.fit(
        x_train, y_train,
        batch_size=8,
        epochs=25,
        validation_data=(x_val, y_val)
    )

    # Salva o modelo treinado
    model.save('fingerprint_segmentation_model.h5')
    print("\nModelo treinado salvo como 'fingerprint_segmentation_model.h5'")

if __name__ == '__main__':
    main()
