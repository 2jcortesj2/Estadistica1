import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def remove_outliers_iqr(data):
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    data = np.where(data>upper_bound, upper_bound, np.where(data<lower_bound,lower_bound,data))
    return data[(data >= lower_bound) & (data <= upper_bound)]

columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
heart_df = pd.read_csv('heart_failure_clinical_records_dataset.csv', usecols= columns)

for i in heart_df.columns:
    heart_df[i] = remove_outliers_iqr(heart_df[i])

scaler =StandardScaler()
df_estand = scaler.fit_transform(heart_df)
df_estand = pd.DataFrame(df_estand, columns=heart_df.columns)

pca = PCA(n_components= 6) # Numero 6 por el numero de variables
data_pca = pca.fit_transform(df_estand)

# Hay que normalizar los datos para que tengan una escala en comun
# scale
scaled_1 = scale(heart_df) # Z-score

# MinMaxScaler
scaler = MinMaxScaler()
scaled_2 = scaler.fit_transform(heart_df)

# Por defecto usa la distancia euclidea
km = KMeans(
    n_clusters = 2, init='random',
    n_init=10, random_state=0
)

y_km = km.fit_predict(scaled_1)

from mpl_toolkits.mplot3d import Axes3D


# Función para inicializar el gráfico
def init():
    ax.view_init(elev=20, azim=30)  # Puedes ajustar los ángulos iniciales según tu preferencia
    return ax

# Función de actualización para cada frame de la animación
def update(frame):
    ax.view_init(elev=20, azim=frame)  # Cambia el ángulo de azimut para girar la figura
    return ax

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

columna_1 = 0
columna_2 = 1
columna_3 = 2  # Asegúrate de que tienes una tercer columna en tu matriz de datos.

ax.scatter(
    scaled_1[y_km == 0, columna_1], scaled_1[y_km == 0, columna_2], scaled_1[y_km == 0, columna_3],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='Supervivencia'
)

ax.scatter(
    scaled_1[y_km == 1, columna_1], scaled_1[y_km == 1, columna_2], scaled_1[y_km == 1, columna_3],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='Muerte'
)

ax.scatter(
    km.cluster_centers_[:, columna_1], km.cluster_centers_[:, columna_2], km.cluster_centers_[:, columna_3],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroides'
)

ax.set_xlabel('Edad')
ax.set_ylabel('Creatinina fosfoquinasa')
ax.set_zlabel('Fracción de eyección')
ax.set_title('Clusters 3D')
plt.legend()
plt.grid()
plt.show()



# Configuración de la animación
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), init_func=init, blit=False)

# Guardar la animación como un archivo GIF
ani.save('cluster_animation.gif', writer='imagemagick', fps=30)

# Mostrar la animación en el cuaderno (si estás usando un entorno interactivo)
plt.show()
