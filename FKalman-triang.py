
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:23:05 2019

@author: bastian
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import os
from datetime import datetime
from scipy.linalg import block_diag
from scipy import stats
import fastavro
import time

import random
from numpy import linalg as LA
import matplotlib.cm as cm

from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


sns.set()


#%%

def avro_df(filepath, encoding):
    # Open file stream
    with open(filepath, encoding) as fp:
        # Configure Avro reader
        reader = fastavro.reader(fp)
        # Load records in memory
        records = [r for r in reader]
        # Populate pandas.DataFrame with records
        df = pd.DataFrame.from_records(records)
        # Return created DataFrame
        return df
#%%

mediciones = avro_df('D:/bckup/MEMO/portal_la_reina/portal_la_reina/' + 
                     'reina-2019-08-11.avro',
                     'rb')    


#%% Discovery Segmentación



os.chdir('D:\\bckup\\MEMO\\Proyectos\\Shopping Agosto19')

# mediciones =
# pd.read_csv('/home/bastian/Proyectos/Shopping Agosto19/
# PRUEBA_TRIANLULACION_PLR.csv',header=0)

# mediciones = (pd.read_csv('/home/bastian/Proyectos/' +
#                          'Discovery Error triangulación/' +
#                          'tests_CMX_PLR_RAW_2019' +
#                          '0930_PRUEBAS_DS.CSV', header=0))

mediciones = (pd.read_csv('D:\\bckup\\MEMO\\Proyectos\\Shopping ' +
                          'Agosto19\\PLR_validacion.csv', header=0))

errores = (pd.read_csv('D:\\bckup\\MEMO\\Proyectos\\Shopping Agosto19\\'+
                       'Triangulacion PLR\\Prueba_triangulacion.csv',
                       sep=';', header=0))

# Se realizará un ordenamiento de los datos de errores por fecha y
# mac address para luego cruzar con la base de mediciones.


# se elimina columna vacía

errores = errores.drop(labels='Unnamed: 16', axis=1)

# formato de fecha

errores['start_time'] = pd.to_datetime(errores['start_time'])

# Se agrupa por mac address y se ordena
# por fecha
errores = (errores.groupby('mac_address').apply
           (pd.DataFrame.sort_values, 'start_time'))

# eliminar duplicados

mediciones = mediciones.drop_duplicates()

fp = ("D:\\bckup\\MEMO\\Proyectos\\Shopping Agosto19\\" +
      "PYTHON_GCP_PROD\\INPUTS\\stores\\PLR_1stfloor_zones_20190325.dbf")

ap_coord = (pd.read_csv("D:\\bckup\\MEMO\\Proyectos\\" +
                        "Discovery Error triangulación\\ap_coords_PLR.csv",
                        sep=','))

ap_coord['x_px'] = ap_coord['x_px']/3.281
ap_coord['y_px'] = ap_coord['y_px']/3.281

gdf_polygon = gpd.read_file(fp)

prop_terreno_plr = (pd.read_csv("D:\\bckup/MEMO\\Proyectos\\" +
                                "Discovery Error triangulación\\" +
                                "proporciones_tiendas_plr.csv", sep=','))

prop_terreno_plr.set_index('Tienda', inplace=True)
prop_terreno_plr.sort_index(inplace=True)
prop_terreno_plr = prop_terreno_plr[(prop_terreno_plr.T != 0).any()]
prop_terreno_plr['Proporcion'] = (prop_terreno_plr['Visitas'] /
                                  sum(prop_terreno_plr['Visitas']))
prop_terreno_plr.drop(index='Maxi K', inplace=True)

# Posterior a settear el orden ascendente se implementará la media movil
# %% Algoritmo media móvil centrada para cada coordenada (rolling)

data_triang = (mediciones.sort_values(['deviceId', 'date', 'time'],
                                      ascending=False).reset_index(drop=True).
               copy())
data_triang.drop(['geoCoordinateLatitude', 'geoCoordinateLongitude', 'entity'],
                 axis=1, inplace=True)
# data_triang['x_corregido_mm'] = data_triang.groupby(['deviceId','date']).
# locationCoordinateX.apply(lambda x: x.rolling
# (window=3,center=True,min_periods=1).mean())
# data_triang['y_corregido_mm'] = data_triang.groupby(['deviceId','date']).
# locationCoordinateY.apply(lambda x: x.rolling(window=3,center=True,
# min_periods=1).mean())

data_triang['confidenceFactor'] = data_triang['confidenceFactor']/3.281
data_triang['pond'] = (data_triang.groupby(['deviceId', 'date']).
                       confidenceFactor.apply(lambda x: 1/x))

# %% Operaciones Fecha

# Se computa la fecha en un formato util para implementar luego una
# cuantificación del tiempo entre mediciones
data_triang["datetime"] = (data_triang.apply(lambda x: x.date + " " + x.time,
                           axis=1))
data_triang['datetime'] = (pd.to_datetime(data_triang['datetime'],
                           format="%Y-%m-%dT%H:%M:%S.%f"))

# Se cuantifica el tiempo entre mediciones, si se da el caso de que entre
# ambas mediciones exista un tiempo mayor a 280 segundos, se agrupan los
# ptos siguientes como una trayectoria aislada de la anterior.
data_triang['deltat'] = (data_triang.groupby(['deviceId',
                         'date']).datetime.diff())
data_triang['deltat'] = (np.abs(data_triang['deltat'].dt.total_seconds()).
                         fillna(value=np.inf))
data_triang['cambiotrayect'] = data_triang['deltat'] > 250
data_triang['numero_trayectoria'] = (data_triang.groupby(['deviceId',
                                     'date']).cambiotrayect.cumsum())

# %% Se procesa la columna locationMapHierarchy para poder usarla de validación
# a futuro

data_triang['locationMapHierarchy'] = (data_triang['locationMapHierarchy'].
                                       apply(lambda x: x.split(">")[-1]))
data_triang['macup'] = data_triang['deviceId'].apply(lambda x: x.upper())
data_triang['macup'] = data_triang['macup'].apply(lambda x: x.replace(':', ""))
data_triang['macup'] = data_triang['macup'].str.slice(0, 6, 1)
# %% Procesamiento de Mac's para realizar filtro por blacklist

dictmacs = (pd.read_csv("D:/bckup/MEMO/Proyectos/" +
                        "Discovery Error triangulación/" +
                        "diccionario_macs_manufacturer.csv",
                        sep=',',header=0))
blst = pd.read_csv("D:/bckup/MEMO/Proyectos/" +
                   "Discovery Error triangulación/blacklist_fabricantes.csv",
                   sep=',', header=0)
blacklist = pd.merge(dictmacs, blst, on='shortName', how='inner')
blacklist.reset_index()
blacklist.drop(['manufacturerUpper', 'shortName'], inplace=True, axis=1)
to_drop = blacklist['Assignment'].copy()
data_triang = (data_triang
               [~data_triang['macup'].str.contains('|'.join(to_drop))])


#%%
def set_labels(title=None, x=None, y=None):
  
    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)
    if title is not None:
        plt.title(title)


# %%Funciones para realizar la media móvil modificada

def weighted_moving_average(coord, weights, window=3):
    mm1 = []
    # Caso base (existan menos mediciones que el tamaño de la ventana)
    if len(coord)-1 <= 1:
        return coord
# El tamaño de momento de la ventana es 3, por lo que se realizan operaciones
# de media movil cada 3 puntos agregando como ponderador el confidence factor
    else:
        if window == 3:
            for i in range(len(coord)):
                if i == range(len(coord))[0]:
                    mm1.append((coord[i] * weights[i] + coord[i+1] *
                                weights[i+1]) / (weights[i] + weights[i+1]))
                elif i == range(len(coord))[-1]:
                    mm1.append((coord[i-1] * weights[i-1]+coord[i] *
                                weights[i]) / (weights[i-1] + weights[i]))
                else:
                    mm1.append((coord[i-1] * weights[i-1] + coord[i] *
                                weights[i]+coord[i+1]*weights[i+1]) /
                               (weights[i-1]+weights[i]+weights[i+1]))
    return mm1


def correccion_coord(coord, vec, cf):
    # Funcion util para imputar los valores que la media movil corrige
    # y tienen un confidence factor lo suficientemente bajo para que sean
    # confiables por si solos.
    for i in range(len(coord)):
        if cf[i] <= 10:
            vec[i] = coord[i]
    return vec

# %% Variables para media movil custom

t_inicio = time.time()
def custom_moving_average(dataframe):
    # Funcion que utiliza las 2 antes creadas para computar la media movil
    # custom respetando los grupos de mac id y día
    dfs = []
    for index_df, df in dataframe.groupby(['deviceId',
                                           'date', 'numero_trayectoria']):
        df = df.copy()
        weights = df['pond'].values
        inputx = df['locationCoordinateX'].values
        inputy = df['locationCoordinateY'].values
        cf = df['confidenceFactor'].values

        mmx = correccion_coord(inputx, weighted_moving_average(inputx,
                                                               weights), cf)
        mmy = correccion_coord(inputy, weighted_moving_average(inputy,
                                                               weights), cf)

        df['x_corregido_cf'] = mmx
        df['y_corregido_cf'] = mmy
        dfs.append(df)
    dataframe = pd.concat(dfs)
    return dataframe


data_triangv2 = data_triang.copy()
data_triang = custom_moving_average(data_triangv2)

tiempo_correr = (time.time() - t_inicio)  

# %%Preprocesado de los datos para darlos como inputs al modelo de KF

# Se toma el deviceId, numero de trayectoria, fecha

data_hist = data_triang[['deviceId', 'numero_trayectoria', 'datetime']].copy()

# Se adecúa la resolución del formato de la fecha

data_hist['datetime'] = (data_hist['datetime'] -
                         pd.to_timedelta(data_hist.datetime.dt.microsecond,
                                         unit='microsecond'))

data_hist = (data_hist.sort_values(['deviceId', 'numero_trayectoria',
                                   'datetime'], ascending=False).
             reset_index(drop=True))

data_hist['dt'] = (data_hist.groupby(['deviceId', 'numero_trayectoria']).
                   datetime.diff())

data_hist['dt'] = np.abs(data_hist['dt'].dt.total_seconds())

data_hist.dt.dropna(inplace=True)

dt_prom = data_hist.groupby(['deviceId', 'numero_trayectoria'])['dt'].mean()
dt_prom = dt_prom.reset_index()
dt_prom.rename(columns={'dt': 'dt_prom'}, inplace=True)

data_hist = data_hist.merge(dt_prom, on=['deviceId', 'numero_trayectoria'],
                            how='left')


data_hist.dt_prom.hist(bins=np.linspace(0, 100, 100), figsize=(16, 4),
                       grid=True)
data_hist.dt.hist(bins=np.linspace(0, 100, 100), figsize=(16, 4),
                  grid=True)
plt.title('Conteos de Deltas de tiempo vs su promedio')
plt.xlabel('Tiempo (s)')
plt.ylabel('Conteos')
# %% Fecha


data_triang['time'] = pd.to_datetime(data_triang['time'])
time_1 = pd.DatetimeIndex(data_triang.time)
data_triang['time'] = (data_triang['time'] -
                       pd.to_timedelta(data_triang.time.dt.microsecond,
                                       unit='microsecond'))
data_triang.sort_values(by=['deviceId', 'numero_trayectoria', 'time'],
                        inplace=True)
data_triang.reset_index(drop=True, inplace=True)

# %%Tasas de Refresco Globales

data_triang['deltat'].hist(bins=np.arange(0, 100, 1), figsize=(16, 4))
plt.title('Tasas de Refresco Globales')
# %%Tasas de refresco por dispositivo (Mediana de los dt de cada dispositivo)

(data_triang['deltat'].groupby(data_triang['deviceId']).
 median().hist(bins=np.arange(0, 200, 1), figsize=(16, 4)))
plt.title('Tasas de refresco por dispositivo'+
          '(Mediana de los dt de cada dispositivo)')

def vel(dx, dt_s):
    try:
        return dx/dt_s
    except Exception:
        return np.nan
# %% Calculo de direccion entre cada punto


def get_heading_and_vel(dataframe):
    df = dataframe.copy()
    df[['d_x', 'd_y']] = (df.groupby(['deviceId'])[['locationCoordinateX',
                                                    'locationCoordinateY']]
                          .diff())
    df['heading'] = np.arctan2(df['d_y'], df['d_x'])
    df['velocidad_x'] = [vel(dx, dt_s) for dx, dt_s in zip(df.d_x, df.deltat)]
    df['velocidad_y'] = [vel(dy, dt_s) for dy, dt_s in zip(df.d_y, df.deltat)]
    return df


data_triang = get_heading_and_vel(data_triang)
# %% Velocidad

data_triang['velocidad_x'].hist(bins=np.arange(0, 3, 0.1), figsize=(16, 4))
data_triang['velocidad_y'].hist(bins=np.arange(0, 3, 0.1), figsize=(16, 4))
# %% Conficence Factor
(data_triang['confidenceFactor'].
 hist(bins=np.arange(0, 100, 1), figsize=(16, 4)))
plt.title('Confidence Factor')

varcf = data_triang.loc[data_triang['confidenceFactor'] <= 20]
varcf = np.var(varcf.confidenceFactor)


# %% Ajuste del mapa

scale_x = 2.7976800402851403
scale_y = 2.739050408528786

max_x = data_triang['locationCoordinateX'].max()
min_x = data_triang['locationCoordinateX'].min()


max_y = data_triang['locationCoordinateY'].max()
min_y = data_triang['locationCoordinateY'].min()

mapa = mpimg.imread('mapa_plr.jpg')

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# %% Plot test


plt.figure(figsize=(16, 12))

x_coords = scale_x * data_triang['locationCoordinateX'].values
y_coords = scale_y * data_triang['locationCoordinateY'].values

x_px = scale_x * (ap_coord['x_px'].values-1)
y_px = scale_y * (ap_coord['y_px'].values-2)

plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))
plt.scatter(x_coords, y_coords, color="blue", s=9, marker='o', alpha=0.008,
            label='Dispositivos Triangulados')
plt.scatter(x_px, y_px, color="red", s=200, marker='o',
            label='Puntos de Acceso')
leg = plt.legend()

for lh in leg.legendHandles: 
    lh.set_alpha(1)

plt.xlim(0, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)
plt.savefig('paccs.png', format='png')

# %% Generación puntos ficticios

# x_data=[70.]*13
# x_data[5]=10.
# x_data[10]=150.
#
# y_data=[20.,40.,60.,80.,100.,125.,120.,140.,160.,180.,200.,220.,250.]
#
# x_data=np.array(x_data)/scale_x
# y_data=np.array(y_data)/scale_y
#
# x_data=x_data.tolist()
# y_data=y_data.tolist()
#
# z_kalman=np.column_stack((x_data,y_data))


# #%%#%%
#
# data_kalman=data_triang.loc[data_triang['deviceId'] == macid].copy()
# #data_kalman=data_kalman.loc[data_triang['locComputeType'] == 'AOA']
# #
# x_data=data_kalman.locationCoordinateX.values.tolist()
# y_data=data_kalman.locationCoordinateY.values.tolist()
# z_kalman=np.column_stack((x_data,y_data))

data_triang = (data_triang.loc[data_triang['locComputeType'] == 'RSSI'].
                    copy())
#%%
plt.clf()

data_triang['boxcox'], ml_val = stats.boxcox(data_triang['confidenceFactor'].
                                              values)

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.hist(data_triang['confidenceFactor'], bins = np.arange(0, 49, 1),
        label='Factor de Confianza', color='green')

# leg = ax.legend()

# for lh in leg.legendHandles: 
#     lh.set_alpha(1)


ax.set_xlabel('Valores de Factor de Confianza')
ax.set_ylabel('Cantidad de dispositivos')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(color='grey')
ax.set_title('Factor de Confianza', fontsize = 20)

plt.savefig('fcf.eps', format='eps')


#%%
plt.clf()
from scipy import stats

data_triang['boxcox'], ml_val = stats.boxcox(data_triang['confidenceFactor'].
                                              values)

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.hist(data_triang['boxcox'], bins = np.arange(0, 49, 1),
        label='Factor Transformado', color='red')


ax.set_xlabel('Valores de Factor de Confianza')
ax.set_ylabel('Cantidad de dispositivos')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(color='grey')
ax.set_title('Factor de Confianza Transformado', fontsize = 20)

plt.savefig('fcf2.eps', format='eps')

#%%
plt.clf()
from scipy import stats

data_triang['boxcox'], ml_val = stats.boxcox(data_triang['confidenceFactor'].
                                              values)

fig, ax = plt.subplots(1, figsize=(10, 6))


ax.hist(data_triang['confidenceFactor'], bins = np.arange(0, 49, 1),
        label='Factor de Confianza', color='green')
# (data_triang['confidenceFactor'].
#   hist(bins=np.arange(0, 49, 1), figsize=(16, 4),
#             label='Factor de Confianza', color='green'))
# sns.distplot(data_triang['confidenceFactor'])
# sns.distplot(data_triang['boxcox'])
ax.hist(data_triang['boxcox'], bins = np.arange(0, 49, 1), alpha=0.5,
        label='Factor Transformado', color='red')
# (data_triang['boxcox'].
#   hist(bins=np.arange(0, 49, 1), figsize=(16, 4),alpha=0.5,
#             label='Factor Transformado', color='red'))

leg = ax.legend()

for lh in leg.legendHandles: 
    lh.set_alpha(1)


ax.set_xlabel('Valores de Factor de Confianza')
ax.set_ylabel('Cantidad de dispositivos')
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(color='grey')
ax.set_title('Factores de Confianza Comparados', fontsize = 20)

plt.savefig('fcf3.png', format='png')
#%% Kalman


def pos_vel_filter(x, R, Q, dt, compute_log_likelihood=False):
    """ Crea un filtro de kalman que implementa un modelo [x dx y dy].T
    """
    P = np.diag([70., 1, 140, 1])
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x[0], x[1], x[2], x[3]])

    # Matriz de transición de estados
    kf.F = np.array([[1., dt, 0, 0.],
                     [0., 1., 0., 0],
                     [0., 0., 1., dt],
                     [0., 0., 0., 1.]])
    # Adquirir Measurements
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 0., 1., 0.]])

    # Ruido de Medicion
    kf.R = np.diag([R**2, R**2])

    # Ruido de Proceso
    kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt, var=Q)
    kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt, var=Q)
    kf.P[:] = P
  
    return kf



def correr_kalman(zs, dt=1, **kwargs):
    x_data = zs.locationCoordinateX.values.tolist()
    y_data = zs.locationCoordinateY.values.tolist()
    cfs = zs.boxcox.values.tolist()
    x0 = np.array([[x_data[0]], [1.], [y_data[0]], [1.]])
    kf = pos_vel_filter(x0, R=0, Q=0, dt=1)
    measurements = np.column_stack((x_data, y_data))

    # xs, cov = [], []
    xs, valps, cov = [], [], []

    for z, i in zip(measurements, cfs):
        kf.R = np.diag([i, i])
        q = Q_discrete_white_noise(2, dt, var=((i)/100))
        kf.Q = block_diag(q, q)
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)
        vecp, valp = LA.eig(kf.P)
        valps.append(valp)
        

    xs, valps, covs = np.array(xs), np.array(valps), np.array(cov)
    nz_x = xs[:, 0]
    nz_y = xs[:, 2]
    vp_x = valps[:,0,0]
    vp_y = valps[:,2,2]
    cv_x = covs[:,0,0]
    cv_y = covs[:,2,2]
    return nz_x, nz_y, vp_x, vp_y, cv_x, cv_y


def kalman_dataframe(dataframe):
    dfs = []
    # cvs = []
    # disps = []
#    for index_df, df in dataframe.groupby(['locComputeType', 'deviceId',
#                                           'date', 'numero_trayectoria']):
    for index_df, df in dataframe.groupby(['deviceId', 'date',
                                           'numero_trayectoria']):
        df = df.copy()
        df['x_k'], df['y_k'], df['vp_x'], df['vp_y'], df['cv_x'], df['cv_y'] = correr_kalman(df)
        dfs.append(df)
        # disps.append(index_df)
        # cvs.append(cv)
    dataframe = pd.concat(dfs)
    # return dataframe, cvs, disps
    return dataframe


#%%Mac Singular
#from kf_book.book_plots import plot_kf_output

#macids = data_triang['deviceId'].sample(n=1).iloc[0:].tolist()

#macid = macids[0]

#%% Kalman Singular
#sampledata = data_triang.loc[data_triang['deviceId'] == macid].copy()

#dkalman_s = correr_kalman(sampledata)

#%% Observar convergencia

#fig, (ax1, ax2) = plt.subplots(2)

#fig.suptitle('Comparación de Valores para ambos ejes coordenados')

#ax1.plot(list(range(len(dkalman_s.x[:,0,0].tolist()))),
         #dkalman_s.x[:,0,0].tolist())
#ax1.plot(list(range(len(sampledata.locationCoordinateX.values.tolist()))),
         #sampledata.locationCoordinateX.values.tolist())
#ax1.set(ylabel='x (metros)', xlabel='Numero de mediciones')


# %%Correr Filtro de Kalman
# macid='20:47:da:41:2b:47'
# macids=[macid]
# data_test=data_triang.loc[data_triang['deviceId'] == macid]
#
# data_kalman=kalman_dataframe(data_test)
t_inicio = time.time()
# data_kalman, cv, cv_indexs = kalman_dataframe(data_triang)
data_kalman = kalman_dataframe(data_triang)
tiempo_correr = (time.time() - t_inicio)  


#%%

data_kalman['dx_cmx_mm'] = np.abs(np.subtract(data_kalman.locationCoordinateX,
                                               data_kalman.x_corregido_cf))

data_kalman['dy_cmx_mm'] = np.abs(np.subtract(data_kalman.locationCoordinateY,
                                               data_kalman.y_corregido_cf))


data_kalman['dx_cmx_kf'] = np.abs(np.subtract(data_kalman.locationCoordinateX,
                                               data_kalman.x_k))
data_kalman['dy_cmx_kf'] = np.abs(np.subtract(data_kalman.locationCoordinateY,
                                               data_kalman.y_k))

# %% deviceId particular (RandomSample)

# macids = data_kalman['deviceId'].sample(n=1).iloc[0:].tolist()

# macid = macids[0]

##1

# macid = '48:c7:96:c8:76:6b'
# macids = [macid]


##3

# macid = 'e4:58:b8:b7:57:20'
# macids = [macid]

##pm

# macid = 'd0:31:69:09:71:be'
# macids = [macid]

##hm
macid = 'a4:d1:8c:61:72:04'
macids = [macid]

##tray

# macid = 'cc:c3:ea:5e:b5:d2'
# macids = [macid]

# #kalman mac test
# macid = 'd8:68:c3:2a:95:51'
# macids = [macid]

# kmactest2
# macid = 'c4:9a:02:15:07:05'
# macids = [macid]

# dc:66:72:1e:07:91
# macid = '20:47:da:41:2b:47'
# macids = [macid]
# macid = '90:63:3b:0f:fa:53'
# macs utiles: ['00:17:23:13:a4:b0' 90:63:3b:0f:fa:53' '90:63:3b:64:8e:99']

sampledata = data_kalman.loc[data_kalman['deviceId'] == macid].copy()

x_coords = scale_x * sampledata.locationCoordinateX
y_coords = scale_y * sampledata.locationCoordinateY
x_k = scale_x * sampledata.x_k
y_k = scale_y * sampledata.y_k
x_cf = scale_x * sampledata.x_corregido_cf
y_cf = scale_y * sampledata.y_corregido_cf
vp_x = scale_y * sampledata.vp_x
vp_y = scale_y * sampledata.vp_y


#%%Seleccion Ptos para mostrar en otro graf

x_sel = x_coords[0:9]
y_sel = y_coords[0:9]
 
x_ksel = x_k[0:9]
y_ksel = y_k[0:9]

x_cfsel = x_cf[0:9]
y_cfsel = y_cf[0:9]


#%%

def plot_residual_limits(Ps, fig, ax, stds=1.):
    " Plotea el margen de desviaciones estandar deseado "

    std = np.sqrt(Ps) * stds

    ax.plot(-std, color='k', ls=':', lw=2)
    ax.plot(std, color='k', ls=':', lw=2)
    ax.fill_between(range(len(std)), -std, std,
                 facecolor='#ffff00', alpha=0.3)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black') 
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.grid(color='grey')
    

def plot_residuals(xs, xs2, cov, title, y_label, stds=5):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    res = xs - xs2
    ax.plot(res)
    ax.set_xlim(left=0)
    plot_residual_limits(cov, fig, ax, stds)
    set_labels(title, 'Número de Medicion', y_label)


def hist_residuals(xs, xs2, title, y_label):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    res = xs - xs2
    ax.hist(res, color = 'blue')
    # hist_residual_limits(cov, fig, ax, stds)
    set_labels(title, 'Metros (m)', y_label)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black') 
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.grid(color='grey')
    
# def hist_residual_limits(Ps, fig, ax, stds=1.):

#     std = np.sqrt(Ps) * stds

#     ax.hist(std, color='blue')
#     # ax.plot(-std, color='k', ls=':', lw=2)
#     # ax.plot(std, color='k', ls=':', lw=2)
#     # ax.fill_between(range(len(std)), -std, std,
#     #              facecolor='#ffff00', alpha=0.3)
#     ax.spines['bottom'].set_color('black')
#     ax.spines['top'].set_color('black') 
#     ax.spines['right'].set_color('black')
#     ax.spines['left'].set_color('black')
#     ax.grid(color='grey')

#%%

plot_residuals(sampledata.locationCoordinateX.to_numpy(),
               sampledata.x_k.to_numpy(),
               sampledata.cv_x.to_numpy(),
               title='Residuos Posición X (5$\sigma$)',
               y_label='Metros (m)')
plt.savefig('residuospx_hm.eps', format='eps')

#%%

plot_residuals(sampledata.locationCoordinateY.to_numpy(),
               sampledata.y_k.to_numpy(),
               sampledata.cv_y.to_numpy(),
               title='Residuos Posición Y (5$\sigma$)',
               y_label='Metros (m)')
plt.savefig('residuospy_hm.eps', format='eps')

#%%

hist_residuals(sampledata.locationCoordinateX.to_numpy(),
               sampledata.x_k.to_numpy(),
               title='Residuos Posición X (5$\sigma$)',
               y_label='Cantidad de Mediciones')
plt.savefig('hresiduospx_hm.eps', format='eps')

#%%

hist_residuals(sampledata.locationCoordinateY.to_numpy(),
               sampledata.y_k.to_numpy(),
               title='Residuos Posición Y (5$\sigma$)',
               y_label='Cantidad de Mediciones')
plt.savefig('hresiduospy_hm.eps', format='eps')

#%%

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

fig.suptitle('Comparación de Valores para ambos ejes coordenados')

ax1.plot(list(range(len(sampledata.x_k.values.tolist()))),
         sampledata.x_k.values.tolist(), c='blue')
ax1.plot(list(range(len(sampledata.locationCoordinateX.values.tolist()))),
         sampledata.locationCoordinateX.values.tolist(), c = 'red', 
         alpha = 0.5)
ax1.set(ylabel='x (metros)')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')


ax2.plot(list(range(len(sampledata.y_k.values.tolist()))),
         sampledata.y_k.values.tolist(), c='blue')
ax2.plot(list(range(len(sampledata.locationCoordinateY.values.tolist()))),
         sampledata.locationCoordinateY.values.tolist(), c = 'red', 
         alpha = 0.5)
ax2.set(ylabel='y (metros)', xlabel='Numero de mediciones')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')

ax2.grid(color='grey')

#%%

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

fig.suptitle('Valores sin procesar vs Filtro de Kalman para coordenada X')

ax1.plot(list(range(len(sampledata.locationCoordinateX.values.tolist()))),
         sampledata.locationCoordinateX.values.tolist(), c = 'blue', 
         label='Valores sin Procesar')
ax1.set(ylabel='x (metros)')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=sampledata.locationCoordinateX.values.min()-5,
             top=sampledata.locationCoordinateX.values.max()+5)
ax1.legend()

ax2.plot(list(range(len(sampledata.x_k.values.tolist()))),
         sampledata.x_k.values.tolist(), c='red', 
         label='Valores de salida del Filtro de Kalman')
ax2.set(ylabel='x (metros)')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.grid(color='grey')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=sampledata.locationCoordinateX.values.min()-5,
             top=sampledata.locationCoordinateX.values.max()+5)
ax2.set_xlabel('Número de Medición')
ax2.legend()


plt.savefig('fkx_hm.eps', format='eps')
#%%

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

fig.suptitle('Valores sin procesar vs Filtro de Kalman para coordenada Y')

ax1.plot(list(range(len(sampledata.locationCoordinateY.values.tolist()))),
         sampledata.locationCoordinateY.values.tolist(), c = 'blue', 
         label='Valores sin Procesar')
ax1.set(ylabel='y (metros)')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=sampledata.locationCoordinateY.values.min()-5,
             top=sampledata.locationCoordinateY.values.max()+5)
ax1.legend()

ax2.plot(list(range(len(sampledata.y_k.values.tolist()))),
         sampledata.y_k.values.tolist(), c='red', 
         label='Valores de salida del Filtro de Kalman')
ax2.set(ylabel='y (metros)')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.grid(color='grey')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=sampledata.locationCoordinateY.values.min()-5,
             top=sampledata.locationCoordinateY.values.max()+5)
ax2.set_xlabel('Número de Medición')
ax2.legend()

plt.savefig('fky_hm.eps', format='eps')

#%%

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))


fig.suptitle('Valores sin procesar v/s Media Móvil eje X')

ax1.plot(list(range(len(sampledata.locationCoordinateX.values.tolist()))),
          sampledata.locationCoordinateX.values.tolist(), c = 'blue', 
          label='Valores sin Procesar')

ax1.grid(True)
# ax1.plot(list(range(len(sampledata.locationCoordinateX.values.tolist()))),
#          sampledata.locationCoordinateX.values.tolist(), c = 'red', 
#          alpha = 0.5)

ax1.legend()
ax1.set(ylabel='x (metros)')

ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=sampledata.locationCoordinateX.values.min()-5,
             top=sampledata.locationCoordinateX.values.max()+5)

ax2.plot(list(range(len(sampledata.x_corregido_cf.values.tolist()))),
         sampledata.x_corregido_cf.values.tolist(), c='red',
         label='Valores Media Movil')
ax2.grid(True)

ax2.set(ylabel='x (metros)', xlabel='Numero de mediciones')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.grid(color='grey')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=sampledata.locationCoordinateX.values.min()-5,
             top=sampledata.locationCoordinateX.values.max()+5)
ax2.set_xlabel('Número de Medición')
ax2.legend()

plt.savefig('cmxvsmm3.eps', format='eps')

#%%

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))


fig.suptitle('Valores sin procesar v/s Media Móvil eje Y')

ax1.plot(list(range(len(sampledata.locationCoordinateY.values.tolist()))),
          sampledata.locationCoordinateY.values.tolist(), c = 'blue',
          label='Valores sin Procesar')

ax1.legend()
ax1.set(ylabel='y (metros)')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=sampledata.locationCoordinateY.values.min()-5,
             top=sampledata.locationCoordinateY.values.max()+5)

ax2.plot(list(range(len(sampledata.y_corregido_cf.values.tolist()))),
          sampledata.y_corregido_cf.values.tolist(), c='red',
          label='Valores Media Móvil')
ax2.set(ylabel='y (metros)', xlabel='Numero de mediciones')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.grid(color='grey')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=sampledata.locationCoordinateY.values.min()-5,
             top=sampledata.locationCoordinateY.values.max()+5)
ax2.set_xlabel('Número de Medición')
ax2.legend()

plt.savefig('cmxvsmm4.eps', format='eps')

#%%

fig, ax1 = plt.subplots(1, figsize=(10, 6))


fig.suptitle('Factores de Confianza')

ax1.plot(list(range(len(sampledata.confidenceFactor.values.tolist()))),
         sampledata.confidenceFactor.values.tolist(), c = 'blue', 
          label='Valores sin Procesar')

ax1.grid(True)


# ax1.legend()
ax1.set(ylabel=f'x (metros\N{SUPERSCRIPT TWO})')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')

plt.savefig('cf_hm.eps', format='eps')

#%%


#Varianza en X
print("La Varianza en X de la Media Móvil es: " +
      str(np.var(sampledata.x_corregido_cf.values)))
print("La Varianza en X del Filtro de Kalman es: " +
      str(np.var(sampledata.x_k.values)))
print("La Varianza en X de las Mediciones sin procesar es: " +
      str(np.var(sampledata.locationCoordinateX.values)))

#Varianza de Y
print("La Varianza en Y de la Media Móvil es: " +
      str(np.var(sampledata.y_corregido_cf.values)))
print("La Varianza en Y del Filtro de Kalman es: " +
      str(np.var(sampledata.y_k.values)))
print("La Varianza en Y de las Mediciones sin procesar es: " +
      str(np.var(sampledata.locationCoordinateY.values)))




#%%


fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

fig.suptitle('Comparación de Valores propios para ambos ejes coordenados')

ax1.plot(list(range(len(sampledata.vp_x.values.tolist()))),
         sampledata.vp_x.values.tolist(), color = 'blue')
ax1.set(ylabel='Valor  X', xlabel='Numero de mediciones')
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')

ax2.plot(list(range(len(sampledata.vp_y.values.tolist()))),
         sampledata.vp_y.values.tolist(), color = 'green')
ax2.set(ylabel='Valor a Y', xlabel='Numero de mediciones')
ax2.spines['bottom'].set_color('black')
ax2.spines['top'].set_color('black') 
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.grid(color='grey')

# %%Test Preliminar de Trayectorias 1

plt.figure(figsize=(16, 9))

plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))

if max(x_coords.astype(int))+1 > 2:
    colores = cm.coolwarm(np.linspace(0, 1, max(x_coords.astype(int))+1))
else:
    colores = cm.coolwarm(np.linspace(0, 1, 2))

plt.subplot(131)
plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))

for x, y, c in zip(x_coords, y_coords, colores):
    plt.scatter(x, y, color=c, s=100, marker='o', edgecolor='black')

plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)
plt.title('Trayectoria Sin Procesar' + '\n para la MAC ID: ' + macid)

plt.subplot(132)
plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))
for x, y, c in zip(x_cf, y_cf, colores):
    plt.scatter(x, y, color=c, s=100, marker='o', edgecolor='black')
plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)
plt.title('Trayectoria Procesada por Media Movil custom' +
          '\n para la MAC ID: ' + macid)

plt.subplot(133)
plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))
for x, y, c in zip(x_k, y_k, colores):
    plt.scatter(x, y, color=c, s=100, marker='o', edgecolor='black')
plt.title('Trayectoria Procesada con Filtro de Kalman' + '\n para la MAC ID:'
          + ' ' + macid)

plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)



# %%Test Preliminar de Trayectorias 2
plt.figure(figsize = (16,9))

plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))

plt.subplot(131)
plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
plt.plot(x_coords,y_coords, 'bo-',markersize=10,label='Trayectoria Original',
         markeredgecolor='black')
plt.title('Trayectoria Sin Procesar' + '\n para la MAC ID: ' + macid)
plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)


plt.subplot(132)
plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
plt.plot(x_cf,y_cf,'r-o',markersize=10,label='Trayectoria con MM custom',
         markeredgecolor='black')
plt.title('Trayectoria Procesada por Media Movil custom' +
          '\n para la MAC ID: ' + macid)
plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)


plt.subplot(133)
plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
plt.plot(x_k,y_k,'go-',markersize=10,label='Trayectoria con Filtro de Kalman',
         markeredgecolor='black')
plt.title('Trayectoria Procesada con Filtro de Kalman' + '\n para la MAC ID:'
          + ' ' + macid)
plt.xlim(scale_x * min_x, scale_x * max_x)
plt.ylim(scale_y * min_y, scale_y * max_y)

#%%

fig, ax1 = plt.subplots(1, figsize=(10, 6))

plt.imshow(rgb2gray(mapa), cmap=plt.get_cmap('gray'))

plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
ax1.plot(x_sel,y_sel, 'bo-',markersize=10,label='Trayectoria Original',
         markeredgecolor='black')
ax1.set_title('Trayectoria Sin Procesar' + '\n para la MAC ID: ' + macid)
ax1.set_xlim(scale_x * min_x, scale_x * max_x)
ax1.set_ylim(scale_y * min_y, scale_y * max_y)
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')

plt.savefig('tray1_comps.eps', format='eps')
#%%

fig, ax1 = plt.subplots(1, figsize=(10, 6))
plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
ax1.plot(x_cfsel,y_cfsel,'r-o',markersize=10,label='Trayectoria con MM custom',
         markeredgecolor='black')
ax1.set_title('Trayectoria Procesada por Media Movil Modificada' +
          '\n para la MAC ID: ' + macid)
ax1.set_xlim(scale_x * min_x, scale_x * max_x)
ax1.set_ylim(scale_y * min_y, scale_y * max_y)
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')

plt.savefig('tray2_comps.eps', format='eps')
#%%

fig, ax1 = plt.subplots(1, figsize=(10, 6))
plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
ax1.plot(x_ksel,y_ksel,'go-',markersize=10,label='Trayectoria con Filtro de Kalman',
         markeredgecolor='black')
ax1.set_title('Trayectoria Procesada con Filtro de Kalman' + '\n para la MAC ID:'
          + ' ' + macid)
ax1.set_xlim(scale_x * min_x, scale_x * max_x)
ax1.set_ylim(scale_y * min_y, scale_y * max_y)
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black') 
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.grid(color='grey')

plt.savefig('tray3_comps.eps', format='eps')
# %% To CSV

data_triang_corr = data_kalman.copy()
data_raw_test = data_kalman.copy()
data_kalman_corr = data_kalman.copy()
data_triang_corr['y_corregido_cf'] = -1 * data_triang_corr['y_corregido_cf']
data_raw_test['locationCoordinateY'] = (-1 *
                                        data_raw_test['locationCoordinateY'])
data_kalman_corr['y_k'] = -1 * data_kalman_corr['y_k']
data_triang_corr = data_triang_corr[['deviceId', 'locComputeType', 'datetime',
                                     'x_corregido_cf', 'y_corregido_cf',
                                     'confidenceFactor']]
data_raw_test = data_raw_test[['deviceId', 'locComputeType', 'datetime',
                               'locationCoordinateX', 'locationCoordinateY',
                               'confidenceFactor']]
data_kalman_corr = data_kalman_corr[['deviceId', 'locComputeType', 'datetime',
                                     'x_k', 'y_k', 'confidenceFactor']]
data_triang_corr.to_csv('coord_PLR_modif.csv')
data_raw_test.to_csv('coord_PLR_raw.csv')
data_kalman_corr.to_csv('coord_PLR_kalman.csv')

data_aoa_valid = (data_triangv2.loc[data_triangv2['locComputeType'] == 'AOA'].
                  copy())
data_aoa_valid['locationCoordinateY'] = (-1 *
                                         data_aoa_valid['locationCoordinateY'
                                                        ])
data_aoa_valid = data_aoa_valid[['deviceId', 'locComputeType', 'datetime',
                               'locationCoordinateX', 'locationCoordinateY',
                               'confidenceFactor']]
data_aoa_valid.to_csv('coord_PLR_AOA_VALID.csv')
# %% Dataframe to GeoPandas
# Archivo de polígonos: gdf_polygon

# archivo de ptos:data_triang_corr (CF), data_raw_test (raw)


def puntos_conteo(dataframe, poligono, metodo=None):
    df = dataframe.copy()
    if metodo == 'mmc':
        df['coordinates'] = (list(zip(df.x_corregido_cf, df.y_corregido_cf)))
    elif metodo == 'kalman':
        df['coordinates'] = (list(zip(df.x_k, df.y_k)))
    else:
        df['coordinates'] = (list(zip(df.locationCoordinateX,
                             df.locationCoordinateY)))
    df.coordinates = df.coordinates.apply(Point)
    gdf_points = gpd.GeoDataFrame(df, geometry='coordinates')
    gdf_points.crs = poligono.crs
    sjoin = gpd.sjoin(poligono, gdf_points, how='left')
    df_sjoin = pd.DataFrame(sjoin)
    df_sjoin['id'] = df_sjoin['id'].astype(str)
    return df_sjoin


df_sjoin = puntos_conteo(data_triang_corr, gdf_polygon, 'mmc')
df_sjoin_raw = puntos_conteo(data_raw_test, gdf_polygon)
df_sjoin_kalman = puntos_conteo(data_kalman_corr, gdf_polygon, 'kalman')
df_sjoin_aoa = puntos_conteo(data_aoa_valid, gdf_polygon)

# %%


def contar_por_tiempo(df, hora_inicio, hora_final,
                      minutos_inicio=0, minutos_final=0, t_perm=False,
                      labels_data_empirica=[]):
    df_puntos_interior = df.loc[df['id'].apply(lambda x: x[0] == '2')].copy()
    # datetime(year, month, day, hour, minute, second, microsecond)
    año = int(df_puntos_interior['datetime'].dt.year.min())
    mes = int(df_puntos_interior['datetime'].dt.month.min())
    dia = int(df_puntos_interior['datetime'].dt.day.min())
    momento_inicio = (datetime(año, mes, dia, hora_inicio,
                               minutos_inicio, 0, 0))
    momento_termino = (datetime(año, mes, dia, hora_final,
                                minutos_final, 0, 0))
    selection_count = (df_puntos_interior.loc[(df_puntos_interior.datetime >=
                                               momento_inicio)
                                              & (df_puntos_interior.datetime <=
                                                  momento_termino)])
    selection_count = (selection_count.loc
                       [selection_count['confidenceFactor'] <= 10])
    if t_perm is True:
        t_perm = (selection_count.groupby(['tienda', 'deviceId'])['datetime'].
                  apply(lambda x: x.iloc[0] - x.iloc[-1]).dt.seconds)
        t_perm.rename("t_perm", inplace=True)
        df_t_perm = t_perm.reset_index()
        df_t_perm = df_t_perm[(df_t_perm['t_perm'] >= 60) &
                              (df_t_perm['t_perm'] <= 7200)]
        cuenta = df_t_perm.groupby(['tienda']).deviceId.nunique()
        cuenta = cuenta.rename("Visitas")
    else:
        cuenta = selection_count.groupby(['tienda']).deviceId.nunique()
        cuenta = cuenta.rename("Visitas")
    if labels_data_empirica is not None:
        cuenta_comparar = cuenta.reindex(index=labels_data_empirica)
        return cuenta, cuenta_comparar, t_perm
    else:
        return cuenta, t_perm


# %%
cuenta, cuenta_comparar, seleccion = contar_por_tiempo(df_sjoin, 11, 13, 20, 0,
                                                       True,
                                                       prop_terreno_plr.index)
aux_c1 = contar_por_tiempo(df_sjoin_raw, 11, 13, 20, 0, True,
                           prop_terreno_plr.index)
cuenta_raw, cuenta_raw_comparar, seleccion_raw = aux_c1
aux_c2 = contar_por_tiempo(df_sjoin_kalman, 11, 13, 20, 0, True,
                           prop_terreno_plr.index)
cuenta_kalman, cuenta_kalman_comparar, seleccion_kalman = aux_c2

aux_c3 = contar_por_tiempo(df_sjoin_aoa, 11, 13, 20, 0, True,
                           prop_terreno_plr.index)
cuenta_aoa, cuenta_aoa_comparar, seleccion_comparar = aux_c3
# %%
cuenta_terreno = prop_terreno_plr['Visitas'].values.tolist()

proporcion_sincorregir = (cuenta_raw_comparar/(cuenta_raw_comparar.sum()))
proporcion_corregidamm = (cuenta_comparar/(cuenta_comparar.sum()))
proporcion_kalman = (cuenta_kalman_comparar/(cuenta_kalman_comparar.sum()))
proporcion_aoa = (cuenta_aoa_comparar/(cuenta_aoa_comparar.sum()))
proporcion_terreno = prop_terreno_plr['Proporcion']

labels = proporcion_sincorregir.index.values.tolist()
valores_sincorregir = proporcion_sincorregir.values.tolist()
valores_corregidos = proporcion_corregidamm.values.tolist()
valores_corregidos_kalman = proporcion_kalman.values.tolist()
valores_aoa = proporcion_aoa.values.tolist()
valores_terreno = prop_terreno_plr['Proporcion'].values.tolist()

# %%


def delt_diff(data, d_empirica):
    aux_1 = np.abs(np.subtract(d_empirica, data))
    result = np.divide(aux_1, d_empirica)
    return result


# delta(diferencia) por Visitas
deltas_cmx = delt_diff(cuenta_raw_comparar, cuenta_terreno)
deltas_mmcustom = delt_diff(cuenta_comparar, cuenta_terreno)
deltas_kalman = delt_diff(cuenta_kalman_comparar, cuenta_terreno)
deltas_aoa = delt_diff(cuenta_aoa_comparar, cuenta_terreno)
# %%
fig, ax = plt.subplots(figsize=(16, 9))


x = np.arange(len(labels))
samp = np.random.randint(0, high=100, size=len(labels)).tolist()
y = random.sample(samp, 29)
ax.scatter(x, y, alpha=0.5, c='blue', s=deltas_cmx*100, label='Conteo por CMX'
           )
ax.scatter(x, y, alpha=0.5, c='green', s=deltas_mmcustom*100)
ax.scatter(x, y, alpha=0.5, c='orange', s=deltas_kalman*100,
           label='Conteo por Filtro de Kalman')
ax.set_title('Comparación Conteos de entrada a tienda')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.axes.get_yaxis().set_ticks([])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True,
          shadow=True, ncol=2)
for i, valor in enumerate(zip(deltas_cmx, deltas_kalman)):
    name = ('\u0394% CMX: ' + str(np.around(valor[0], decimals=2)) + '\n' +
            '\u0394% Kalman: ' + str(np.around(valor[1], decimals=2)))
    ax.annotate(name, (x[i], y[i]))

fig.tight_layout()

plt.show()
# %%

plt.clf()
fig, ax = plt.subplots(figsize=(16, 9))

x = np.arange(len(labels))
width = 0.2

rects1 = ax.bar(x - 2*width, valores_sincorregir, width,
                label='Proporcion sin corrección de trayectorias')
rects2 = ax.bar(x - width, valores_corregidos, width,
                label='Proporcion con corrección de trayectorias')
rects3 = ax.bar(x, valores_corregidos_kalman, width,
                label='Proporcion con Filtro de Kalman')
rects4 = ax.bar(x + width, valores_terreno, width,
                label='Proporcion empirica')

ax.set_ylabel('Valores Porcentuales')
ax.set_title('Proporciones de entradas a tiendas')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()
fig.tight_layout()

plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 6))

valores_terreno_2 = prop_terreno_plr['Visitas'].values.tolist()

valores_sincorregir_2 = cuenta_raw_comparar.values.tolist()
valores_corregidos_2 = cuenta_comparar.values.tolist()
valores_corregidos_kalman_2 = cuenta_kalman_comparar.values.tolist()

x = np.arange(len(labels))
width = 0.2

rects1 = ax.bar(x - 2*width, valores_sincorregir_2, width,
                label='Conteos CMX')
rects2 = ax.bar(x - width, valores_corregidos_2, width,
                label='Conteos Media Móvil')
rects3 = ax.bar(x, valores_corregidos_kalman_2, width,
                label='Conteos Filtro de Kalman')
rects4 = ax.bar(x + width, valores_terreno_2, width,
                label='Conteos empiricos')
ax.set_ylabel('Conteos')
ax.set_title('Comparación conteos de entrada a tienda')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(color='grey')

fig.tight_layout()

plt.show()

# %%

fig, ax = plt.subplots(figsize=(10, 6))

valores_terreno_2 = prop_terreno_plr['Visitas'].values.tolist()

valores_sincorregir_2 = cuenta_raw_comparar.values.tolist()
valores_corregidos_2 = cuenta_comparar.values.tolist()
valores_corregidos_kalman_2 = cuenta_kalman_comparar.values.tolist()

x = np.arange(len(labels))
width = 0.25

rects1 = ax.bar(x - width, valores_sincorregir_2, width,
                label='Conteos CMX')
rects2 = ax.bar(x, valores_corregidos_2, width,
                label='Conteos Media Móvil')
rects3 = ax.bar(x + width, valores_corregidos_kalman_2, width,
                label='Conteos Filtro de Kalman')
ax.set_ylabel('Conteos')
ax.set_title('Conteos de entrada a tienda intervalo de tiempo 12 a 13 hrs')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.grid(color='grey')

fig.tight_layout()



plt.savefig('conteos.eps', format='eps')
plt.show()
#%%

print(str(cuenta_raw.sum()))

print(str(cuenta_raw_comparar.sum()))

print(str(cuenta_kalman_comparar.sum()))

#%%

# def plot_conteo_tienda_s(tienda,tiendas_list=labels,conteos_sc =
#                          valores_sincorregir_2, conteos_mm =
#                          valores_corregidos_2, conteos_kf = 
#                          valores_corregidos_kalman_2):

#     sc_cont = dict(zip(tiendas_list, conteos_sc))
#     mm_cont = dict(zip(tiendas_list, conteos_mm))
#     kf_cont = dict(zip(tiendas_list, conteos_kf))
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     xt = np.arange(3)
     
    
    
    

# %%

fig, ax = plt.subplots(figsize=(16, 9))

x = np.arange(len(labels))
width = 0.3

rects1 = ax.bar(x - width, deltas_cmx.tolist(), width,
                label='Error Porcentual CMX')
rects2 = ax.bar(x, deltas_mmcustom.tolist(), width,
                label='Error Porcentual de Media Móvil')
rects3 = ax.bar(x + width, deltas_kalman.tolist(), width,
                label='Error Porcentual de Filtro de Kalman')
ax.set_ylabel('Error Porcentual')
ax.set_title('Comparación errores porcentuales de entrada a tienda')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.legend()

# def autolabel(rects):
#    """Da el valor del eje x exacto para cada barra"""
#    for rect in rects:
#        height = rect.get_height()
#        ax.annotate('{}'.format(np.around(height, decimals=2)),
#                    xy=(rect.get_x() + rect.get_width() / 2,
#                        np.around(height, decimals=2)),
#                    xytext=(0, 5),  # 3 points vertical offset
#                    textcoords="offset points",
#                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

fig.tight_layout()

plt.show()

print('Error Promedio de CMX ' + str(np.around((100*np.mean(deltas_cmx)),
                                               decimals=3))+'%')
print('Error Promedio de Media Móvil ' + str(np.around((100*
                                                        np.mean(
                                                            deltas_mmcustom))
                                             ,decimals=3))+'%')
print('Error Promedio de Filtro de Kalman ' + str(np.around((100*
                                                             np.mean(
                                                                 deltas_kalman
                                                                 ))
                                                  ,decimals=3))+'%')


# %%
# Memoria Carlos Tampier 2015
# Unscented Kalman Filter
# Matriz de covarianza per punto (Varianza)
# Entropía de la trayectoria

#ax = axs[0]
#ax.errorbar(x, valores_sincorregir, yerr=mse_cmx, ecolor='k', fmt='-co',
#            capthick=2)
#ax.set_ylabel('% de Visitas')
#ax.set_title('Distribución y error para proporciones de ' +
#             '\n conteo por CMX')
#
#ax = axs[1]
#ax.errorbar(x, valores_corregidos, yerr=mse_mmcustom, fmt='-yo', ecolor='k',
#            capthick=2)
#ax.set_ylabel('% de Visitas')
#ax.set_title('Distribución y error para proporciones de ' +
#             '\n conteo por Media Móvil Custom')
#
#ax = axs[2]
#ax.errorbar(x, valores_corregidos_kalman, yerr=mse_kalman, fmt='-ro',
#            ecolor='k', capthick=2)
#ax.set_ylabel('% de Visitas')
#ax.set_title('Distribución y error para proporciones de ' +
#             '\n conteo por filtro de kalman')
#ax.set_xticks(x)
#ax.set_xticklabels(labels, rotation='vertical')




#%%
#################################NO CORRER ABAJO! ###############################################

#data_triang['dx'] = data_triang['locationCoordinateX'].groupby(data_triang['deviceId']).diff()
#data_triang['dy'] = data_triang['locationCoordinateY'].groupby(data_triang['deviceId']).diff()
##%%
#def vel(dx,dt_s):
#    try:
#        return dx/dt_s
#    except:
#        return np.nan
#


 #%% Covs
# Se sacan los vectores propios de cada hipermatriz de covarianzas del
# filtro de kalman

# def vp_cov(listcov):
#     vec_p = []
#     for i in listcov:
#         eival, eivec = np.linalg.eig(i)
#         vec_p.append(eivec)
#     return vec_p
#%%
#
#data_triang['velocidad_x'] = [vel(dx,dt_s) for dx,dt_s in zip(data_triang.dx,data_triang.dt)]
#data_triang['velocidad_y'] = [vel(dy,dt_s) for dy,dt_s in zip(data_triang.dy,data_triang.dt)]
#data_triang['velocidad_x'].hist(bins = np.arange(0,2,0.1), figsize = (16,4))

#%%
#data_triang['velocidad_x'].groupby(data_triang['deviceId']).mean().hist(bins = np.arange(0,2,0.1), figsize = (16,4))
#data_triang['velocidad_y'].groupby(data_triang['deviceId']).mean().hist(bins = np.arange(0,2,0.1), figsize = (16,4))
#



#%%
#def plot_tray_temp(device_list):
#    plt.figure(figsize = (16,12))
#
#    for i in range(len(device_list)):
#
#        plt.subplot(int(np.ceil(len(device_list)/2)),2,i+1)
#
#        #sacamos coordenadas segun device y los centros por zona
#        device = device_list[i]
#        data_device = data_triang.loc[data_triang['deviceId'] == device].iloc[0:401]
#
#        x_coords = scale_x*data_device['locationCoordinateX'].values
#        y_coords = scale_y*data_device['locationCoordinateY'].values
##        x_soft   = scale_x*data_device['x_corregido_mm'].values
##        y_soft   = scale_y*data_device['y_corregido_mm'].values
#        x_cf     = scale_x*data_device['x_corregido_cf'].values
#        y_cf     = scale_y*data_device['y_corregido_cf'].values
#        tray     = data_device['numero_trayectoria'].values
#
#        x_centroid = scale_x*data_device['locationCoordinateX'].groupby(data_device['numero_trayectoria']).mean().values
#        y_centroid = scale_y*data_device['locationCoordinateY'].groupby(data_device['numero_trayectoria']).mean().values
##        x_mmcentroid = scale_x*data_device['x_corregido_mm'].groupby(data_device['numero_trayectoria']).mean().values
##        y_mmcentroid = scale_y*data_device['y_corregido_mm'].groupby(data_device['numero_trayectoria']).mean().values
#        x_cfcentroid = scale_x*data_device['x_corregido_cf'].groupby(data_device['numero_trayectoria']).mean().values
#        y_cfcentroid = scale_x*data_device['y_corregido_cf'].groupby(data_device['numero_trayectoria']).mean().values
#
#
#
#        #generamos mapa de colores
#        try:
#            colors_zona = cm.hot(np.linspace(0,1,max(data_device['numero_trayectoria'].astype(int))+1))
#        except:
#            colors_zona = cm.hot(np.linspace(0,1,2))
#
#        colors = colors_zona[list(data_device['numero_trayectoria'].astype(int))]
#
#        #ploteamos imagen en grises
#        try:
#            plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
#        except:
#            pass
#
#        #ploteamos puntos con color segun zona
#        count=0
#        for x,y,c,z in zip(x_cf,y_cf,colors,tray):
#            plt.scatter(x,y, color = c, s = 150, marker = 'o', edgecolor = 'black',label= z if z!=tray[count-1] else "",alpha = 0.7)
#            count+=1
#
#        #primer y ultimo punto
#        x_first = scale_x*data_device['locationCoordinateX'].groupby(data_device['numero_trayectoria']).first().values
#        y_first = scale_y*data_device['locationCoordinateY'].groupby(data_device['numero_trayectoria']).first().values
##        x_mmfirst = scale_x*data_device['x_corregido_mm'].groupby(data_device['numero_trayectoria']).first().values
##        y_mmfirst = scale_y*data_device['y_corregido_mm'].groupby(data_device['numero_trayectoria']).first().values
#        x_cfirst = scale_x*data_device['x_corregido_cf'].groupby(data_device['numero_trayectoria']).first().values
#        y_cfirst = scale_y*data_device['y_corregido_cf'].groupby(data_device['numero_trayectoria']).first().values
#
#        x_last = scale_x*data_device['locationCoordinateX'].groupby(data_device['numero_trayectoria']).last().values
#        y_last = scale_y*data_device['locationCoordinateY'].groupby(data_device['numero_trayectoria']).last().values
##        x_mmlast = scale_x*data_device['x_corregido_mm'].groupby(data_device['numero_trayectoria']).last().values
##        y_mmlast= scale_y*data_device['y_corregido_mm'].groupby(data_device['numero_trayectoria']).last().values
#        x_clast = scale_x*data_device['x_corregido_cf'].groupby(data_device['numero_trayectoria']).last().values
#        y_clast = scale_y*data_device['y_corregido_cf'].groupby(data_device['numero_trayectoria']).last().values
#
#        #ploteamos centroides de zonas
#        for x,y,c in zip(x_cfirst,y_cfirst,colors_zona):
#           plt.scatter(x,y, color = c, s = 100, marker = 'X', edgecolor = 'black')
#
#        #for x,y,c in zip(x_clast,y_clast,colors_zona):
#            #plt.scatter(x,y, color = c, s = 100, marker = 'v', edgecolor = 'black')
#
#        #for x,y,c in zip(x_soft,y_soft,colors_zona):
#           # plt.scatter(x,y, color = c, s = 150, marker = 'v', edgecolor = 'black')
#            #plt.plot(x,y, color = c,ls='-',linewidth=5)
#
#        #for x,y,c in zip(x_cf,y_cf,colors_zona):
#         #   plt.scatter(x,y, color = c, s = 150, marker = 'X', edgecolor = 'black')
#
#        #ploteamos centroides de zonas
#        #for x,y,c in zip(x_centroid,y_centroid,colors_zona):
#        #   plt.scatter(x,y, color = c, s = 100, marker = 'X', edgecolor = 'black')
#
#
#        plt.xlim(scale_x*min_x,scale_x*max_x)
#        plt.ylim(scale_y*min_y,scale_y*max_y)
#
#        try:
#            plt.title('Comportamiento del dispositivo ' + device + ' \n Numero de trayectorias: ' + str(max(data_device['numero_trayectoria'].astype(int))))
#        except:
#            plt.title('Comportamiento del dispositivo ' + device + ' \n Numero de trayectorias: sin trayectoria')
#        plt.legend(bbox_to_anchor=(1.04,1),loc='upper left',title='Número de trayectoria')
#    plt.show()
#plot_tray_temp(macids)
#


#%%plot con círculos para concentración de ptos
#plt.clf()
#plt.figure(figsize = (16,12))
#plt.imshow(rgb2gray(mapa),cmap = plt.get_cmap('gray'))
#plt.plot(x_coords,y_coords, 'bx-',markersize=10,label='Trayectoria Original')
#plt.plot(x_k,y_k,'c^-',markersize=10,label='Trayectoria con Filtro de Kalman')
#plt.plot(x_cf,y_cf,'ro',markersize=10,label='Trayectoria con MM custom')
#plt.legend(loc='upper right')
#fig = plt.gcf()
#ax = fig.gca()
#circle1 = plt.Circle((cenx_raw, ceny_raw), dist_raw, color='b',alpha=0.3)
#circle2 = plt.Circle((cenx_k, ceny_k), dist_k, color='c',alpha=0.4)
#circle3 = plt.Circle((cenx_cfmm, ceny_cfmm), dist_cfmm, color='r',alpha=0.5)
#ax.add_artist(circle1)
#ax.add_artist(circle2)
#ax.add_artist(circle3)
#plt.xlim(scale_x*min_x,scale_x*max_x)
#plt.ylim(scale_y*min_y,scale_y*max_y)
##ax.legend([circle1,circle2,circle3],['Radio de mediciones puras','Radio de mediciones corregidas con MM','Radio de mediciones corregidas con MM custom'])
#plt.show()


#%% Calculo centro de zona de error

#cenx_raw=np.mean(x_coords)
#ceny_raw=np.mean(y_coords)
#
#cenx_k=np.mean(x_k)
#ceny_k=np.mean(y_k)
#
#cenx_cfmm=np.mean(x_cf)
#ceny_cfmm=np.mean(y_cf)
#
#dist_raw=max(np.sqrt(np.power(x_coords-cenx_raw,2)+np.power(y_coords-ceny_raw,2)))
#dist_k=max(np.sqrt(np.power((x_k)-cenx_k,2)+np.power((y_k)-ceny_k,2)))
#dist_cfmm=max(np.sqrt(np.power((x_cf)-cenx_cfmm,2)+np.power((y_cf)-ceny_cfmm,2)))



#%%kalman Sin DATAFRAME

#def pos_vel_filter(x,P,R,Q,dt):
#    """ Crea un filtro de calman que implementa un modelo [x y dx dy].T
#    """
#
#    kf = KalmanFilter(dim_x=4, dim_z=2)
#    kf.x = np.array([x[0],x[1],x[2],x[3]])
#    kf.F = np.array([[1.,0.,dt,0.],
#                     [0.,1.,0.,dt],
#                     [0.,0.,1.,0.],
#                     [0.,0.,0.,1.]])  # Matriz de transición de estados
#
#    kf.H = np.array([[1.,0.,0.,0.],
#                     [0.,1.,0.,0.]])    # Adquirir Measurements
#
#    kf.R *= R                     # Incertidumbre
#    if np.isscalar(P):
#        kf.P *= P                 # covariance matrix
#    else:
#        kf.P[:] = P               # [:] makes deep copy
#    if np.isscalar(Q):
#        kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=Q)
#    else:
#        kf.Q[:] = Q
#    return kf
#
#def correr_kalman(zs,dt=1,**kwargs):
#    x_data=zs[:,0]
#    y_data=zs[:,1]
#    P=np.array([[random.randint(0,int(max(x_data))//2),0.,0.,0.],
#             [0.,random.randint(0,int(max(y_data))//2),0.,0.],
#             [0.,0.,0.09,0.],
#             [0.,0.,0.,0.09]])
#    x0=np.array([[200.],[75.],[0.],[0.]])
#    R=0.6
#    Q=0
#    kf=pos_vel_filter(x0,P,R,Q,dt)
#
#    xs,cov=[],[]
#
#    for z in zs:
#        kf.predict()
#        kf.update(z)
#        xs.append(kf.x)
#        cov.append(kf.P)
#
#    xs,cov=np.array(xs),np.array(cov)
#    nz_x= xs[:,0]
#    nz_y= xs[:,1]
#    nz_dx= xs[:,2]
#    nz_dy= xs[:,3]
#    new_z=np.concatenate((nz_x,nz_y),axis=1)
#    new_v=np.concatenate((nz_dx,nz_dy),axis=1)
#    return nz_x,nz_y
#
#data_kalman['x_k'],data_kalman['y_k']=correr_kalman(z_kalman)

#data_kalman['x_kalman']=z_new_kalman[:,0]
#data_kalman['y_kalman']=z_new_kalman[:,1]

#dg_x_kalman=z_new_kalman[:,0]
#dg_y_kalman=z_new_kalman[:,1]

#z_new_kalman=test_kalman_filter(x_data,y_data,0.1,0.6,1)
#dg_x_kalman=z_new_kalman[:,0]
#dg_y_kalman=z_new_kalman[:,1]


#%% Kalman Filter Sin FilterPy Para trayectorias Creadas
#Juguete para ajustar parámetros

#def test_kalman_filter(zs,dt,R_var,Q_var):
#    dt=dt #Paso
#    R_var=R_var #Varianza de sensores
#    Q_var=Q_var #Varianza del proceso (Mediciones)
##    z_x=df.locationCoordinateX.values.tolist()
##    z_y=df.locationcoordinateY.values.tolist()
#    measurements=zs
#    z_x=measurements[:,0]
#    z_y=measurements[:,1]
#    dim=4
#
#    x=np.array([[200.],[75.],[0.],[0.]]) #Estado inicial
#    P=np.diag([random.randint(0,int(max(z_x))//2), #Matriz de covarianza
#               random.randint(0,int(max(z_y))//2), #De los estados
#               0.09,0.09])
#    F=np.array([[1.,0.,dt,0.],
#                [0.,1.,0.,dt],
#                [0.,0.,1.,0.],
#                [0.,0.,0.,1.]])  # Matriz de transición de estados
#    H=np.array([[1.,0.,0.,0.],
#                [0.,1.,0.,0.]]) # Adquirir Measurements
#    R=np.array([[R_var]])
#    Q=Q_discrete_white_noise(dim=dim,dt=dt,var=Q_var)
#
#    x_new,y_new,dx_new,dy_new,cov=[],[],[],[],[]
#
#
#    for z in measurements:
#        z=z.reshape(-1, 1)
#
#        # predict
#        x = dot(F, x)
#        P = dot(F, P).dot(F.T) + Q
#
#        #update
#        S = dot(H, P).dot(H.T) + R
#        K = dot(P, H.T).dot(inv(S))
#        y = z - dot(H, x)
#        x += dot(K, y)
#        P = P - dot(K, H).dot(P)
#
#        x_new.append(x[0])
#        y_new.append(x[1])
#        dx_new.append(x[2])
#        dy_new.append(x[3])
#        cov.append(P)
#    new_pos=np.concatenate((x_new,y_new),axis=1)
#    return new_pos
#
#z_new_kalman=test_kalman_filter(z_kalman,1.,0.6,0)
#dg_x_kalman=z_new_kalman[:,0]
#dg_y_kalman=z_new_kalman[:,1]


# from filterpy.discrete_bayes import normalize

# def gen_dist_prob(dataframe):
#     # Funcion que genera distribuciones gaussianas para cada trayectoria
#     # en relación con su factor de confianza
#     dfs = []
#     for index_df, df in dataframe.groupby(['deviceId','date', 
#                                            'numero_trayectoria']):
#         df = df.copy()
#         cfs = df['confidenceFactor'].values.copy()
#         norm_dist_vals = normalize(cfs)
#         norm_dist_vals = 1 - norm_dist_vals 
#         df['valor_en_fdp'] = norm_dist_vals
#         dfs.append(df)   
#     dataframe = pd.concat(dfs)
#     return dataframe


