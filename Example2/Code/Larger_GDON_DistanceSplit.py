import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import deepxde as dde
from deepxde.backend import tf
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler , QuantileTransformer
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import math
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
import keras.backend as K
from tf_siren import SinusodialRepresentationDense
print(dde.__version__)
# dde.config.disable_xla_jit()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class DeepONetCartesianProd(dde.maps.NN):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        self.geoNet = layer_sizes_branch[0]
        self.geoNet2 = layer_sizes_branch[1]

        self.HD = layer_sizes_branch[2]
        self.VEC = layer_sizes_branch[3]
        
        self.outNet = layer_sizes_trunk[0]
        self.outNet2 = layer_sizes_trunk[1]

        self.b = tf.Variable(tf.zeros(1))


    def call(self, inputs, training=False):
        x_func1 = inputs[0] # [ bs , 10 ] , implicit geom
        x_loc = inputs[1] # [ bs , Npt , 4 ] , output coordinates + imp dist


        # Encode implicit geom
        x_func1_1 = self.geoNet(x_func1) # Output: [ bs , hidden dim ]

        # Encode output coordinates
        x_loc_1 = self.outNet(x_loc) # Output: [ bs , Npt , hidden dim ]


        # Mix data 
        mix = tf.einsum("bh,bnh->bnh", x_func1_1 , x_loc_1) # Output: [ bs , Npt , hidden dim ]
        # Max pool?
        # mix_reduced = tf.math.reduce_max( mix , axis=1 ) # Output: [ bs , hidden dim ]
        mix_reduced = tf.math.reduce_mean( mix , axis=1 ) # Output: [ bs , hidden dim ]


        # Further encode
        x_func1 = self.geoNet2(mix_reduced) # Output: [ bs , hidden dim , 4 ]
        x_loc = self.outNet2(mix) # Output: [ bs , Npt , hidden dim ]

        ss = x_func1.shape
        x_func1 = tf.reshape( x_func1 , [ ss[0] , self.HD , self.VEC ] )
        ss = x_loc.shape
        x_loc = tf.reshape( x_loc , [ ss[0] , ss[1] , self.HD , self.VEC ] )

        # Einsum
        x = tf.einsum("bhc,bnhc->bnc", x_func1 , x_loc)

        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        
        return tf.math.sigmoid( x )
        # return 0.5 * ( tf.math.sin( x ) + 1. )
        # return x


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1][indices]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y


########################################################################################################
seed = 2024
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

# Parameters
RUN = 0
N_load_paras = 9 # Number of load parameters
N_comp = 4
HIDDEN = 32
batch_size = 16
fraction_train = 0.8
N_epoch = 600000
data_type = np.float32
N_out_pt = 5000
learning_rate = 2e-3
w0 = 10.
activation = "swish"
scale = 2

sub = '_GDON'+str(N_out_pt)+'_larger_DistSplit'

HIDDEN = int( HIDDEN * scale )
print('\n\nModel parameters:')
print( sub )
print( 'N_comp  ' , N_comp )
print( 'HIDDEN  ' , HIDDEN )
print( 'batch_size  ' , batch_size )
print( 'fraction_train  ' , fraction_train )
print( 'learning_rate  ' , learning_rate )
print( 'w0  ' , w0 )
print( 'activation  ' , activation )
print( 'scale  ' , scale )
print('\n\n\n')


#############################################################################################################
outNet = tf.keras.Sequential(
    [
        InputLayer(input_shape=(None,4,)),
        Dense( int(50*scale) ,activation=activation),
        Dense( int(50*scale) ,activation=activation),
        Dense( HIDDEN ,activation=activation),
                              ]
)
print('\n\noutNet:')
outNet.summary()

outNet2 = tf.keras.Sequential(
    [
        InputLayer(input_shape=(None,HIDDEN,)),
        SinusodialRepresentationDense( HIDDEN*2 ,activation='sine',w0=w0),
        SinusodialRepresentationDense( HIDDEN*2 ,activation='sine',w0=w0),
        SinusodialRepresentationDense( HIDDEN*N_comp ,activation='sine',w0=w0),

                              ]
)
print('\n\noutNet2:')
outNet2.summary()


geoNet = tf.keras.Sequential(
    [
        InputLayer(input_shape=(N_load_paras,)),
        Dense( int(50*scale) ,activation=activation),
        Dense( int(50*scale) ,activation=activation),
        Dense( HIDDEN ,activation=activation),
                              ]
)
print('\n\ngeoNet:')
geoNet.summary()

geoNet2 = tf.keras.Sequential(
    [
        InputLayer(input_shape=(HIDDEN,)),
        Dense( HIDDEN*2 ,activation=activation),
        Dense( HIDDEN*2 ,activation=activation),
        Dense( HIDDEN*N_comp ,activation=activation),
                              ]
)
print('\n\ngeoNet2:')
geoNet2.summary()


#######################################################################################
# Build DeepONet
net = DeepONetCartesianProd(
        [ geoNet , geoNet2 , HIDDEN , N_comp ], [ outNet , outNet2 ] , activation, "Glorot normal")


#######################################################################################
base = '../'
data_raw = np.load(base+'Parameters_valid.npy')[ :2500 , :N_load_paras ] # Geom params


# Outputs
tmp = np.load('./Outputs_rpt'+str(RUN)+'_N'+str(N_out_pt)+'.npz')
output_pos = tmp['a'] # [ 2500 , Npt , 4 ]
output_vals = tmp['b'] # [ 2500 , Npt , 4 ]


# Threshold
output_vals[:,:,0] = np.clip( output_vals[:,:,0] , 0. , 310. ) # vM stress
output_vals[:,:,1] = np.clip( output_vals[:,:,1] , -0.008 , 0.005 ) # UX
output_vals[:,:,2] = np.clip( output_vals[:,:,2] , -0.0005 , 0.02 ) # UY
output_vals[:,:,3] = np.clip( output_vals[:,:,3] , -0.005 , 0.01 ) # UZ


# Scale
scaler_fun = MinMaxScaler

xyzd_scalers = scaler_fun()
ss = output_pos.shape
tmp = output_pos.reshape([ ss[0]*ss[1] , ss[2] ])
xyzd_scalers.fit(tmp)
output_pos = xyzd_scalers.transform( tmp ).reshape(ss)

output_scalers = scaler_fun()
ss = output_vals.shape
tmp = output_vals.reshape([ ss[0]*ss[1] , ss[2] ])
output_scalers.fit(tmp)
output_vals = output_scalers.transform( tmp ).reshape(ss)


para_scaler = scaler_fun()
para_scaler.fit(data_raw)
data_raw = para_scaler.transform( data_raw )

import pickle
pickle.dump(xyzd_scalers, open('xyzd_scaler', 'wb'))
pickle.dump(output_scalers, open('output_scaler', 'wb'))
pickle.dump(para_scaler, open('para_scaler', 'wb'))




#####################################################################################
# Distance split
shifted_paras = data_raw - data_raw[0] # Take the first design as reference
parameter_distance = np.linalg.norm( shifted_paras[:,:-1] , axis=-1 ) # Normalized distance in the parameter space
order = np.argsort( parameter_distance )


# Train / test split
N_valid_case = len(output_vals)
N_train = int( N_valid_case * fraction_train )
train_case = order[ :N_train ] # Nearest ones in training
test_case =  order[ N_train: ] # Far-away ones in testing




# geo
u0_train = data_raw[ train_case , :: ].astype(data_type)
u0_testing = data_raw[ test_case , :: ].astype(data_type)

# Output coords
xyz_train = output_pos[ train_case , :: ].astype(data_type)
xyz_testing = output_pos[ test_case , :: ].astype(data_type)

# Output vals
s_train = output_vals[ train_case , : ].astype(data_type)
s_testing = output_vals[ test_case , : ].astype(data_type)

print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('xyz_train.shape', xyz_train.shape)
print('xyz_testing.shape', xyz_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)


# Pack data
x_train = (u0_train.astype(data_type), xyz_train.astype(data_type))
y_train = s_train.astype(data_type) 
x_test = (u0_testing.astype(data_type), xyz_testing.astype(data_type))
y_test = s_testing.astype(data_type)
data = TripleCartesianProd(x_train, y_train, x_test, y_test)


# Build model
model = dde.Model(data, net)


def inv( data , scaler ):
    ss = data.shape
    tmp = data.reshape([ ss[0]*ss[1] , ss[2] ])
    return scaler.inverse_transform( tmp ).reshape(ss)

def err_L2( true_vals , pred_vals ):
    return np.linalg.norm(true_vals - pred_vals , axis=1 ) / np.linalg.norm( true_vals , axis=1 )

def err_MAE( true_vals , pred_vals ):
    return np.mean( np.abs(true_vals - pred_vals) , axis=1 )


def vm_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,0]
    pred_vals = inv( y_pred , output_scalers )[:,:,0]
    return np.mean( err_L2( true_vals , pred_vals ) )
def vm_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,0]
    pred_vals = inv( y_pred , output_scalers )[:,:,0]
    return np.mean( err_MAE( true_vals , pred_vals ) )


def ux_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,1]
    pred_vals = inv( y_pred , output_scalers )[:,:,1]
    return np.mean( err_L2( true_vals , pred_vals ) )
def ux_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,1]
    pred_vals = inv( y_pred , output_scalers )[:,:,1]
    return np.mean( err_MAE( true_vals , pred_vals ) )


def uy_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,2]
    pred_vals = inv( y_pred , output_scalers )[:,:,2]
    return np.mean( err_L2( true_vals , pred_vals ) )
def uy_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,2]
    pred_vals = inv( y_pred , output_scalers )[:,:,2]
    return np.mean( err_MAE( true_vals , pred_vals ) )


def uz_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,3]
    pred_vals = inv( y_pred , output_scalers )[:,:,3]
    return np.mean( err_L2( true_vals , pred_vals ) )
def uz_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,3]
    pred_vals = inv( y_pred , output_scalers )[:,:,3]
    return np.mean( err_MAE( true_vals , pred_vals ) )


metrics = [ vm_L2,vm_MAE , ux_L2,ux_MAE , uy_L2,uy_MAE , uz_L2,uz_MAE ]


model.compile(
    "adam",
    lr=learning_rate,
    decay=("inverse time", 1, learning_rate/10.),
    # loss=relativeDiff,
    metrics=metrics,
)
losshistory1, train_state1 = model.train(iterations=N_epoch, batch_size=batch_size, model_save_path="./mdls/TrainedModel"+sub)
np.save('losshistory'+sub+'.npy',losshistory1)


####################################################################################################################################
# Evaluation
# On subsamples
y_pred = model.predict(data.test_x)
# Invert
u_pred = inv( y_pred , output_scalers )
u_test = inv( y_test , output_scalers )
err_u_subset = np.zeros( [ len(test_case) , 4 ] )
for cc in range( 4 ):
    err_u_subset[ : , cc ] = err_MAE( u_test[:,:,cc] , u_pred[:,:,cc] )


# On original mesh points
base = '../'
stress_full = np.load(base+'us_valid.npz')
xyzd_full = np.load(base+'xyzd_valid.npz')


err_u_mesh = np.zeros( [ len(test_case) , 4 ] )
predictions = {}
import pyvista as pv
index_map = np.load('../RVE_index_valid.npy')

for idx , tc in enumerate( test_case ):
    input_paras = u0_testing[ idx : (idx+1) ] # Already scaled

    name = 'Valid' + str(tc)
    xyzd = np.expand_dims( xyzd_full[name] , axis=0 )
    outputs = np.expand_dims( stress_full[name] , axis=0 )
    
    # Scale inputs
    xyzd[0] = xyzd_scalers.transform( xyzd[0] )

    # Pack
    x_test = (input_paras.astype(data_type), xyzd.astype(data_type))

    # Predict
    y_pred = model.predict(x_test)

    # Invert
    u_pred = inv( y_pred , output_scalers )

    for cc in range( 4 ):
        err_u_mesh[ idx , cc ] = err_MAE( outputs[:,:,cc] , u_pred[:,:,cc] )[0]
    predictions[ name ] = u_pred[0]


    # Write vtk
    rve_id = index_map[ tc ]
    vtk = pv.read('../vtks/RVE'+str(rve_id)+'.vtk')
    # Write predictions
    vtk['vM_pred'] = u_pred[:,0]
    vtk['u_pred'] = u_pred[:,1:]

    # Write FE results
    my_fe = stress_full[name]
    vtk['vM_fe'] = my_fe[:,0]
    vtk['u_fe'] = my_fe[:,1:]

    vtk.save('./RVE'+str(rve_id)+'_pred.vtk')



np.savez_compressed('Predictions'+sub+'.npz',**predictions)
np.savez_compressed('Errors'+sub+'.npz',a=err_u_subset,b=err_u_mesh)


print('For vM stress:')
print('On subset, MAE: ' , np.mean(err_u_subset[:,0]) )
print('On mesh, MAE: ' , np.mean(err_u_mesh[:,0]) )


print('\nFor UX:')
print('On subset, MAE: ' , np.mean(err_u_subset[:,1]) )
print('On mesh, MAE: ' , np.mean(err_u_mesh[:,1]) )
print('\nFor UY:')
print('On subset, MAE: ' , np.mean(err_u_subset[:,2]) )
print('On mesh, MAE: ' , np.mean(err_u_mesh[:,2]) )
print('\nFor UZ:')
print('On subset, MAE: ' , np.mean(err_u_subset[:,3]) )
print('On mesh, MAE: ' , np.mean(err_u_mesh[:,3]) )