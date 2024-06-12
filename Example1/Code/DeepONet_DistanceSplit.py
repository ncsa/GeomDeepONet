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
# from tf_siren import SinusodialRepresentationDense
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
        self.outNet = layer_sizes_trunk[0]
        self.b = tf.Variable(tf.zeros(1))


    def call(self, inputs, training=False):
        x_func1 = inputs[0] # [ bs , 10 ] , implicit geom
        x_loc = inputs[1] # [ bs , Npt , 3 ] , output coordinates


        # Encode implicit geom
        x_func = self.geoNet(x_func1) # Output: [ bs , hidden dim ]

        # Encode output coordinates
        x_loc = self.outNet(x_loc) # Output: [ bs , Npt , hidden dim ]

        x_loc = tf.expand_dims( x_loc , axis=-1 )

        # Einsum
        x = tf.einsum("bh,bnhc->bnc", x_func , x_loc)

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
N_load_paras = 4 # Number of load parameters
N_comp = 1
HIDDEN = 32
batch_size = 16
fraction_train = 0.8
N_epoch = 150000
data_type = np.float32
N_out_pt = 5000
learning_rate = 2e-3
activation = "swish"

sub = '_DON_Benchmark'+str(N_out_pt)+'_DistSplit'

print('\n\nModel parameters:')
print( sub )
print( 'N_comp  ' , N_comp )
print( 'HIDDEN  ' , HIDDEN )
print( 'batch_size  ' , batch_size )
print( 'fraction_train  ' , fraction_train )
print( 'learning_rate  ' , learning_rate )
print( 'activation  ' , activation )
print('\n\n\n')


#############################################################################################################

outNet = tf.keras.Sequential(
    [
        InputLayer(input_shape=(None,3,)),
        Dense( 50 ,activation=activation),
        Dense( 100 ,activation=activation),
        # Dense( 100 ,activation=activation),
        Dense( 50 ,activation=activation),
        Dense( HIDDEN ,activation=activation),

                              ]
)
print('\n\noutNet:')
outNet.summary()





geoNet = tf.keras.Sequential(
    [
        InputLayer(input_shape=(N_load_paras,)),

        Dense( 50 ,activation=activation),
        Dense( 100 ,activation=activation),
        # Dense( 100 ,activation=activation),
        Dense( 50 ,activation=activation),
        Dense( HIDDEN ,activation=activation),
                              ]
)
print('\n\ngeoNet:')
geoNet.summary()



# exit()

#######################################################################################
# Build DeepONet
net = DeepONetCartesianProd(
        [ geoNet ], [ outNet ] , activation, "Glorot normal")


#######################################################################################
base = '../'
data_raw = np.load(base+'input_params.npy') # Geom params


# Outputs
tmp = np.load('./Outputs_rpt'+str(RUN)+'_N'+str(N_out_pt)+'.npz')
output_pos = tmp['a'][:,:,:3] # [ 1000 , Npt , 3 ]
output_vals = tmp['b'] # [ 1000 , Npt , 1 ]


# Threshold
output_vals[:,:,0] = np.clip( output_vals[:,:,0] , 0. , 4.5e8 ) # vM stress


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

def u_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,0]
    pred_vals = inv( y_pred , output_scalers )[:,:,0]
    return np.mean( err_L2( true_vals , pred_vals ) )

def u_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,0]
    pred_vals = inv( y_pred , output_scalers )[:,:,0]
    return np.mean( err_MAE( true_vals , pred_vals ) )


def vm_L2( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,1]
    pred_vals = inv( y_pred , output_scalers )[:,:,1]
    return np.mean( err_L2( true_vals , pred_vals ) )

def vm_MAE( y_train , y_pred ):
    true_vals = inv( y_train , output_scalers )[:,:,1]
    pred_vals = inv( y_pred , output_scalers )[:,:,1]
    return np.mean( err_MAE( true_vals , pred_vals ) )



if N_comp == 1:
    metrics = [ u_L2 , u_MAE ]
else:
    metrics = [ u_L2 , u_MAE , vm_L2 , vm_MAE ]


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
err_u_subset = err_MAE( u_test[:,:,0] , u_pred[:,:,0] )

# On original mesh points
base = '../'
stress_full = np.load(base+'stress_targets.npz')
xyzd_full = np.load(base+'xyzs.npz')

err_u_mesh = []
predictions = {}
import pyvista as pv
for idx , tc in enumerate( test_case ):
    input_paras = u0_testing[ idx : (idx+1) ] # Already scaled

    name = 'Job_' + str(tc)
    xyzd = np.expand_dims( xyzd_full[name][:,:3] , axis=0 )
    outputs = np.expand_dims( stress_full[name] , axis=0 )
    

    # Scale inputs
    xyzd[0] = xyzd_scalers.transform( xyzd[0] )

    # Pack
    x_test = (input_paras.astype(data_type), xyzd.astype(data_type))

    # Predict
    y_pred = model.predict(x_test)

    # Invert
    u_pred = inv( y_pred , output_scalers )

    err_u_mesh.append( err_MAE( outputs[:,:,0] , u_pred[:,:,0] )[0] )
    predictions[ name ] = u_pred[0]

    # Write vtk
    vtk = pv.read('../vtks/Geom'+str(tc)+'.vtk')
    vtk['fe'] = stress_full[name][:,0]
    vtk['pred'] = u_pred[:,0]
    vtk.save('./Geom'+str(tc)+'_pred.vtk')



np.savez_compressed('Predictions'+sub+'.npz',**predictions)
np.savez_compressed('Errors'+sub+'.npz',a=err_u_subset,b=err_u_mesh)


plt.hist( err_u_subset , bins=25 , color='b' , alpha=0.6 , density=True ) 
plt.hist( err_u_mesh , bins=25 , color='r' , alpha=0.6 , density=True ) 
plt.legend(['Subset' , 'Original mesh points'])
tit = 'test_errors'
plt.title(tit)
plt.savefig(tit+sub+'.pdf')
plt.close()

print('For vM stress:')
print('On subset, MAE: ' , np.mean(err_u_subset) )
print('On mesh, MAE: ' , np.mean(err_u_mesh) )