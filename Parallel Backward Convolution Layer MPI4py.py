from mpi4py import MPI
import numpy as np
import cv2 as cv
import time
from IPython.display import Image
from mpi4py.MPI import COMM_WORLD as Comm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class ConvolutionLayer:
     def __init__(self, kernel_num, kernel_size):
        """
        Constructor takes as input the number of kernels and their size. I assume only squared filters of size kernel_size x kernel_size
        """
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        # Generate random filters of shape (kernel_num, kernel_size, kernel_size). Divide by kernel_size^2 for weight normalization
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size**2)
      
     def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        # Extract image height and width
        image_h, image_w = image.shape
        self.image = image
        # The number of patches, given a fxf filter is h-f+1 for height and w-f+1 for width
        for h in range(image_h-self.kernel_size+1):
            for w in range(image_w-self.kernel_size+1):
                patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
                yield patch, h, w
     def forward_prop(self, image):
        """
        Perform forward propagation for the convolutional layer.
        """
        # Extract image height and width
        image_h, image_w = image.shape
        # Initialize the convolution output volume of the correct size
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        #print(convolution_output)
        for patch, h, w in self.patches_generator(image):
            # Perform convolution for each patch
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return convolution_output
     

     def back_prop(self, dE_dY, alpha):
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically max pooling layer.
        It updates the kernels' weights
        """
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Distribute patches to different processes
        local_patches = []
        for patch, h, w in self.patches_generator(self.image):
            if rank == (h*w % size):
                local_patches.append((patch, h, w))
               
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in local_patches:
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]

        
        # Gather results from all processes
        gathered_dk = comm.gather(dE_dk, root=0)
 
        if rank == 0:
            # Update the parameters
            #print("-->> ",gathered_dk)
            dE_dk = np.sum(gathered_dk, axis=0)
            self.kernels -= alpha*dE_dk
     #       print("\n-->> ", dE_dk , "\n")
        
        return dE_dk


image = np.array([[ 0.25057137, -0.71970999, -0.32280236, -0.75119535 , 1.51969909],
 [-0.73966613 , 1.1000746  , 0.70552405 ,-0.66624819, -0.88671778],
 [-0.27693189 , 1.76313974 ,-0.38925585 , 0.34870784 , 0.14532596],
 [-0.89287442 , 0.76516577 , 1.47164109 , 0.41687256 , 2.13189543],
 [-1.57744842 , 1.03911702 , 0.80262811 , 1.20457136 , 0.28729534]])

# Create a ConvolutionLayer object with 2 kernels of size 3x3
conv_layer = ConvolutionLayer(1, 3)

# Perform forward propagation
conv_output = conv_layer.forward_prop(image)

# Compute gradient of the loss function with respect to the output
dE_dY = np.array( [[[ 0.38797414],
  [ 0.19486651],
  [ 0.89915891]],

 [[ 0.18858356],
  [ 0.43961626],
  [-0.32920919]],

 [[-1.21039232],
  [ 1.15138377],
  [ 1.99337296]]])

# Perform backpropagation with a learning rate of 0.1
dE_dk = conv_layer.back_prop(dE_dY, 0.1)
#print("dE_dk ", dE_dk)
if rank == 0:
    #print("Convolution output:")
    print("\nIntersting " ,dE_dk , "\n")
