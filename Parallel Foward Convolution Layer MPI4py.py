from mpi4py import MPI
import numpy as np
import cv2 as cv2
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
        self.kernels = np.array([[[ 0.01777215, -0.03415011, -0.02200378],
        [ 0.04552032 , 0.23696946 , 0.05907649],
        [-0.07219312,  0.07629909 ,-0.03409934]]])
       
    #        print("filter ",self.kernels,"\n")
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
        # MPI initialization
        #comm = MPI.COMM_WORLD
        #rank = comm.Get_rank()
        #size = comm.Get_size()

        # each process computes (H/P) * (W/P) patches and K kernels.
        # time complexity of each process is O((H/P)*(W/P)K)
        # the gather operation takes O(P)
        # total time complexity will be O(HW/PK + P)



        
        # Extract image height and width
        image_h, image_w = image.shape
        # Initialize the convolution output volume of the correct size
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))

        if rank == 0:
            kernels = self.kernels
        else:
            kernels = None
        kernels = comm.bcast(kernels, root=0)

        # Divide the patches among the processes
        patches_list = list(self.patches_generator(image))
        num_patches = len(patches_list)
        patches_per_process = num_patches // size
        start_index = rank * patches_per_process
        end_index = start_index + patches_per_process
       
        if rank == size - 1:
            end_index = num_patches
        local_patches = patches_list[start_index:end_index]
        #print("rank: " , rank , local_patches)
        # Perform convolution for each patch
        #print("self kernels " , kernels)
        for patch, h, w in local_patches:
            convolution_output[h,w] = np.sum(patch*kernels, axis=(1,2))

        # Gather the results from all processes
        convolution_output = comm.gather(convolution_output, root=0)

        if rank == 0:
            # Combine the results to form the final convolution output
            for i in range(1, size):
                convolution_output[0] += convolution_output[i]
            convolution_output = convolution_output[0]

        return convolution_output

     def back_prop(self, dE_dY, alpha):
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically max pooling layer.
        It updates the kernels' weights
        """
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        return dE_dk





cl = ConvolutionLayer(kernel_num=1, kernel_size=3)

# Generate a sample input image
if rank == 0:
    image = cv2.imread('/home/saghichaghi22/dog.jpg', cv2.IMREAD_GRAYSCALE)
    #print(image)
else:
    image = None
image = comm.bcast(image, root=0)

# Perform forward propagation
print("start")
start_time = time.time()
convolution_output = cl.forward_prop(image)
# Print the output
if rank == 0:
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Convolution output:")
    print(convolution_output[0][0])


#if rank == 0:
#tok =  cv.imread('dog.jpg', cv.IMREAD_GRAYSCALE)
#tok = image = np.random.rand(5, 5)
# Generate patches
#a= ConvolutionLayer(1,3)
#a.patches_generator(tok)
#tomi = a.forward_prop_mpi(tok)
#print(tomi)
