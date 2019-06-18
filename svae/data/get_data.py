import pickle
import numpy as np
from sklearn.decomposition import PCA as PCA
from PIL import Image
import math
from . import image_processing as IM
import glob
import sys
from .. distributions import distributions

BSDSloc='/srv/data/data0/gbarello/data/BSDS300/'


####################################################################
## Function to calculate the Gaussian function of a rotated 2D point

def G(x,y,s,r,f):
    # [x,y] are the location, r is the orientation s is the overall 
    # variance and f looks to be the eccentricity
    a = np.cos(r)*x + np.sin(r)*y                                   # Calculate rotation into eccentricity dimension 1
    b = np.cos(r)*y - np.sin(r)*x                                   # Calculate rotation into eccentricity dimension 2 
    return np.exp(-((a/s)**2 + (b/(f*s))**2)/2)                     # Retun the Gaussian function evaluated at that location, given r, s and f

####################################################################
## Function to create a Gaussian receptive field

def gauss_RF(size,c,r,f = 2):
    # size is the length of each patch side in pixels, 
    # c is the center of the RF, r is the orientation, 
    # and f is the eccentricity
    out = np.zeros([size,size])                                     # Initialize the output to zeros
    for i in range(size):                                           # Loop over rows
        for j in range(size):                                       # Loop over columns
            out[i,j] = G(i-c[0],j-c[1],1,r,f)                       # For every point, calculate the Gaussian function value

    return out
            
####################################################################
## Function to create a random dictionary

def rand_dict(patch_size,nvar):
    cen = np.random.uniform(0,patch_size,[nvar,2])                  # Generate a set of random receptive field center locations
    rot = np.random.uniform(0,2*np.pi,[nvar])                       # Generate a random set of receptive field orientations
    D   = np.array([gauss_RF(patch_size,cen[k],rot[k]) for k in range(nvar)])/10 # for each ceenter and rotation, generate an eliptical receptive field
    return D

####################################################################
##

def get_CNN_dat(data,pca,whiten):
    cov = np.reshape(pca.explained_variance_,[1,-1])
    if whiten:
        data = pca.inverse_transform(data/np.sqrt(cov))
    else:
        data = pca.inverse_transform(data)

    return data

####################################################################
## Function to make a synthetic dataset

def make_synthetic_data(dist,patch_size,nvar,ndat = 100000):
    
    try:                                                            # First try to load pre-made synthetic data
        F = open("./datasets/syn_dict_{}_{}".format(patch_size,nvar),"rb") # If it succeeds then the file should open
        D = pickle.load(F)                                          # Load the data and pass it through
        F.close()                                                   # Close the file

    except:                                                         # In case of failure, create some synthetic data and save it
        D = rand_dict(patch_size,nvar)                              # Data includes a random dictionary
        D = np.reshape(D,[nvar,-1])                                 # Reshape the dictionary to be have the features as columns
        
        F = open("./datasets/syn_dict_{}_{}".format(patch_size,nvar),"wb") # Open a file to save the data to
        pickle.dump(D,F)                                            # Save the dictionary to file
        F.close()

    R = dist(ndat,nvar)                                             # ????
    return np.dot(R,D)#[ndat,patch_size**2]                         # Return the dictionary and whatever R is

####################################################################
## Function to get data patches

def get_data(patch_size,nvar,dataset = "BSDS",whiten = True,CNN = False):
    from scipy.ndimage.filters import gaussian_filter as gfilt

    try:
        F = open("./datasets/{}_{}_{}_{}".format(patch_size,nvar,dataset,whiten),"rb")  # Open the file with the data
        dataset = pickle.load(F)                                    # Load the data from the file
        F.close()                                                   # Close the file 
        white,fit_data,fit_var,fit_test = dataset                   # Create ???

    except:
        if dataset == "bruno":                                                      # use the Sparse Coding original images
            data = np.reshape(read_dat("./datasets/bruno_dat.csv"),[512,512,10])    # Load the full dataset from file
            data = np.transpose(data,[2,1,0])                                       # Re-organize the data
            data = np.array([gfilt(i,.5) for i in data])                            # Filter the images with a Gaussian filter (Whitening)
            data = (data + data.min())/(data.max() - data.min())                    # Normalize the data to the min and set the dynamic range to 1. 
            
            data = np.reshape(np.concatenate([IM.split_by_size(d,patch_size) for d in data]),[-1,patch_size*patch_size])
            
        elif dataset == "MNIST":                                  # Select MNIST digits
            data = read_dat("./../../data/MNIST/mnist_train.csv") # MNIST data file 
            lab  = data[:,0]                                      # Extract first index to be in "lab" <-- ??
            data = data[:,1:]                                     # The remainder of the indices are the actual data
            data = np.reshape(data,[-1,28*28])                    # Reshape data into vectors
            data = (data + data.min())/(data.max() - data.min())  # Change the dynamic range to be between 0 and 1
        
        elif dataset == "BSDS":
            imlist = np.squeeze(IM.get_array_data(BSDSloc + "iids_train.txt"))
            data   = [IM.get_filter_samples(BSDSloc + "images/train/" + i + ".jpg",size = patch_size) for i in imlist]
            data   = np.concatenate(data)                           # 
            data   = np.reshape(data,[-1,patch_size*patch_size])    # Reshape the data to be the vecors with size = num pixels
            print("BSDS data size: {}".format(data.shape))          # Print out the size of the data 
            
        else:                                                       # Otherwise make some synthetic data from a given distribution
            f,g,dist = distributions.get_distribution(dataset)      #     
            data     = make_synthetic_data(dist,patch_size,nvar)    #

        LL   = len(data)                                            # Get the number of datapoints
        var  = data[:int(LL/10)]                                    #
        test = data[int(LL/10):int(2*LL/10)]                        #
        data = data[int(2*LL/10):]                                  #
        
        white    = PCA(nvar,copy = True,whiten = whiten)            # Extablish the PCA decomposition 
        fit_data = white.fit_transform(data)                        #
        fit_var  = white.transform(var)                             #
        fit_test = white.transform(test)                            #

        fit_data = np.random.permutation(fit_data)                  #
        fit_var  = np.random.permutation(fit_var)                   #
        fit_test = np.random.permutation(fit_test)                  #
                
        F = open("./datasets/{}_{}_{}_{}".format(patch_size,nvar,dataset,whiten),"wb") # Open a file to save some data in
        pickle.dump([white,fit_data,fit_var,fit_test],F)            # Dump the data into the file
        F.close()                                                   # Close the file that now has the data in it

    if CNN:
        fit_data = get_CNN_dat(fit_data,white,whiten)               #
        fit_var  = get_CNN_dat(fit_var,white,whiten)                #
        fit_test = get_CNN_dat(fit_test,white,whiten)               #
        
    return np.float32(fit_data),np.float32(fit_var),np.float32(fit_test),white # Return...???

####################################################################
## Function to read data

def read_dat(f):
    F   = open(f,"r")                                               # Open the file that was wassed through (f)
    out = []                                                        # Initialize the output to an empty array
    for l in F:                                                     # For all lines in the file F (read in as strings)
        temp     = l.split(",")                                     # Split the text string at every comma ","
        temp[-1] = temp[-1][:-1]                                    # Remove the last index?
        out.append([float(x) for x in temp])                        # Append the values to the full output
    F.close()                                                       # Close the file
    return np.array(out)
        
####################################################################
####################################################################
