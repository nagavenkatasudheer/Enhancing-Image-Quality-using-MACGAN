import sys
from commonfunctions import *
from matplotlib.patches import Rectangle

def sobel_fn(img,threshold):
    hx =  np.array([
        [-1,-2,-1],
        [ 0,0,0],
        [ 1,2,1]
    ])

    hy =  np.array([
        [ -1,0,1],
        [ -2,0,2],
        [ -1,0,1]
    ])
    xImg = convolve2d(img, hx)
    yImg= convolve2d(img, hy)
    
    newImg=np.sqrt(xImg**2+yImg**2) #edge strength
    
    newImg[newImg<threshold]=0
    newImg[newImg>threshold]=1
    
    return newImg

def SunnyImageDetection(img):
    #sobel edge detection
    # getting the height and width of the image to use them to define a threshold 
    width, height=img.shape
    # we use sobel edge detection to know if an image is sunny or not
    step1=sobel_fn(img,0.5)
    isSunny=np.sum(step1)
    result=False
    # we use a threshold as a factor of the images size
    if(isSunny >0.07*width*height):
        # if the image is found to be sunny it should be discarded and does not continue the rest of the processing
        result=True

        
    return result
# our own implimentation of the hough transform
def houghTransform(img):
    step1=sobel_fn(img,0.3)
    width, height=img.shape
    
    # rMax is diagonal distance --> euclidean distance from origin point to end point
    rMax=round(math.dist((0,0),(height-1,width-1)))

    # Range of r is from -Rmax to Rmax
    # Range of theta from -90 to 90
    angles = np.arange(-90,90)
    cosineArray = np.cos(np.deg2rad(angles))
    sineArray = np.sin(np.deg2rad(angles))
    
    # create hough space where R is vertical axis and theta is horizontal
    rows = int(2* rMax)
    cols = len(angles)
    houghSpace = np.zeros((rows,cols))
    
    # Get indices of edge points
    yEdge , xEdge =  np.nonzero(step1)
    
    # Take steps to reduce computations
    for x,y in zip(xEdge, yEdge):
    #  we use a step size  of 2 or more to reduce computation time
        for theta in range(0,len(angles),1):
            # rMax is added to map r (from - Rmax to Rmax) value into hough space (from 0 to 2*Rma) 
            r = round(x*cosineArray[theta] + y*sineArray[theta])+ rMax -1
            if r >= houghSpace.shape[0]:
                continue
            houghSpace[r,theta]+=1
    
        
    return (step1, houghSpace)


# function to get the peak points in the hough space (ta2riban heya dih el 8alat)
def houghPeaks(houghSpace, threshold):
    r,a =  np.nonzero(houghSpace > threshold)
    return r,a

def VanishingPointDetection(t1):
    # getting the hough transform with built in function
    hough_space, angles, distances = hough_line(t1)
    thres = round(0.8* np.max(hough_space))
    # getting the angles and distances at peak points in the hough space
    acumm, a, r = hough_line_peaks(hough_space, angles, distances,thres)

    # drawing the dominating lines in the image and calculating the y value of the intersection (the vanishing point)
    lineImg = np.zeros(t1.shape) 
    fig, ax = plt.subplots()
    ax.imshow(t1)

    # transforming from polar coordinates to cartesian coordinates
    # V0 =alpha*U0 +beta
    aArr=np.zeros(len(r)) 
    bArr=np.zeros(len(r))
    alpha=np.zeros(len(r))
    beta=np.zeros(len(r))

    i = 0
    for  dist, angle in zip(r,a):
        # drawing the dominating lines
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])    
        ax.axline((x0, y0), slope=np.tan(angle + np.pi/2))
        # calculating aplha and beta to get cartisean coordinates
        aArr[i]=np.cos(angle)
        bArr[i]=np.sin(angle)
        alpha[i]= - (aArr[i]/bArr[i])
        beta[i]= (dist/bArr[i])
        i+=1
        
    # solving the 2 equations together   
    u0= np.ceil((beta[1]-beta[0])/(alpha[0]-alpha[1]))
    
    # getting the vanishing point->the point of intersection    
    yVanishing= alpha[0]*u0 + beta[0]
    # print("yVanishing = ",yVanishing)
        
    ax.set_xlim((0,lineImg.shape[1]))
    ax.set_ylim((lineImg.shape[0], 0))
    plt.tight_layout()
    plt.show()

    return yVanishing
# our own implementation of the iterrative thresholding algorithm used to segment the sky from the road
def IterativeThresholding(img):
    
    img=(img).astype('uint8')
    
    numPixels= histogram(img)[0]
    greyLevels=histogram(img)[1]

    totalNumberOfPixels=np.cumsum(numPixels)[-1]
    numOfGreyPerK=0


    auxArray = numPixels * greyLevels
    numOfGreyPerK=np.cumsum(auxArray)[-1]
    Tinit=round(numOfGreyPerK/totalNumberOfPixels)
    a_num=numPixels[greyLevels<Tinit]
    a_glevels=greyLevels[greyLevels<Tinit]
    a_total=np.cumsum(a_num)[-1]

    b_num=numPixels[greyLevels>Tinit]
    b_glevels=greyLevels[greyLevels>Tinit]
    b_total=np.cumsum(b_num)[-1]

    auxArray_a= a_num * a_glevels
    auxArray_b= b_num * b_glevels

    T_a = round(np.cumsum(auxArray_a)[-1]/a_total)
    T_b = round(np.cumsum(auxArray_b)[-1]/b_total)
    T_new=(T_a+T_b)/2

    T_old=Tinit
    while(T_new !=T_old):
        a_num=numPixels[greyLevels<T_new]
        a_glevels=greyLevels[greyLevels<T_new]
        a_total=np.cumsum(a_num)[-1]

        b_num=numPixels[greyLevels>T_new]
        b_glevels=greyLevels[greyLevels>T_new]
        b_total=np.cumsum(b_num)[-1]

        auxArray_a= a_num * a_glevels
        auxArray_b= b_num * b_glevels

        T_a = round(np.cumsum(auxArray_a)[-1]/a_total)
        T_b = round(np.cumsum(auxArray_b)[-1]/b_total)

        T_old=T_new
        T_new=(T_a+T_b)/2
        
    img[img>T_new]=255
    img[img<T_new]=0
    return img

def SkyRoadLimitHeight(t1,out1):
    # for approximation purposes we will get the y coordinate of the intersection between the sky and the road segment at
    # 3 different places
    h,w= t1.shape

    # the 3 values of the x coordinate at which we will calculate
    q_1= w//4
    q_2= w//2
    q_3= (3*w)//4

    # the values of the y coordinates that will be averaged together
    r_1= 0
    r_2= 0
    r_3= 0

    # getting the values of the y coordinates to be averaged with 3 for loops
    for i in range(0,h,1):
        if out1[i][q_1] <  1:
            r_1= i
            break

    for i in range(0,h,1):
        if out1[i][q_2] <  1:
            r_2= i
            break        

    for i in range(0,h,1):
        if out1[i][q_3] <  1:
            r_3= i
            break
            
    # the Y coordinate that represents the line of intersection between the sky and the road
    yAvg = (r_1 + r_2 + r_3)//3
    # print("yAvg= ",yAvg)

    # drawing this line of intersection
    x = [0, w-1]
    y = [yAvg,yAvg] # the y we got after averaging 3 values of the top of the image to the first different pixel
    plt.plot(x, y, color="red", linewidth=3)
    plt.imshow(t1)
    plt.show()

    return yAvg


# Dark Channel Prior Function
# Parameters: Image and the structural element
# Returns: Dark Channel Prior Image
def DarkChannelPrior(img, se):
    # Split the image into its channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # Get the minimum of the channels
    min_channel = np.minimum(np.minimum(r, g), b)
    # Create a structural element
    structural_element = np.ones((se, se), np.uint8)
    # Get the minimum of the image
    darkChannel = erosion(min_channel, structural_element)
    return darkChannel

def AtmosphericLightEstimation(darkChannel, img):
    # Get the size of the image
    size = img.shape[0] * img.shape[1]
    # We are going to pick the top 0.1% brightest pixels in the dark channel according to the research paper
    # I maxed between 1 and the number of pixels in the dark channel because the minimum number of pixels in an image is 1
    totalBrightestPixels = int(max(math.floor(size * 0.001), 1))
    # Reshape the dark channel into a 1D array to be sorted
    darkReshaped = darkChannel.reshape(1, size)
    # We are going to sort the dark channel ascendingly , but return the corresponding indices instead of the values
    darkChannelIndices = darkReshaped.argsort()
    # Reshaped the dark channel indices again for easy slicing (when taking only the brightest pixels)
    darkChannelIndicesReshaped = darkChannelIndices.reshape(size, 1)
    # Get the brightest pixels in the dark channel
    # We are going to take the last 0.1% brightest pixels
    brightestPixels = darkChannelIndicesReshaped[- totalBrightestPixels : :]
    # Reshape the image into a matrix to be used in getting the original RGB values of the brightest pixels
    # It is as if each row is a pixel and each column is a channel
    imgReshaped = img.reshape(size, 3)
    # Get the original RGB values of the brightest pixels
    # We are going to calculate the average of the brightest pixels
    brightestPixelsRGB = imgReshaped[brightestPixels]
    # Get the average of the brightest pixels
    atmosphericLight = np.mean(brightestPixelsRGB, axis=0)
    return atmosphericLight

def TransmissionEstimation(img, atmosphericLight, se):
    # If we remove the fog thoroughly, the image may seem unnatural and we may lose the feeling of depth. 
    # So, we can optionally keep a very small amount of fog for the distant objects by introducing a constant parameter w (0≤ w ≤1)
    w = 0.95
    image = np.zeros(img.shape)
    # Normalize each channel in the image by Atmospheric Light
    for ind in range(0,3):
        image[:,:,ind] = (img[:,:,ind] / (atmosphericLight[0][ind]))
    # Get the dark channel prior of the transmission map and apply the equation:
    # t = 1 - w * DarkChannelPrior(t)
    # Transmission essentially will look opposite to the dark channel picture.
    transmission_map = 1 - w * DarkChannelPrior(image, se)
    return transmission_map

def BilateralPixel(image, i, j, sigma_d, sigma_r):    
    denomenator = 0
    numerator = 0
    # Loop through the neighbouring pixels and calculate their average
    for k in range(i-1, i+2):
        for l in range(j-1, j+2):
            # Get the distance between the pixel at (k, l) and the pixel at (i, j) and divide it by sigma_d^2 (according to the equation)
            term1 = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
            # Get the intensity of the pixel at (k, l) and the pixel at (i, j)
            i1 = image[k, l]
            i2 = image[i, j] 
            # Get the difference between the intensity of the pixel at (k, l) and the pixel at (i, j) and divide it by sigma_r^2 (according to the equation)
            term2 = np.exp(-((i1 - i2)** 2) / sigma_r ** 2)
            denomenator += term1 * term2
            numerator += term1 * term2 * image[k, l]
    # Get the denoised pixel value
    Id = numerator / denomenator
    return Id

def BilateralFilter(image, sigma_d, sigma_r):
    filtered_image = np.zeros(image.shape)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            filtered_image[i, j] = BilateralPixel(image, i, j, sigma_d, sigma_r)
    return filtered_image

# Guided Filter Function
# Parameters: Image, p, r, Epsilon
# p = Transmission Map, r = 60 (size of filter), Epsilon = 0.0001 (from research paper)
# Returns: Guided Filter Image
# The guided filter uses a local linear model as an edge-preserving filter.
# Faster than the bilateral filter.
# Used averaging filter using open cv rather than skimage because it is faster, skimage took over 1 minute to run this cell
# while open cv took less than 1 second
def GuidedFilter(img, p, r, eps):
    # Get the mean of the image and the transmission map
    meanI = cv2.boxFilter(img, cv2.CV_64F, (r, r))
    meanP = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    # Get the mean of the image and the transmission map multiplied together
    meanIp = cv2.boxFilter(img * p, cv2.CV_64F, (r, r))
    # Get the mean of the image squared
    meanII = cv2.boxFilter(img * img, cv2.CV_64F, (r, r))
    # Get the variance of the image
    varI = meanII - meanI * meanI
    # Get the covariance of the image and the transmission map
    covIp = meanIp - meanI * meanP
    # Get the a and b values
    a = covIp / (varI + eps)
    b = meanP - a * meanI
    # Get the mean of a and b
    meanA = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    meanB = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # Final result after applying the guided filter
    q = meanA * img + meanB
    return q

def SoftMatting(img, transmission_map, filter_type):
    # Convert the image to grayscale
    img_gray = rgb2gray(img)
    # Normalize the image
    img_gray = np.float64(img_gray) / 255
    refined_img_bilateral = np.zeros(img_gray.shape)
    
    # Next, we are going to use 3 different types of filters: Gaussian, Bilateral and Guided Filters
    # We are going to use the Gaussian filter to smooth the transmission map, sigma = 15 after trial and error 
    if filter_type == 'Gaussian':
        refined_img = gaussian(transmission_map, 15)
    
    # We are going to use the Bilateral filter to smooth the image
    if filter_type == 'Bilateral':
        refined_img = BilateralFilter(transmission_map, sigma_d=200, sigma_r=200)
    
    # We are going to use the Guided filter to smooth the image
    if filter_type == 'Guided':
        refined_img = GuidedFilter(img_gray, transmission_map, 60, 0.0001)
    
    # We are going to use the built in bilateral filter to smooth the image
    if filter_type == 'Built in Bilateral':
        refined_img = cv2.bilateralFilter(transmission_map.astype(np.float32), 30, 200, 200)

    return refined_img

def RecoverSceneRadiance(img, atmosphericLight, refined_img, t0):
    # We are going to use the equation:
    # J = (I - A) / max(t, t0) + A
    # Where J is the recovered scene radiance, I is the original image, A is the atmospheric light, 
    # t is the refined transmission map and t0 is a small constant, like a lower bound for the transmission map
    recovered_img = np.zeros(img.shape, dtype=np.int64)
    # Get max of the refined transmission map and t0
    max_t = np.maximum(refined_img, t0)
    # Normalize each channel in the image by Atmospheric Light
    for index in range(0,3):
        recovered_img[:,:,index] = ((img[:,:,index] - atmosphericLight[0][index]) / max_t) + atmosphericLight[0][index]
    
    return recovered_img


def white_patch(image, percentile=90):
    """
    White balance image using White patch algorithm
    Parameters
    ----------
    percentile : integer, optional
                  Percentile value to consider as channel maximum
    clip: any value less than 0 becomes zero and any value bigger than 1 is 1

    """
    white_patch_image = img_as_ubyte((image / np.percentile(image,percentile)).clip(0, 1))
    return white_patch_image


def gray_world(image):
    image = image.transpose(2, 0, 1).astype(np.uint32)   # hwc -> chw (channel height width) 
    image[0] = np.minimum(image[0]*(np.average(image[1])/np.average(image[0])),255)
    image[2] = np.minimum(image[2]*(np.average(image[1])/np.average(image[2])),255)
    
    return  image.transpose(1, 2, 0).astype(np.uint8)

def ground_truth(image, x, y, mode='mean'):   
    """
    White balance image using Ground-truth algorithm
    Parameters
    ----------
    x & y : image patch starting dimensions 
    
    mode : mean or max, optional
          Adjust mean or max of each channel to match patch
    """
    image_patch = image[x:x+100,y:y+100]
    
    if mode == 'mean':
        image_gt = ((image * (image_patch.mean() /image.mean(axis=(0,1)))).clip(0, 255).astype(int))
                       
                   
    if mode == 'max':
        image_gt = ((image * 1.0 / image_patch.max(axis=(0,1))).clip(0, 1))
                    
    
    if image.shape[2] == 4:
        image_gt[:,:,3] = 255
    return image_gt