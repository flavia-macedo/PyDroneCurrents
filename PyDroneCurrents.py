
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

'''
Currents from drone videos

Package developed to obtain surface current fields.



PyDroneCurrents ==> pdc

Packages needed to run the code:
Numpy
Matplotlib
OpenCv
Imageio


Functions:
    pdc.create_struct_georeference(video_name, dt, offset_homewater_Z, file_cal);
    pdc.run_georeference_struct(CamPos_ST, video_name, time_limits, file_cal, dt);
    pdc.get_STCFIT_from_IMG_SEQ(IMG_SEQ);
    pdc.get_w_dispersion_relation(Kx, Ky, Ux, Uy, water_depth, domain);
    pdc.get_dispersion_relation(Spectrum, water_depth, Ux, Uy, w_width);
    pdc.get_spectrum_dispersion_relation(Spectrum, STCFIT);
    pdc.retrieve_power_spectrum(IMG_3D, dx,dy,dt,K_limits, W_limits);
    pdc.run_currents(STCFIT,IMG_SEQ);
    pdc.plot_windows(STCFIT);
    pdc.run_currents(STCFIT,IMG_SEQ);
    pdc.plot_currents_camera(SG_Ux, SG_Uy);
    pdc.rotate_point_3D(x,y,yaw,pitch,roll);
    pdc.georeference_vectors(CamPos_ST,STCFIT, x1, y1, SG_Ux, SG_Uy);
    pdc.plot_georeference_vectors(STCFIT, grid_long_geo, grid_lat_geo, long_geo, lat_geo, Ux_geo, Uy_geo);
    pdc.land_mask(mask, latitude, longitude, Ux, Uy);
    pdc.plot_vectors_mask(points, uv, grid_long_geo, grid_lat_geo, STCFIT);



by Flavia

'''



def create_struct_georeference(video_name, dt, offset_homewater_Z, file_cal):
    '''
    Function created to check the variables inserted in the code and adjust the necessary values to run the code

    Input: 
    video_name = Video to be analized (path complete)
    dt = time between frames 
    offset_homewater_Z = distance between home position and water surface in meters
    file_cal = file calibration with calibration parameters from camera

    Output:
    video_name, dt, offset_homewater_Z, file_cal

    '''

    # Add values if variables is empty
    if video_name == []:
        raise NameError('video_name file is empty') 
        
    elif dt == 0: 
        dt= 0.16
        
    # Check dt =>must be a multiple of 1/frame_rate:
    if video_name != 0:
        v = cv2.VideoCapture(video_name) #Read video
        frame_rate = int(v.get(cv2.CAP_PROP_FRAME_COUNT)) # Get how many frames are in the video
        fps = v.get(cv2.CAP_PROP_FPS) # Acquisition rate (frames per seconds)
        if fps == 0:
            raise ValueError('Variable from video read incorrect, maybe video_name not contain the video file')
        duration = frame_rate/fps #Duration video
        dt_gap_video= 1/fps
    if np.remainder(dt,dt_gap_video) != 0:
        closes_dt = np.around(dt/dt_gap_video)*dt_gap_video
        print(' dt exchange for:' ,str(closes_dt))
        dt = closes_dt

    # Load file calibration
    if file_cal == 0:     
        raise NameError('Calibration file not valid') 
                
    # Create object video
    v = cv2.VideoCapture(video_name) # Read video

    if (np.int32(v.get(cv2.CAP_PROP_FRAME_WIDTH))) != file_cal['size_X']:
        raise ValueError('Size of calibration file and image do not match')

    if (np.int32(v.get(cv2.CAP_PROP_FRAME_HEIGHT))) != file_cal['size_Y']:
        raise ValueError('Size of calibration file and image do not match')
                                
    return video_name, dt, offset_homewater_Z, file_cal


###################################################################################################
###################################################################################################


def run_georeference_struct(CamPos_ST, video_name, time_limits, file_cal, dt):
    '''
    
    Generates the IMG_SEQ structure which contains the sequence of images extracted from the video containing the real image size parameters.
    Extracts the frames and convert to grayscale
    Convert the image grid in distance in meters to camera center 
    
    Input: 
        CamPos_ST =  camera position struct from parameters extrinsic
        video_name = Video to be analized (path complete)
        dt = time between frames 
        offset_homewater_Z = distance between home position and water surface in meters
        file_cal = file calibration with calibration parameters from camera
        time_limits = time of the video to be analyzed in seconds [initial_time, end_time]
   
    
    Output:
      IMG_SEQ:
        IMG = Image 3D (Nx, Ny, Nt).
        gridX = 2D grid X data in meters (Nx,Ny).
        gridY = 2D grid Y data in meters (Nx,Ny).
        dx = X resolution in meters.
        dy = Y resolution in meters.
        dt = time resolution in seconds.
        altitude = altitude to the water surface in meters, (video altitude + offset_home2water_Z).
        pitch: video pitch (for Nadir pitch = -90)
        roll: video roll (for Nadir roll = 0)
        heading: video heading to North (yaw)
        Longitude: video longitude in degrees
        Latitude: video latitude in degrees
    
    '''

    # Naming variables from metadata to be used in the function
    offset_homewater_Z = 0.1
    heading = (CamPos_ST['yaw'])
    pitch = (CamPos_ST['pitch'])
    roll = (CamPos_ST['roll'])
    Height_drone = (CamPos_ST['Height'])
    Longitude = (CamPos_ST['LONGITUDE'])
    Latitude = (CamPos_ST['LATITUDE'])
    pitch = pitch + 90   #Adjusted pitch: The pitch value for Nadir position must be 0 and 90 in the horizon. But DJI drone camera pitch is set the 0 position the horizon and -90 in Nadir.The camera pitch is shifted 90 degres.
    altitude = CamPos_ST['Height'] + offset_homewater_Z #Altitude adjusted
    v = cv2.VideoCapture(video_name) #Read video
    fps = v.get(cv2.CAP_PROP_FPS) # Frames per seconds
    video_ts= np.arange(time_limits[0], time_limits[1], dt) # Arrangement of the video time you want to extract the frames
    video_ts = video_ts * fps

    ## Get Frames ##
    IMG_SEQ1=[]
    
    video = imageio.get_reader(video_name)  # Read frame


    def rgb_to_gray(rgb_frame):
        return np.dot(rgb_frame[...,:3], [0.2989, 0.5870, 0.1140]) # Convert to gray scale



    # Looping to do the process on all frames:
    for j in range(0,len(video_ts)): 
        
        frame = video.get_data(int(video_ts[j]))                            
        image_gray = rgb_to_gray(frame)
              

        if j == 0:  
            
            # Get camera configuration parameters:
            size_X = image_gray.shape[1]
            size_Y = image_gray.shape[0]                
            altitude_after_offset =  (altitude + file_cal['camera_offset_Z'])

            # Get image grid in meters from calibration FOV parameters
            x_vec = np.linspace(altitude_after_offset*np.tan(np.radians(-file_cal['fov_x']/2)), altitude_after_offset*np.tan(np.radians(file_cal['fov_x']/2)), size_X)
            y_vec = np.linspace(altitude_after_offset*np.tan(np.radians(file_cal['fov_y']/2)), altitude_after_offset*np.tan(np.radians(-file_cal['fov_y']/2)), size_Y)
           
            # X is a 2D matrix (Ny,Nx) containing the distance in meters to camera center in horizontal direction (x direction).
            # Y is a 2D matrix (Ny,Nx) containing the distance in meters to camera center in vertical direction (y direction).
            X,Y = np.meshgrid(x_vec,y_vec)

            IMG = image_gray
           
            # Flip Y dimension and permute dimension 1 by 2:
            X_eq = np.transpose(np.flipud(X))
            Y_eq = np.transpose(np.flipud(Y))

            IMG_SEQ2 = {'IMG':np.zeros((X_eq.shape[0],X_eq.shape[1], len(video_ts)), order='C', dtype='int'),
                           'gridX': X_eq, 
                           'gridY': Y_eq, 
                           'dt':dt, 
                           'dx':X_eq[1,0] -X_eq[0,0], 
                           'dy':Y_eq[0,1] - Y_eq[0,0], 
                           'altitude': altitude, 
                           'roll': roll, 
                           'pitch': pitch, 
                           'heading': heading, 
                           'Latitude': Latitude, 
                           'Longitude': Longitude} 
            
            IMG_SEQ1.append(IMG_SEQ2)


        
        
        print('Image', str(j), 'of', str(len(video_ts)))

        IMG_SEQ1[0]['IMG'][:,:,j] = np.transpose(np.flipud(image_gray))

        

    cv2.destroyAllWindows()
    v.release()
    IMG_SEQ = IMG_SEQ1[0]

    return IMG_SEQ


###################################################################################################
###################################################################################################

def get_STCFIT_from_IMG_SEQ(IMG_SEQ):

    '''
    Function used to split frames into subframes

    Input:
      IMG_SEQ:
        IMG = Image 3D (Nx, Ny, Nt).
        gridX = 2D grid X data in meters (Nx,Ny).
        gridY = 2D grid Y data in meters (Nx,Ny).
        dx = X resolution in meters.
        dy = Y resolution in meters.
        dt = time resolution in seconds.
        altitude = altitude to the water surface in meters, (video altitude + offset_home2water_Z).
        pitch: video pitch (for Nadir pitch = -90)
        roll: video roll (for Nadir roll = 0)
        heading: video heading to North (yaw)
        Longitude: video longitude in degrees
        Latitude: video latitude in degrees

    Output:
      STCFIT:
        Generic:
        gridX = 2D X grid (Horizontal) in meters
        gridY = 2D Y grid (Vertical) in meters
        image = image example
        Longitude = Longitude in degrees
        Latitude = Latitude in degrees
        heading = angle to north in degrees
        altitude = altitude in meters
        
        Windows:
        N_fit_windows = number of windows created
        w_corners_dim1 = window coners in x
        w_corners_dim2 = window coners in y
        sq_size_m = square size window in meter
        sq_dist_m = square distance in meter between windows
        average_depth_win = average depth water in meters

        fit_param:
        Ux_FG_2D = Ux first guess matrix
        Uy_FG_2D = Uy first guess matrix
        Ux_SG_2D = Ux second guess matrix
        UY_SG_2D = Uy second guess matrix
        w_width_FG: first guess filter width in w [rad/s] 
        w_width_SG: second guess filter width in w [rad/s] 
        waveLength_limits_m: wavelength in meters [min, max]
        wavePeriod_limits_sec: wave period in seconds [min, max]
        K_limits: wave number in rad/m [min, max]
        W_limits: wave frequency in rad/sec [min, max]


    '''
    # Image size
    size_IMG_SEQ = np.shape(IMG_SEQ['IMG'])

    # Square size in meter for the current fit windows. Set 'sq_size_m' to fit 10 squares in dim1 (x):
    sq_size_m = ((np.ndarray.max(IMG_SEQ['gridX'])) - (np.ndarray.min(IMG_SEQ['gridX']))) / 10.2

    # Square distance in meter between current fit windows:
    sq_dist_m = sq_size_m

    # Depth water
    water_depth_mask_2D = 10
    
    #  Ux current interval for the first guess [m/s] [min, max]
    Ux_limits_FG = [-1.5 , 1.6] 
    # Ux current interval for the first guess [m/s]  [min, max]
    Uy_limits_FG = [-1.5 , 1.6]

    # Resolution in [m/s] for the first guess vector:
    U_FG_res = 0.1

    # Filter with in rad/s for first guess:
    w_width_FG = 1

    # Resolution in [m/s] for the second guess vector:
    U_SG_res = 0.025

    # Filter with in rad/s for second guess:
    w_width_SG = w_width_FG / 2

    # Value pixel size in meters
    pixel_size = np.sqrt(((IMG_SEQ['dx']) **2) + ((IMG_SEQ['dy'])**2))
    
    # Wave-lenght limits in meters, for the dispersion relation fit. [min, max]
    waveLength_limits_m = [(4 * pixel_size), (sq_size_m /4)]
    
    # Wave period limits in seconds, for the dispersion relation fit. [min, max]
    wavePeriod_limits_sec = [(IMG_SEQ['dt']), (IMG_SEQ['dt'] * size_IMG_SEQ[2])]

    ## obtain windows fit corners according to 'sq_size_m' and 'sq_dist_m':
    sq_size_pix = sq_size_m/ IMG_SEQ['dx']  # Size value of a square side in pixels
    sq_dist_pix = sq_dist_m/IMG_SEQ['dx']

    ## Creating starting and end point windows in dimension X:
    aux = np.arange(1,size_IMG_SEQ[0],sq_size_pix)  # Array that contains the boundaries of each square in pixels
    offset_center_dim1 = (size_IMG_SEQ[0] - aux[-1]) /2 # Pixel value left at the end after clipping the squares
    # The offset_center_dim1 value is divided by two to be placed on each side of the image and center the squares region
    center_dim1 = np.arange((offset_center_dim1 -(sq_dist_pix/2)),size_IMG_SEQ[0],sq_dist_pix) # Centering the image 
    ini_dim1 = np.around(center_dim1 - (sq_size_pix/2))
    end_dim1 = np.around(center_dim1 + (sq_size_pix/2) - 1)
    center_dim1 = np.around(center_dim1)

    # Delete values out of the IMG_SEQ bounds (dim1)
    center_dim1 = center_dim1[1:]
    ini_dim1 = ini_dim1[1:]
    end_dim1= end_dim1[1:]
    end_dim1[4]=end_dim1[4]+1  

    # Creating starting and end point windows in dimension Y
    aux = np.arange(1,size_IMG_SEQ[1],sq_size_pix)  
    offset_center_dim2 = (size_IMG_SEQ[1]-aux[-1])/2               
    center_dim2 = np.arange(offset_center_dim2 -(sq_dist_pix/2),size_IMG_SEQ[1],sq_dist_pix) 
    ini_dim2 = np.around(center_dim2 - (sq_size_pix/2))
    end_dim2 = np.around(center_dim2 + (sq_size_pix/2 ) -1)
    center_dim2 = np.around(center_dim2)

    # Delete values out of the IMG_SEQ bounds (dim2)
    center_dim2= center_dim2[1:]
    center_dim2[2] = center_dim2[2]+1  
    ini_dim2 = ini_dim2[1:]
    end_dim2 = end_dim2[1:]

    ## Generate all the posible center combinations:
    [center_dim1_2D,center_dim2_2D] = np.meshgrid(center_dim1,center_dim2, indexing='ij')
    [ini_dim1_2D,ini_dim2_2D] = np.meshgrid(ini_dim1,ini_dim2, indexing='ij')
    [end_dim1_2D,end_dim2_2D] = np.meshgrid(end_dim1,end_dim2, indexing='ij')
    N_fit_windows = np.size(center_dim1_2D)

    ## Get the size of each subframe:
    w_corners_dim1 = np.array([np.reshape(center_dim1_2D,( N_fit_windows), order='F'),np.reshape(ini_dim1_2D, ( N_fit_windows), order='F'),np.reshape(end_dim1_2D,( N_fit_windows), order='F')])                                                  
    w_corners_dim2 =  np.array([np.reshape(center_dim2_2D,( N_fit_windows),order='F'),np.reshape(ini_dim2_2D, ( N_fit_windows), order='F'),np.reshape(end_dim2_2D,( N_fit_windows),order='F')])
    size_w_corners_dim1 = np.shape(w_corners_dim1)

    ## Generate the data sctruct:
    # General image information:
    image = IMG_SEQ['IMG']
    Generic = {'gridX': IMG_SEQ['gridX'], 'gridY':IMG_SEQ['gridY'], 'image': image, 'Longitude': IMG_SEQ ['Longitude'], 'Latitude':IMG_SEQ['Latitude'], 'heading': IMG_SEQ['heading'], 'altitude': IMG_SEQ['altitude']}

    # Generate windows struct
    Windows = {'N_fit_windows': N_fit_windows,'w_corners_dim1':w_corners_dim1, 'w_corners_dim2':w_corners_dim2,'sq_size_m': sq_size_m, 'sq_dist_m':sq_dist_m, 'average_depth_win':water_depth_mask_2D}

    # Generate current fit struct
    # Create first guess matrix (containing speed limits):
    Ux_FG_vec =  np.arange(Ux_limits_FG[0],Ux_limits_FG[1], U_FG_res)
    Uy_FG_vec =  np.arange(Uy_limits_FG[0],Uy_limits_FG[1], U_FG_res)
    [Ux_FG_2D,Uy_FG_2D] = np.meshgrid(Ux_FG_vec,Uy_FG_vec, indexing='ij')

    # Create second guess matrix:
    Ux_SG_vec =  np.arange(-U_FG_res,U_FG_res+U_SG_res,U_SG_res)  
    Uy_SG_vec =  np.arange(-U_FG_res,U_FG_res+U_SG_res,U_SG_res)
    [Ux_SG_2D,Uy_SG_2D] = np.meshgrid(Ux_SG_vec, Uy_SG_vec, indexing='ij')

    # Generate wave number(K) and radial frequency (W):
    K_limits = [2*np.pi/waveLength_limits_m[1], 2*np.pi/waveLength_limits_m[0]]
    W_limits = [2*np.pi/wavePeriod_limits_sec[1], 2*np.pi/wavePeriod_limits_sec[0]]

    # Generate parameters fit sctruct:
    fit_param = {'Ux_FG_2D':Ux_FG_2D,'Uy_FG_2D':Uy_FG_2D,'Ux_SG_2D':Ux_SG_2D,'Uy_SG_2D':Uy_SG_2D, 'w_width_FG':w_width_FG,'w_width_SG':w_width_SG,'waveLength_limits_m':waveLength_limits_m,'wavePeriod_limits_sec': wavePeriod_limits_sec,'K_limits':K_limits,'W_limits':W_limits}

    out_fit = []

    STCFIT = {'Generic':Generic,'Windows':Windows,'fit_param':fit_param,'out_fit':out_fit}

    return STCFIT

###################################################################################################
###################################################################################################

def get_w_dispersion_relation(Kx, Ky, Ux, Uy, water_depth, domain):

    '''
    Retrieve w from dispersion relation:

              w =  sqrt(9.80665 .* K .* tanh(K .* water_depth)) + (Kx .* Ux) + (Ky .* Uy)
    
              
    Input:
        Kx = wave number in rad/meter
        Ky = wave number in rad/meter
        Ux = vector current in x
        Uy = vector currenr in y
        water_depth = water depth in meters
        domain = +1 positive or -1 negative

    Output:
        w_adjusted = radial frequency from fundamental mode in rad/m

    '''
   
    K = np.sqrt(Kx**2 +  Ky**2)
    W_adjusted=  (domain* np.sqrt(9.80665 * K * np.tanh(K * water_depth))) + (Kx * Ux) + (Ky * Uy)
    
    return W_adjusted

###################################################################################################
###################################################################################################

def get_dispersion_relation(Spectrum, water_depth, Ux, Uy, w_width):

    '''

    Function that returns a mask corresponding to the valid values of the parameters K and W.

    Input:
       Spectrum = Spectrum structure from retrieve_power_spectrum
       water_depth = water depth in meters
       Ux, Uy = vector u and v corresponding to speed in meters/seconds
       w_width = filter width in w

    Output:   
        DS_3D_mask = logic mask with the same size tham power_Spectrum, where is set to TRUE the areas corresponding to the dispersion relation
    '''
    # Creates a mask matrix as W size output with false values
    DS_3D_mask = np.full(np.shape(Spectrum['W_3D']), False)
    
    Kx = Spectrum['Kx_3D'][:,:,0]
    Ky = Spectrum['Ky_3D'][:,:,0]
    
    ######## Application function 'get_w_dispersion_relation' #######
    W_adjusted = get_w_dispersion_relation(Kx, Ky, Ux, Uy, water_depth, 1)

    # Return W according filter width
    wMin_2D =  W_adjusted - w_width
    wMax_2D =  W_adjusted  + w_width

    # Create a 3D matriz for 'w'
    wMin_3D = np.dstack([wMin_2D]*(np.size(DS_3D_mask[0,0,:])))
    wMax_3D = np.dstack([wMax_2D]*(np.size(DS_3D_mask[0,0,:])))

    DS_3D_mask[(Spectrum['W_3D']>=wMin_3D) & (Spectrum['W_3D']<=wMax_3D)] = True

    if (W_adjusted[:]<0).any:

        ######## Application function 'get_w_dispersion_relation' #######
        W_adjusted = get_w_dispersion_relation(Kx, Ky, Ux, Uy, water_depth, -1)

        wMin_2D = W_adjusted  - w_width
        wMax_2D = W_adjusted  + w_width

        wMin_3D = np.dstack([wMin_2D]*(np.size(DS_3D_mask[0,0,:])))
        wMax_3D = np.dstack([wMax_2D]*(np.size(DS_3D_mask[0,0,:])))

        DS_3D_mask[(Spectrum['W_3D']>=wMin_3D) & (Spectrum['W_3D']<=wMax_3D)] = True
        
    return DS_3D_mask

###################################################################################################
###################################################################################################


def get_spectrum_dispersion_relation(Spectrum, STCFIT):
    '''
    Function that returns valid velocity values
    
    Input:
       Spectrum
       STCFIT

    Output:
       FG_fit and SG_fit that contains the velocity vectors from first and second guess, and the values corresponding signal and noise.
         
            
 
    '''

    ############# FIRST GUESS #######################
   
    Ux_FG_vec = (STCFIT['fit_param']['Ux_FG_2D']).flatten(order='F')
    Uy_FG_vec = (STCFIT['fit_param']['Uy_FG_2D']).flatten(order='F')
    w_width =  STCFIT['fit_param']['w_width_FG']
       
    signal_vec = np.empty(np.shape(Ux_FG_vec))
    signal_vec[:]=np.NaN

    signal_nvalues = np.empty(np.shape(Ux_FG_vec))
    signal_nvalues[:]=np.NaN

    noise_vec = np.empty(np.shape(Ux_FG_vec))
    noise_vec[:]=np.NaN
    
    noise_nvalues = np.empty(np.shape(Ux_FG_vec))
    noise_nvalues[:]=np.NaN

    water_depth=STCFIT['Windows']['average_depth_win']
    
    
    for j in range(len(Ux_FG_vec)):
               
        #### Application function 'get_dispersion_relation' ####
        DS_3D_mask = get_dispersion_relation(Spectrum, water_depth, Ux_FG_vec[j], Uy_FG_vec[j], w_width)
          
        # Get signal-to-noise ratio
        signal = Spectrum['power_Spectrum'][DS_3D_mask]
        noise = Spectrum['power_Spectrum'][DS_3D_mask==0]
          
        signal_vec[j]= np.nansum(signal)       
        noise_vec[j] = np.nansum(noise)

        signal_nvalues[j]= np.sum(np.isfinite(signal))
        noise_nvalues[j] = np.sum(np.isfinite(noise))
        
    # Reshape output vectors
    signal_FG_2D = np.reshape(signal_vec,(np.shape(STCFIT['fit_param']['Ux_FG_2D'])), order="F").copy()
    noise_FG_2D = np.reshape(noise_vec, (np.shape(STCFIT['fit_param']['Ux_FG_2D'])), order ='F').copy()
    
    signal_nvalues_2D = np.reshape(signal_nvalues,(np.shape(STCFIT['fit_param']['Ux_FG_2D'])), order = 'F').copy()
    noise_nvalues_2D = np.reshape(noise_nvalues,(np.shape(STCFIT['fit_param']['Ux_FG_2D'])), order= "F").copy()
                                                          
    # Get signal-to-noise ratio
    SNR_FG =  signal_FG_2D/noise_FG_2D
    SNR_density_FG =  (signal_FG_2D/ signal_nvalues_2D) / (noise_FG_2D / noise_nvalues_2D)
    
    SNR_2D=SNR_density_FG
    Ux_2D=np.round((STCFIT['fit_param']['Ux_FG_2D']),2)
    Uy_2D=np.round((STCFIT['fit_param']['Uy_FG_2D']),2)
    
    SNR_max = np.nanmax(SNR_2D)
    lin_ind = np.unravel_index(np.nanargmax(SNR_2D, axis=None), SNR_2D.shape)
    
    Ux_fit_FG = Ux_2D[lin_ind]
    Uy_fit_FG = Uy_2D[lin_ind]
   
    SNR_FG_max = SNR_FG[lin_ind]
    
    SNR_density_FG_max = SNR_2D[lin_ind]

    # Show results of First Guess
    print(['Ux_Fg: ', str(Ux_fit_FG) ,' m/s',   'Uy_Fg: ', str(Uy_fit_FG), ' m/s,   SNR density: ' ,str(SNR_density_FG_max) ])

    # Create first guess sctructure
    FG_fit = {'signal_2D':signal_FG_2D,'noise_2D':noise_FG_2D,'signal_nvalues_2D':signal_nvalues_2D,'noise_nvalues_2D':noise_nvalues_2D,'SNR_2D':SNR_FG,'SNR_density_2D':SNR_2D,'SNR_FG_max':SNR_FG_max,'SNR_density_FG_max': SNR_density_FG_max,'Ux_2D':STCFIT['fit_param']['Ux_FG_2D'],'Uy_2D':STCFIT['fit_param']['Uy_FG_2D'],'Ux_fit':Ux_fit_FG,'Uy_fit':Uy_fit_FG}
    
    
    ############# SECOND GUESS #######################
    
    Ux_SG_2D = (STCFIT['fit_param']['Ux_SG_2D']) + (Ux_fit_FG)  
    Uy_SG_2D =(STCFIT['fit_param']['Uy_SG_2D'])+(Uy_fit_FG)
    
    Ux_SG_vec = Ux_SG_2D.flatten(order='F')
    Uy_SG_vec = Uy_SG_2D.flatten(order='F')
    w_width_S =  STCFIT['fit_param']['w_width_SG']
       
    signal_vec = np.empty(np.shape(Ux_SG_vec))
    signal_vec[:]=np.NaN

    signal_nvalues = np.empty(np.shape(Ux_SG_vec))
    signal_nvalues[:]=np.NaN

    noise_vec = np.empty(np.shape(Ux_SG_vec))
    noise_vec[:]=np.NaN
    
    noise_nvalues = np.empty(np.shape(Ux_SG_vec))
    noise_nvalues[:]=np.NaN

    water_depth=STCFIT['Windows']['average_depth_win']
     
    for j in range(len(Ux_SG_vec)):
        
        #### Application function 'get_dispersion_relation' ####
        DS_3D_mask = get_dispersion_relation(Spectrum, water_depth, Ux_SG_vec[j], Uy_SG_vec[j], w_width_S)
        
        # Get signal-to-noise ratio
        signal = Spectrum['power_Spectrum'][DS_3D_mask]
        noise = Spectrum['power_Spectrum'][DS_3D_mask==0]
          
        signal_vec[j]= np.nansum(signal)       
        noise_vec[j] = np.nansum(noise)

        signal_nvalues[j]= np.sum(np.isfinite(signal))
        noise_nvalues[j] = np.sum(np.isfinite(noise))
    
    # Reshape output vectors
    signal_SG_2D = np.reshape(signal_vec,(np.shape(Ux_SG_2D)), order="F")
    noise_SG_2D = np.reshape(noise_vec, (np.shape(Ux_SG_2D)), order ='F')
    
    signal_nvalues_2D = np.reshape(signal_nvalues,(np.shape(Ux_SG_2D)), order = 'F')
    noise_nvalues_2D = np.reshape(noise_nvalues,(np.shape(Ux_SG_2D)), order= "F")
                                                          
    # Get signal-to-noise ratio
    SNR_SG =  signal_SG_2D/noise_SG_2D
    SNR_density_SG =  (signal_SG_2D/ signal_nvalues_2D) / (noise_SG_2D / noise_nvalues_2D)
    
    SNR_2D_SG=SNR_density_SG
    
    SNR_max_SG = np.nanmax(SNR_2D_SG)
    lin_ind = np.unravel_index(np.nanargmax(SNR_2D_SG, axis=None), SNR_2D_SG.shape)
   
    Ux_fit_SG =  Ux_SG_2D[lin_ind]
    Uy_fit_SG = Uy_SG_2D[lin_ind]   
    SNR_SG_max = SNR_SG[lin_ind]   
    SNR_density_SG_max = SNR_2D_SG[lin_ind]

    # Show results of Second Guess
    print(['Ux_Sg: ', str(Ux_fit_SG) ,' m/s',   'Uy_Sg: ', str(Uy_fit_SG), ' m/s,   SNR density: ' ,str(SNR_density_SG_max) ])

    # Create second guess structure
    SG_fit = {'signal_2D':signal_SG_2D,'noise_2D':noise_SG_2D,'signal_nvalues_2D':signal_nvalues_2D,'noise_nvalues_2D':noise_nvalues_2D,'SNR_2D':SNR_SG,'SNR_density_2D':SNR_2D_SG,'SNR_SG_max':SNR_SG_max,'SNR_density_SG_max': SNR_density_SG_max, 'Ux_fit_SG':Ux_fit_SG,'Uy_fit_SG':Uy_fit_SG,'Ux_2D':STCFIT['fit_param']['Ux_SG_2D'],'Uy_2D':STCFIT['fit_param']['Uy_SG_2D']}
    
    return FG_fit, SG_fit
    


#############################################################################################################################
#############################################################################################################################

def retrieve_power_spectrum(IMG_3D, dx,dy,dt,K_limits, W_limits): 
    
    '''  
    Function that generates a 3D structure with spectral energy according to the limits of K and W.
    Get power spectrum from FFT (Fast Fourier Transform). 
    For more information:
    https://en.wikipedia.org/wiki/Fast_Fourier_transform
    https://www.youtube.com/watch?v=spUNpyF58BY&t=937s

    Input:
        IMG_3D = 3D array with image sequence
        dx = x resolution in meters 
        dy = y resolution in meters 
        dt = time resolution in seconds
        K_limits = Wave-number in rad/m [min, max]
        W_limits =  frequency in rad/sec [min, max] 

    Output:
        Spectrum:
        power_spectrum: spectral energy
        Kx_3D: 3D Kx grid corresponding to power_Spectrum in rad/m
        Ky_3D: 3D Ky grid corresponding to power_Spectrum in rad/m
        W_3D: 3D W grid corresponding to power_Spectrum in rad/sec
        dKx: Kx resolution [rad/m]
        dKy: Ky resolution [rad/m]
        dW: W resolution [rad/sec]
        Kx_orig_limits = Kx limits in rad/m
        Ky_orig_limits = Ky limits in rad/m
        W_orig_limits = w limits in rad/sec 

'''

    [Nx, Ny, Nt] = np.shape(IMG_3D) 
    
    # Get axis Kx, Ky e W from image:
   
    Kx = (2 * np.pi * 1/dx/Nx) * np.arange((- np.ceil((Nx-1)/2)), (np.floor((Nx-1)/2)+1))
    Ky = (2 * np.pi * 1/dy/Ny) * np.arange((- np.ceil((Ny-1)/2)), (np.floor((Ny-1)/2))+1)
    w  = (2 * np.pi * 1/dt/Nt) * np.arange((- np.ceil((Nt-1)/2)), (np.floor((Nt-1)/2))+1)

    dKx = Kx[1]-Kx[0]
    dKy = Ky[1]-Ky[0]
    dW = w[1]-w[0]
    
    # Normalization
    Norm = dKx * dKy * dW
    
    # FFT application:    
    Spectrum_raw = np.fft.fftshift((np.fft.fftn(IMG_3D))/np.size(IMG_3D))
    power_Spectrum = np.abs(Spectrum_raw/Norm) ** 2
    
    # Get limit index of K
    # Get the values only within the bounds of K and W
    # Generates a matrix of the size of Kx, with false and true indicating which values are inside
    ind_x = np.abs(Kx) <= K_limits[1]
    ind_y = np.abs(Ky) <= K_limits[1]
    ind_w = (w>= W_limits[0]) & (w <= W_limits[1])
    
    # Create 3D structure for Kx, Ky e W
    [Kx_3D,Ky_3D,W_3D] = np.meshgrid([Kx[ind_x]],[Ky[ind_y]],[w[ind_w]], indexing= 'ij')

    ## Takes the spectral energy of only the values within the limit of K and W
    power_Spectrum_cut = power_Spectrum[:,:,ind_w]
    power_Spectrum_cut = power_Spectrum_cut[ind_x,:,:]
    power_Spectrum_cut = power_Spectrum_cut[:,ind_y,:]

    # K magnitude 
    K_3D = np.sqrt(Kx_3D**2 + Ky_3D**2)

    # Cuts the spectrum only the energy values that correspond within the limits of K
    power_Spectrum_cut[np.logical_or(K_3D<K_limits[0], K_3D>K_limits[1])] = np.nan   
    
    # News limits of Kx , Ky and W
    Kx_orig_limits = [Kx[0], Kx[-1]]
    Ky_orig_limits = [Ky[0], Ky[-1]]
    W_orig_limits = [w[0], w[-1]]   

    power_Spectrum_cut = power_Spectrum_cut / np.nansum(power_Spectrum_cut[:])
    
    # Creates a dictionary containing the information created from Spectrum:
    Spectrum = {'power_Spectrum':power_Spectrum_cut,'Kx_3D':Kx_3D,'Ky_3D':Ky_3D,'W_3D':W_3D,'dKx':dKx,'dKy':dKy,'dW':dW,'Kx_orig_limits':Kx_orig_limits,'Ky_orig_limits':Ky_orig_limits,'W_orig_limits':W_orig_limits}

    return Spectrum


###################################################################################################
###################################################################################################


def plot_windows(STCFIT):
    '''
    Function creates for plot windows that will run currents
    
    Input:
       STCFIT
       
    Output:
       Figure contain the windows
       
       '''

    ## Get center of window:
    IND_center = np.ravel_multi_index([np.int32(STCFIT['Windows']['w_corners_dim1'][0,:]), np.int32(STCFIT['Windows']['w_corners_dim2'][0,:])], np.shape(STCFIT['Generic']['gridX']), order='F')
    centers_x1 = STCFIT['Generic']['gridX'].flatten(order='F')
    centers_y1 = STCFIT['Generic']['gridY'].flatten(order='F')
    centers_x = centers_x1[IND_center]
    centers_y= centers_y1[IND_center]
    gridX = STCFIT['Generic']['gridX']
    gridY = STCFIT['Generic']['gridY']
   
    ## Plot windows:    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.pcolor(gridX, gridY,  STCFIT['Generic']['image'][:,:,0], cmap='gray') #Plot image
    ax.axis('equal')
    ax.plot(centers_x,centers_y, '.b') #Plot center point of window
    
    offset_x_text = (STCFIT['Windows']['sq_size_m'])/8


    for i in range(0,STCFIT['Windows']['N_fit_windows']):
        
        # Plot square that represent windows
        x_ini =STCFIT['Generic']['gridX'][np.int32(STCFIT['Windows']['w_corners_dim1'][1,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][1,i])]
        x_end =STCFIT['Generic']['gridX'][np.int32(STCFIT['Windows']['w_corners_dim1'][2,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][2,i])]
        y_ini =STCFIT['Generic']['gridY'][np.int32(STCFIT['Windows']['w_corners_dim1'][1,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][1,i])]
        y_end =STCFIT['Generic']['gridY'][np.int32(STCFIT['Windows']['w_corners_dim1'][2,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][2,i])]
        ax.plot((x_ini, x_ini, x_end, x_end, x_ini), (y_ini, y_end, y_end, y_ini ,y_ini), '-b',)

        ## Plota number of window
        x_text = STCFIT['Generic']['gridX'][np.int32(STCFIT['Windows']['w_corners_dim1'][0,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][0,i])] + offset_x_text
        y_text = STCFIT['Generic']['gridY'][np.int32(STCFIT['Windows']['w_corners_dim1'][0,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][0,i])]
        ax.text(x_text,y_text,str(i))
        
    return fig


###################################################################################################
###################################################################################################


def run_currents(STCFIT,IMG_SEQ):

    '''
    Run FFT and wave dispersion relation current fit

    Input:
       STCFIT
       IMG_SEQ

    Output:

       FG_Ux = Vector u first guess
       FG_Uy = Vector v first guess
       SG_Ux = Vector u second guess
       SG_Uy = Vector v second guess
       STCFIT

    '''

    N_fit_windows = STCFIT['Windows']['N_fit_windows']
    IMG= IMG_SEQ['IMG']

    FG_Ux = []
    FG_Uy = []
    SG_Ux = []
    SG_Uy = []
    FG_fit1=[]
    SG_fit1=[]

    for i in range(N_fit_windows):
        print('Window', str(i), 'of', str(np.int32(STCFIT['Windows']['N_fit_windows'])))
        
        # Cuts the image into windows of the size defined in the previous step, generating subframes:
        IMG_SEQ_Window = IMG[np.int32(STCFIT['Windows']['w_corners_dim1'][1,i])-1:np.int32(STCFIT['Windows']['w_corners_dim1'][2,i]),np.int32(STCFIT['Windows']['w_corners_dim2'][1,i])-1:np.int32(STCFIT['Windows']['w_corners_dim2'][2,i]),:]
        
        ##### Application function 'retrieve_power_spectrum' ####    
        Spectrum = retrieve_power_spectrum(IMG_SEQ_Window, IMG_SEQ['dx'],IMG_SEQ['dy'],IMG_SEQ['dt'], STCFIT['fit_param']['K_limits'],STCFIT['fit_param']['W_limits'])
        
        ##### Application function 'get_spectrum_dispersion_relation' ####
        FG_fit, SG_fit = get_spectrum_dispersion_relation(Spectrum, STCFIT)
        
        Ux_fg = -FG_fit['Ux_fit']
        Uy_fg = -FG_fit['Uy_fit']
        Ux_sg = -SG_fit['Ux_fit_SG']
        Uy_sg = -SG_fit['Uy_fit_SG']

        FG_Ux.append(Ux_fg)
        FG_Uy.append(Ux_fg)
        SG_Ux.append(Ux_sg)
        SG_Uy.append(Uy_sg)

        FG_fit1.append(FG_fit)
        SG_fit1.append(SG_fit)
        

    out_fit_i = {'FG_fit':FG_fit1,'SG_fit':SG_fit1}    
    STCFIT['out_fit']= out_fit_i   

    return FG_Ux, FG_Uy, SG_Ux, SG_Uy,STCFIT          


###################################################################################################
###################################################################################################

def plot_currents_camera(SG_Ux, SG_Uy, STCFIT, CamPos): 
    '''
    Function to plot the velocity vectors according to the camera reference. 

    Input:
       SG_Ux: Velocity vector in x orientation.
       SG_Uy: Velocity vector in y orientation.
       STCFIT
       CamPos

    Output:
       Figure with image and vectors

    '''    
    # Retrieve currents from second fit
    Ux = np.array(SG_Ux)
    Uy = np.array(SG_Uy)

    # Change current direction
    Ux = Ux
    Uy = Uy

    # Get camera position
    IND_center = np.ravel_multi_index((np.int32(STCFIT['Windows']['w_corners_dim1'][0,:]),np.int32(STCFIT['Windows']['w_corners_dim2'][0,:])),np.shape(STCFIT['Generic']['gridX']),order='F')

    x1=(STCFIT['Generic']['gridX']).flatten(order='F')[IND_center]
    y1=(STCFIT['Generic']['gridY']).flatten(order='F')[IND_center]

    # Get camera structure
    Camera_currents = {'x':x1,'y':y1,'Ux':Ux,'Uy':Uy}

    ### Plot vectors in camera grid
    gridX = STCFIT['Generic']['gridX']
    gridY = STCFIT['Generic']['gridY']

    fig, ax = plt.subplots(figsize=(9,8))

    plt.pcolor(gridX, gridY,  STCFIT['Generic']['image'][:,:,0], cmap='gray')
    uv = ax.quiver(x1,y1, Ux, Uy, color='y',scale=10,headwidth=2, headaxislength=4, headlength=4)

    ax.set_xlabel('X distance from camera center [meters]')
    ax.set_ylabel('Y distance from camera center [meters]')
    ax.axis('equal')
    ax.set_title('Orientation to North in degrees:'+ str(CamPos['yaw']))

    return fig


#########################################################################################################################################
#########################################################################################################################################

def rotate_point_3D(x,y,yaw,pitch,roll):

    ''' 
    This function is based to Euler Angles that explain the rotate angles. 
    For more information:
    https://en.wikipedia.org/wiki/Euler_angles

    The rotation is done through yaw, pitch and roll values. 
    
    Input:
       x = values corresponding to the x axis
       y = values corresponding to the y axis
       yaw = video heading to North
       pitch = video pitch (for Nadir pitch = -90)
       roll = video roll (for Nadir roll = 0)

    Output:
       pts = Rotate points in x and y.

    '''

    Rr = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))], [0, -np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll))]])
    Rp = np.array([[np.cos(np.deg2rad(pitch)), 0, -np.sin(np.deg2rad(pitch))], [0, 1, 0], [np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch))]])
    Ry = np.array([[np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0], [-np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw)), 0], [0, 0, 1]])

    Rt = np.dot(np.dot(Rr, Rp), Ry)

    pts = np.column_stack((x.flatten('F'), y.flatten('F'), np.zeros_like(y).flatten('F'))) 

    pts = np.dot(Rt, pts.T) 
    
    return pts

#########################################################################################################################################
#########################################################################################################################################


def georeference_vectors(CamPos_ST,STCFIT, SG_Ux, SG_Uy):

    '''

    This function is used to georeference the image and current velocity vectors.
    The image grid is converted from meters to geographic coordinates (latitude and longitude).
    The image grid and velocity vectors u and v are rotated according to true north orientation.

    Input:
       CamPos
       STCFIT
       SG_Ux = Velocity vector in x orientation 
       SG_Uy = Velocity vector in y orientation

    Output:
       Ux_geo = georeferenced velocity vectors in x
       Uy_geo = georeferenced velocity vectors in y
       grid_long_geo = Grid corresponding to longitude
       grid_lat_geo = Grid corresponding to latitude
       long_geo = Longitude corresponding to the center of each window
       lat_geo = Latitude corresponding to the center of each window

    '''


    # Variables camera grid
    IND_center = np.ravel_multi_index((np.int32(STCFIT['Windows']['w_corners_dim1'][0,:]),np.int32(STCFIT['Windows']['w_corners_dim2'][0,:])),np.shape(STCFIT['Generic']['gridX']),order='F')

    x1=(STCFIT['Generic']['gridX']).flatten(order='F')[IND_center]
    y1=(STCFIT['Generic']['gridY']).flatten(order='F')[IND_center]

    # Variable that contain image grid
    gridX = STCFIT['Generic']['gridX']
    gridY = STCFIT['Generic']['gridY']

    ### Rotates the center point of each window ###
    grid_pts = rotate_point_3D(x1,y1,CamPos_ST['yaw'], 0,0)  # Function to rotate points
    x_geo = np.reshape(grid_pts[0,:],np.shape(x1))  # Reshape the grid
    y_geo = np.reshape(grid_pts[1,:],np.shape(y1))


    ### Rotates the current vectors ###
    # Variable that contain the vectors x and y from the second first:
    Ux = np.array(SG_Ux)  
    Uy = np.array(SG_Uy)
    # Function to rotate points:
    vector_rotate = rotate_point_3D(Ux, Uy,CamPos_ST['yaw'], 0,0) 
    #Reshape the vectors:
    Ux_geo = np.reshape(vector_rotate[0,:],np.shape(Ux))  
    Uy_geo = np.reshape(vector_rotate[1,:],np.shape(Uy))

    ### Rotates the image grid ###
    # Function to rotate points:
    grid_center_pts = rotate_point_3D(gridX, gridY, CamPos_ST['yaw'], 0,0) 
    # Reshape the grid for the image shape:
    gridX1 = np.reshape(grid_center_pts[0,:],np.shape(gridX), order = 'F') 
    gridY1 = np.reshape(grid_center_pts[1,:],np.shape(gridY), order = 'F')


    ### Converts grid from meters to lat/long coordinates in degrees
    
    # Longitude -52 : one degree of longitude corresponds to approximately 85394 meters.
    deg_lon = 1/85394
    # Latitude -32 : one degree of latitude corresponds to approximately 110574 meters. 
    deg_lat = 1/110574

    ### Get center point grid image windows ###
    #Multiply the vector grid by the value that equals 1 meter in degrees, passing the values to degrees:
    A= (deg_lon*x_geo)
    B= (deg_lat*y_geo)
    #Adds the reference lat/long of the center point of the image to match the right lat/long:
    long_geo = A + (STCFIT['Generic']['Longitude']) 
    lat_geo = B + (STCFIT['Generic']['Latitude']) 
    
    ### Get center point grid ###
    #Multiply the vector grid by the value that equals 1 meter in degrees, passing the values to degrees:
    A1= (deg_lon*gridX1)
    B1= (deg_lat*gridY1)
    #Adds the reference lat/long of the center point of the image to match the right lat/long:
    grid_long_geo = A1 + (STCFIT['Generic']['Longitude'])
    grid_lat_geo = B1 + (STCFIT['Generic']['Latitude'])
    
    return Ux_geo, Uy_geo, grid_long_geo, grid_lat_geo, long_geo, lat_geo


#########################################################################################################################################
#########################################################################################################################################


def plot_georeference_vectors(STCFIT, grid_long_geo, grid_lat_geo, long_geo, lat_geo, Ux_geo, Uy_geo):

    '''

    Plot the georeference vectors with the image

    Input:
       STCFIT
       grid_long_geo = Grid corresponding to longitude
       grid_lat_geo = Grid corresponding to latitude
       long_geo = Longitude corresponding to the center of each window
       lat_geo = Latitude corresponding to the center of each window
       Ux_geo = georeferenced velocity vectors in x
       Uy_geo = georeferenced velocity vectors in y

    Output:
       Figure contains the georeference vectors and image georeferenced


    '''

    ### Plot vectors in georefecenced image
    fig, ax = plt.subplots(figsize=(9,7))

    ax.pcolor(grid_long_geo, grid_lat_geo,  STCFIT['Generic']['image'][:,:,0], cmap='gray')  # Plot image georeferenced (lat/lon in degrees)
    ax.quiver(long_geo,lat_geo, Ux_geo, Uy_geo, color='y',scale=10, headwidth=2, headaxislength=3, headlength=3)  # Plot quiver with the georeferenced vectors
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.ticklabel_format(useOffset=False, style='plain')
    ax.axis('equal')
    
    return fig


    ##################################################################################################################
    ##################################################################################################################


def  land_mask(mask, latitude, longitude, Ux, Uy):

    '''
    Insert a mask that removes areas of land in the image so as not to generate vectors.

    '''


    
    mask = np.array(mask) #Transform in numpy array

    polygon = Polygon(mask)  # Turns polygon into package object

    latitude = latitude.flatten()
    longitude = longitude.flatten()
    Ux = Ux.flatten()
    Uy = Uy.flatten()

    points = np.vstack((longitude, latitude)).T
    uv = np.vstack((Ux, Uy)).T

    r = 0
    for i, p in enumerate(points):
        point = Point(p)
        if point.within(polygon) == True:
            points[i,:] = np.nan
            uv[i,:] = np.nan

    return (points, uv)


#################################################################################################################
#################################################################################################################

def plot_vectors_mask(points, uv, grid_long_geo, grid_lat_geo, STCFIT):

    
    '''
    Plot the vectors with the mask inserted
    '''

    fig, ax = plt.subplots(figsize=(9,7))

    ax.pcolor(grid_long_geo, grid_lat_geo,  STCFIT['Generic']['image'][:,:,0], cmap='gray')  # Plot image georeferenced (lat/lon in degrees)
    ax.quiver(points[:,0],points[:,1], uv[:,0], uv[:,1], color='y',scale=10, headwidth=2, headaxislength=3, headlength=3)  # Plot quiver with the georeferenced vectors
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.ticklabel_format(useOffset=False, style='plain')
    ax.axis('equal')
    
    return fig

    