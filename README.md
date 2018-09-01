# Advanced Lane Finding Project Report

### Yulong Li  

---  

## Goals  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---  

## Rubric Points

I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one  

Please see this report.  

---  

### Camera Calibration

#### 1. Camera matrix and distortion coefficients calculation

The code for this step is from line 55 to 99 in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original:  
![original](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/calibration1.jpg)  

Calibrated:  
![Calibrated](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/test_undist.jpg)  

---  

### Pipeline (single images)

#### 1. An example of a distortion-corrected image  

Function *camera_cal(img)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 104 to 112 is used for camera calibration for each frame.  
 
![CC](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/Camera%20Calibration.png)  

#### 2. Use of color transforms and gradients to create a thresholded binary image

Function *thresholded_binary(img)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 113 to 164 is used for generating thresholded binary image for each frame. I used a combination of sobel in x and y direction, combined with S channel thresholding of the HLS color space. In the example below, green pixels are from sobel edge detection and blue ones are from HLS S channel.    

![thresholded](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/thresholded.png)  

#### 3. Perspective transform

Function *perspective_transform(f_combined_binary)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 164 to 172 is used for perspective transform for each frame. Perspective transform M is precalculated at beginning and stored in the stored_data class. src and dst are chosen as below:  

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 593, 452      | 400, 0        | 
| 687, 452      | 880, 0      |
| 1047, 693     | 880, 719      |
| 256, 693      | 400, 719        |  
  

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![pt](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/pt.png) 


#### 4. Lane-line pixels identification and polynomial fit

Function *pixels_polyfit(f_binary_warped)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 172 to 445 is used for identifying pixels and fitting the pixels into a quadratic curve. Left line and right line are processed separately and their information is stored in l_line class and r_line class. Last 5 fits, last 5 fits mean, current fit, line detection flag and etc. are recorded.  

When no line is detected, sliding window will be used. When line is detected, sliding windows will be skipped and a margin area around the quadratic line will be used as the searching area. After line detected, the system will check the current fit changing rate to avoid jumps. If it's below thresholds, a weighted mean of the last 5 fits will be calculated with a higher weight on current fit. Otherwise, a lower weight will be assigned to current fit. Besides left line fit will also be mixed with 10% of right line fit to make it more parallel, vice versa.  

![polyfit](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/polyfit.png)  


#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center

Function *curvature_dist(f_img_draw)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 471 to 522 is used for calculating real world lane radius of curvature and the position of the vehicle to center. The results will be printed on each frame.   

#### 6. An example image of the result plotted back down onto the road  

Function *draw_path(f_undist, binary_warped)* in [p4.py](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.py) from line 448 to 471 is used for drawing green shaded area of the lane path on the frames.

![result](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/output_images/result.PNG)  

---

### Pipeline (video)

#### 1. Final video output

Please check out this link [p4.mp4](https://github.com/yulongl/p4_AdvancedLaneFinding/blob/master/p4.mp4).

---

### Discussion

#### 1. Brief discussion about problems / issues in this implementation of this project and potential solutions

##### 1. Restrictions on road conditions and environments  
The algorithm is highly affected by road conditions and environments. The binary image thresholds need to be tuned very carefully for a certain road condition and does not perform well under other environments. I was not able to find a generic set of parameters which performs well on all the provided videos.  
**Potential solution**: deep learning may be helpful finding out the lane lines, but requires a lot of work for labeling.   

##### 2. React slowly to sudden changes
Because of the averaging mechanism, the system reacts slowly to some sudden changes, for example, in bumping road condition.  
**Potential solution**: Increase the filter thresholds and introduce second order derivatives.

##### 3. Processing speed is slow
Average processing speed is around 8fps. This may be a little slow for real time application.  
**Potential solution**: optimize the algorithm. GPU may acclerate the process.
