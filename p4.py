import cv2
import glob
import numpy as np
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.last_5_fits = []
        self.last_5_mean = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.frame_counter = 0


class DataStorage:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None


l_line = Line()
r_line = Line()
stored_data = DataStorage()

# src = np.float32([[592, 452], [686, 452], [1069, 693], [233, 693]])
src = np.float32([[593, 452], [687, 452], [1047, 693], [256, 693]])
dst = np.float32([[400, 0], [880, 0], [880, 719], [400, 719]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
stored_data.M = M
stored_data.Minv = Minv
###################################################################################################################
# camera calibration
# prepare object points
nx = 9  # number of inside corners in x
ny = 6  # number of inside corners in y

# read in calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# object points
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)


# Test undistortion on an image
img = cv2.imread('camera_cal/calibration1.jpg')

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
stored_data.mtx = mtx
stored_data.dist = dist

camera_test = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/camera_test.jpg', camera_test)
# cv2.imshow('camera_test', camera_test)
# cv2.waitKey(0)

###################################################################################################################
# pipeline start
# Provide an example of a distortion-corrected image.


def camera_cal(img):
    mtx = stored_data.mtx
    dist = stored_data.dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imwrite('test_images/straight_lines1_undist.jpg', undist)
    # cv2.imshow('straight_lines1', straight_lines1_undist)
    # cv2.waitKey(0)

    return undist
###################################################################################################################
# color transforms, gradients to create a thresholded binary image
# Grayscale image
# NOTE: we already saw that standard grayscaling lost color information for the lane lines
# Explore gradients in other colors spaces / color channels to see what might work better


def thresholded_binary(img):
    #img = straight_lines1_undist
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaussian_blur
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    thresh_min = 40
    thresh_max = 80
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1

    # Sobel y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    thresh_min = 40
    thresh_max = 80
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1

    combined_sobel = np.zeros_like(sybinary)
    combined_sobel[(sxbinary == 1) & (sybinary == 1)] = 1

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh_min = 175
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (combined_sobel == 1)] = 1

    return combined_binary
###################################################################################################################
# perspective transform


def perspective_transform(f_combined_binary):

    f_binary_warped = cv2.warpPerspective(f_combined_binary, stored_data.M, (f_combined_binary.shape[1], f_combined_binary.shape[0]))
    return f_binary_warped
###################################################################################################################
# find pixels and polyfit
# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image


def pixels_polyfit(f_binary_warped):

    histogram = np.sum(f_binary_warped[f_binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((f_binary_warped, f_binary_warped, f_binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(f_binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = f_binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Left
    if l_line.detected == False:
        # Current positions to be updated for each window
        leftx_current = leftx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = f_binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = f_binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, f_binary_warped.shape[0] - 1, f_binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        l_line.detected = True
    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit = l_line.current_fit
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        if (len(leftx) < 20) | (len(lefty) < 20):
            left_fit = l_line.current_fit
            l_line.detected = False
        else:
        # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, f_binary_warped.shape[0] - 1, f_binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((f_binary_warped, f_binary_warped, f_binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))

    # Right
    if r_line.detected == False:
        rightx_current = rightx_base

        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = f_binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = f_binary_warped.shape[0] - window * window_height
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, f_binary_warped.shape[0] - 1, f_binary_warped.shape[0])
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        r_line.detected = True

    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = f_binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        right_fit = r_line.current_fit
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if (len(rightx) < 20) | (len(righty) < 20):
            right_fit = r_line.current_fit
            r_line.detected = False
        else:
            # Fit a second order polynomial to each
            right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, f_binary_warped.shape[0] - 1, f_binary_warped.shape[0])
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((f_binary_warped, f_binary_warped, f_binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    if l_line.frame_counter == 1:
        l_line.last_5_fits.append(left_fit)
        l_line.last_5_mean = left_fit
        l_line.current_fit = left_fit
    elif 1 < l_line.frame_counter <= 5:
        l_line.last_5_fits.append(left_fit)
        mean_sum = 0
        for fit in l_line.last_5_fits:
            mean_sum += fit
        l_line.last_5_mean = mean_sum/len(l_line.last_5_fits)
        l_line.current_fit = left_fit
    elif l_line.frame_counter > 5:
        left_a_error = np.absolute(left_fit[0] - l_line.current_fit[0])
        left_b_error = np.absolute(left_fit[1] - l_line.current_fit[1])
        left_c_error = np.absolute(left_fit[2] - l_line.current_fit[2])
        if (left_a_error < 0.0001) & (left_b_error < 0.1) & (left_c_error < 80):
            l_line.last_5_fits[0] = l_line.last_5_fits[1]
            l_line.last_5_fits[1] = l_line.last_5_fits[2]
            l_line.last_5_fits[2] = l_line.last_5_fits[3]
            l_line.last_5_fits[3] = l_line.last_5_fits[4]
            l_line.last_5_fits[4] = left_fit
            l_line.last_5_mean = (0.1 * l_line.last_5_fits[0] +
                                  0.2 * l_line.last_5_fits[1] +
                                  0.2 * l_line.last_5_fits[2] +
                                  0.2 * l_line.last_5_fits[3] +
                                  0.3 * l_line.last_5_fits[4])
            l_line.current_fit = l_line.last_5_mean
        else:
            l_line.current_fit = 0.9 * l_line.last_5_mean + 0.1 * left_fit
            l_line.last_5_mean = l_line.current_fit
            l_line.last_5_fits[0] = l_line.last_5_fits[1]
            l_line.last_5_fits[1] = l_line.last_5_fits[2]
            l_line.last_5_fits[2] = l_line.last_5_fits[3]
            l_line.last_5_fits[3] = l_line.last_5_fits[4]
            l_line.last_5_fits[4] = l_line.current_fit

    if r_line.frame_counter == 1:
        r_line.last_5_fits.append(right_fit)
        r_line.last_5_mean = right_fit
        r_line.current_fit = right_fit
    elif 1 < r_line.frame_counter <= 5:
        r_line.last_5_fits.append(right_fit)
        mean_sum = 0
        for fit in r_line.last_5_fits:
            mean_sum += fit
        r_line.last_5_mean = mean_sum/len(r_line.last_5_fits)
        r_line.current_fit = right_fit
    elif r_line.frame_counter > 5:
        right_a_error = np.absolute(right_fit[0] - r_line.current_fit[0])
        right_b_error = np.absolute(right_fit[1] - r_line.current_fit[1])
        right_c_error = np.absolute(right_fit[2] - r_line.current_fit[2])
        if (right_a_error < 0.0001) & (right_b_error < 0.1) & (right_c_error < 80):
            r_line.last_5_fits[0] = r_line.last_5_fits[1]
            r_line.last_5_fits[1] = r_line.last_5_fits[2]
            r_line.last_5_fits[2] = r_line.last_5_fits[3]
            r_line.last_5_fits[3] = r_line.last_5_fits[4]
            r_line.last_5_fits[4] = right_fit
            r_line.last_5_mean = (0.1 * r_line.last_5_fits[0] +
                                  0.2 * r_line.last_5_fits[1] +
                                  0.2 * r_line.last_5_fits[2] +
                                  0.2 * r_line.last_5_fits[3] +
                                  0.3 * r_line.last_5_fits[4])
            r_line.current_fit = r_line.last_5_mean
        else:
            r_line.current_fit = 0.9 * r_line.last_5_mean + 0.1 * right_fit
            r_line.last_5_mean = r_line.current_fit
            r_line.last_5_fits[0] = r_line.last_5_fits[1]
            r_line.last_5_fits[1] = r_line.last_5_fits[2]
            r_line.last_5_fits[2] = r_line.last_5_fits[3]
            r_line.last_5_fits[3] = r_line.last_5_fits[4]
            r_line.last_5_fits[4] = r_line.current_fit

    l_line.current_fit[0] = 0.9 * l_line.current_fit[0] + 0.1 * r_line.current_fit[0]
    l_line.current_fit[1] = 0.9 * l_line.current_fit[1] + 0.1 * r_line.current_fit[1]
    r_line.current_fit[0] = 0.9 * r_line.current_fit[0] + 0.1 * l_line.current_fit[0]
    r_line.current_fit[1] = 0.9 * r_line.current_fit[1] + 0.1 * l_line.current_fit[1]

###################################################################################################################


def draw_path(f_undist, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fit = l_line.current_fit
    right_fit = r_line.current_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, stored_data.Minv, (f_undist.shape[1], f_undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(f_undist, 1, newwarp, 0.3, 0)
    return result
###################################################################################################################


def curvature_dist(f_img_draw):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 110  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 683  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    ploty = np.linspace(0, f_img_draw.shape[0] - 1, f_img_draw.shape[0])
    y_eval = np.max(ploty) / 2
    left_fit = l_line.current_fit
    right_fit = r_line.current_fit

    a_l = left_fit[0] * xm_per_pix / (ym_per_pix**2)
    a_r = right_fit[0] * xm_per_pix / (ym_per_pix**2)
    b_l = left_fit[1] * xm_per_pix / ym_per_pix
    b_r = right_fit[1] * xm_per_pix / ym_per_pix
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * a_l * y_eval + b_l) ** 2) ** 1.5) / np.absolute(2 * a_l)
    right_curverad = ((1 + (2 * a_r * y_eval + b_r) ** 2) ** 1.5) / np.absolute(2 * a_r)
    ave_curved = (left_curverad + right_curverad) / 2
    # Now our radius of curvature is in meters
    left_x_bot = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x_bot = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    dist_mid = ((left_x_bot + right_x_bot) / 2 - 640) * xm_per_pix
    # output to image
    cv2.putText(
        f_img_draw,
        'Radius of Carvature = {:d} m'.format(int(ave_curved)),
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        1, (255, 255, 255))
    cv2.putText(
        f_img_draw,
        'Position to Center = {:f}'.format(dist_mid),
        (10, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1, (255, 255, 255))
    # cv2.putText(
    #     f_img_draw,
    #     'a {:f} b {:f} c{:f}'.format(left_fit[0], left_fit[1], left_fit[2]),
    #     (10, 90),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     1, (255, 255, 255))
    # cv2.putText(
    #     f_img_draw,
    #     'a {:f} b {:f} c{:f}'.format(right_fit[0], right_fit[1], right_fit[2]),
    #     (10, 120),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     1, (255, 255, 255))
###################################################################################################################
# process_image


def process_image(img):
    if l_line.frame_counter <= 5:
        l_line.frame_counter += 1
    if r_line.frame_counter <= 5:
        r_line.frame_counter += 1
    undist = camera_cal(img)
    combined_binary = thresholded_binary(undist)
    binary_warped = perspective_transform(combined_binary)
    pixels_polyfit(binary_warped)
    img_draw = draw_path(undist, binary_warped)
    curvature_dist(img_draw)

    return img_draw


###################################################################################################################
# Video

white_output = 'test.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(0, 5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!

white_clip.write_videofile(white_output, audio=False)

