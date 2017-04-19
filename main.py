import numpy as np
import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from imagelibs import *

# prepare object points
nx, ny = 9, 6
calibration_images = glob.glob('camera_cal/calibration*.jpg')
objpoints,imgpoints,calibration_shape= make_calibrate_points(calibration_images, nx, ny)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, calibration_shape, None, None)

def process_image(img_unstorted):
    img = cv2.undistort(img_unstorted, mtx, dist, None, mtx)
    # combination of color and gradient thresholds to generate a binary image
    binary_warped = threshold_image(img)

    # performed a perspective transform
    img_size = img.shape[1], img.shape[0]
    src = np.float32(
        [[(img_size[0] / 2) - 61, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 63), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 5), 0],
         [(img_size[0] / 5), img_size[1]],
         [(img_size[0] * 4 / 5), img_size[1]],
         [(img_size[0] * 4 / 5), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    perspective_image = cv2.warpPerspective(img, M, dsize=img_size, flags=cv2.INTER_LINEAR)
    binary_warped_perspective = cv2.warpPerspective(binary_warped, M, dsize=img_size, flags=cv2.INTER_LINEAR)
    out_img = np.dstack((binary_warped_perspective, binary_warped_perspective, binary_warped_perspective)) * 255
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped_perspective.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary_warped_perspective[int(binary_warped_perspective.shape[0]/2):, :], axis=0)
    leftpoint = np.int(0) #histogram.shape[0] / 8)
    midpoint = np.int(histogram.shape[0] / 2)
    rightpoint = np.int(histogram.shape[0] * 8 / 8)
    leftx_base = np.argmax(histogram[leftpoint:midpoint]) + leftpoint
    rightx_base = np.argmax(histogram[midpoint:rightpoint]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped_perspective.shape[0] / nwindows)
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped_perspective.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped_perspective.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped_perspective.shape[0] - 1, binary_warped_perspective.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped_perspective).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(img_unstorted, 1, newwarp, 0.3, 0)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    curverad = min(left_curverad, right_curverad)
    # Caculate the distance of center
    midpoint = np.int(binary_warped.shape[1] / 2)
    image_height = binary_warped.shape[0]
    leftpoint = left_fit[0] * image_height * image_height + left_fit[1] * image_height + left_fit[2]
    rightpoint = right_fit[0] * image_height * image_height + right_fit[1] * image_height + right_fit[2]
    diffpoint = (leftpoint + rightpoint) / 2 - midpoint
    diffditance = diffpoint * xm_per_pix

    text1 = "Raduis of Curverad = {:.0f}(m)".format(curverad)
    cv2.putText(result, text1, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    text2 = "Vehicle is {:.2f}m left of center".format(diffditance)
    cv2.putText(result, text2, (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    w,h = result.shape[1]/2,result.shape[0]/2
    binary_warped = cv2.resize(binary_warped, (w,h))
    binary_warped_perspective = cv2.resize(binary_warped_perspective, (w,h))
    binary_warped = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    binary_warped_perspective = np.dstack((binary_warped_perspective, binary_warped_perspective, binary_warped_perspective)) * 255
    perspective_image = cv2.resize(perspective_image, (w, h))
    out_img = cv2.resize(out_img, (w, h))
    vis1 = np.concatenate((binary_warped, out_img), axis=0)
    vis2 = np.concatenate((perspective_image, binary_warped_perspective), axis=0)
    vis = np.concatenate((result, vis1, vis2), axis=1)

    prevfits = (left_fit, right_fit)
    return vis

######### Images #########
print("begin images")
image_files = os.listdir("test_images/")
for image_file in image_files:
    image = mpimg.imread("test_images/" + image_file)
    prevfits = None
    result = process_image(image)
    mpimg.imsave("output_images/" + image_file, result)
    print("save "+image_file)
print("")

######### videos #########
print("begin videos:")
#for file_name in ["project_video.mp4","challenge_video.mp4","harder_challenge_video.mp4"]:
for file_name in ["challenge_video.mp4"]:
    print("begin doing " + file_name)
    write_output = "output_videos/output_" + file_name
    clip = VideoFileClip(file_name)
    white_clip = clip.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    print("end doing " + file_name)