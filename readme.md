<div>

<span class="c26 c29"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 2.67px;">![](images/image05.png "horizontal line")</span>

</div>

<span class="c14"> UDACITY SELF DRIVING CAR ENGINEER NANODEGREE  </span>

<span class="c20">Term 1: Computer Vision and Deep Learning</span>

<span>PROJECT 4</span>

<span class="c13 c30">ADVANCED LANE FINDING</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 4.00px;">![](images/image04.png "horizontal line")</span>

# <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 544.50px; height: 627.97px;">![](images/image13.png)</span>

# <span class="c21 c13">Introduction</span>

<span class="c0">The goals / steps of this project are the following:</span>

*   <span class="c0">Compute the camera calibration matrix and distortion coefﬁcients given a set of chessboard images.</span>
*   <span class="c0">Apply a distortion correction to raw images.</span>
*   <span class="c0">Use color transforms, gradients, etc., to create a thresholded binary image.</span>
*   <span class="c0">Apply a perspective transform to rectify binary image ("bird's-eye view").</span>
*   <span class="c0">Detect lane pixels and ﬁt to ﬁnd the lane boundary.</span>
*   <span class="c0">Determine the curvature of the lane with respect to center.</span>
*   <span class="c0">Warp the detected lane boundaries back onto the original image.</span>
*   <span class="c0">Output visual display of the lane boundaries and numerical estimation of lane curvature.</span>

<span class="c0">Project submission includes following files:</span>

*   <span class="c0">Writeup ‘project4_writeup.html’ file (you’re reading it right now).</span>
*   <span class="c0">Jupyter notebook ‘advanced_lane_detection.ipynb’, which contains all project code and additional commentary of the project implementation.</span>
*   <span class="c0">Example output images for each stage of the processing pipeline in the ‘output_images’ folder.</span>
*   <span class="c0">Output video files in the ‘output_videos’ folder.</span>

<span class="c0">This writeup includes statements and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled. Also, please take a look at jupyter notebook, which contains additional commentary for pipeline implementation.</span>

<span>All rubric points addressed in order and described accordingly to</span> <span class="c31">[Project Specifications](https://www.google.com/url?q=https://review.udacity.com/%23!/rubrics/571/view&sa=D&ust=1488096805078000&usg=AFQjCNGZXtxD2pCUix3WxZhmJIlq87zu_g)</span>

# <span class="c21 c13"></span>

# <span class="c13 c21">Rubric points</span>

## <span class="c12">Camera Calibration</span>

###### <span class="c15">CRITERIA: Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.</span>

<span>T</span><span class="c0">he code for this step is contained in the ﬁrst and second code cells of the IPython notebook.</span>

<span>For each camera calibration image I am computing chessboard corners position using</span> <span class="c1 c9">cv2.findChessboardCorners</span><span> function applied to the grayscaled original image. All found object and corner points stored in separate arrays which are then used as arguments for the</span> <span class="c1">cv2.calibrateCamera</span><span> function which produces calibration and distortion coefficients of the camera. Using these coefficients I am undistorting test calibration images using</span> <span class="c1">cv2.undistort</span><span> function, examples of output are in the</span> <span class="c13">‘output_images/out_calibration[N].jpg’</span><span class="c0"> files.</span>

<span class="c0"> Here is an example of output:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 172.00px;">![](images/image08.png)</span>

<span class="c0"></span>

* * *

<span class="c0"></span>

## <span class="c12">Pipeline (test images)</span>

<span class="c0"></span>

###### <span class="c15">CRITERIA: Provide an example of a distortion-corrected image.</span>

<span class="c0">Code for this step contained it the # 3 code cell of the IPython notebook</span>

<span>I am applying distortion and calibration coefficients for the camera obtained in the previous step  to test road images, using</span> <span class="c1">cv2.undistort</span><span> function. Output files are in the</span> <span class="c13">‘output_images/calibrated_[image_name].jpg’</span><span class="c0"> files.  </span>

<span class="c0">Here is an example of the undistorted test image:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 675.06px; height: 393.17px;">![](images/image10.png)</span>

<span class="c0"></span>

* * *

<span class="c0"></span>

###### <span class="c15">CRITERIA: Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.</span>

<span class="c0">Code for this step contained in the # 4 and  # 5 code cells of the IPython Notebook.</span>

<span>Firstly, I’m creating a HLS color space version of the original image using</span> <span class="c1">cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)</span><span> and separating S channel, which contains a lot of information about possible lane lines on the image. After that I am computing absolute scaled sobel derivative to accentuate lines away from horizontal using</span> <span class="c1">cv2.Sobel</span><span> function. Using th</span><span>is derivative</span><span>, I am computing threshold X gradient using threshold values in the range</span> <span class="c13">(20, 100)</span><span>. I am doing the same thing for S color channel, thresholding it values in the range</span> <span class="c13">(170, 255)</span><span>. So now I have two binary thresholds arrays, one for sobel derivative, and second for the S channel. Now I’m stacking these  binary arrays into one binary array to produce binary image with highlighted possible lane lines. Output files are in the</span> <span class="c13">‘output_images/thresholds_[image_name].jpg’</span><span> files.
Here is an example of the original and resulting image after thresholding have been applied (on stacked thresholds image</span> <span>green color</span><span> stands for</span> <span class="c24">Sobel derivative threshold</span><span> and blue for</span> <span class="c25">S channel threshold</span><span class="c0">):</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 571.50px; height: 341.03px;">![](images/image09.png)</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 373.33px;">![](images/image12.png)</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 377.33px;">![](images/image06.png)</span>

###### <span class="c15">CRITERIA: Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.</span>

<span class="c0">Code for this step contained in the # 6 and # 7 code cells of the IPython Notebook.</span>

<span>To perform a perspective transform, I am using</span> <span class="c1">cv2.getPerspectiveTransform</span><span> function applied to source and destination points coordinates, which produce matrix for perspective image transformation. Then I am using</span> <span class="c1">cv2.warpPerspective</span> <span>function to get warped image</span><span>Output files are in the</span> <span class="c13">‘output_images/perspective_[image_name].jpg’</span><span class="c0"> files.</span>

<span class="c0">Before applying perspective transform, I perform image thresholding. Here is an example of what result image looks like with highlighted source and destination areas:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 380.00px;">![](images/image14.png)</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 380.00px;">![](images/image01.png)</span>

<span class="c0"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 373.33px;">![](images/image11.png)</span>

###### <span class="c15">CRITERIA: Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?</span>

<span class="c0">Code for this step contained in the # 8 code cell of the iPython Notebook.</span>

<span>After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image where the lane lines stand out clearly. To decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line I first take a histogram along all the columns in the lower half of the image:  
</span><span class="c1">histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)</span><span>Then I find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. After this I am defining sliding windows starting from these points  and stepping through the windows one by one to identify non-zero pixels and append these indices to the line pixels lists. This allows me to extract left and right line pixels positions. Then I am fitting those positions into polynomial coefficients using</span> <span class="c1">np.polyfit</span> <span>function.
</span><span>Output files are in the</span> <span class="c13">‘output_images/fit_poly_[image_name].jpg’</span><span class="c0"> files.
Here is example of output:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 372.00px;">![](images/image15.png)</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 346.67px;">![](images/image03.png)</span>

<span class="c7"></span>

* * *

<span class="c7"></span>

###### <span class="c15">CRITERIA: Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.</span>

<span class="c0">Code for this step contained in the # 11 code cells of the iPython Notebook.</span>

<span>I have a polynomial coefficients for the lane lines functions. To find out a curvature of the lane I am simply using formula</span> <span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 153.00px; height: 54.00px;">![](images/image07.png)</span><span> . Also, because I need to transform pixel image distance to meters, I use conversion coefficients for</span> <span class="c13">y = 30/720</span><span> and for</span> <span class="c13">x = 3.7/700</span> <span class="c0">meters per pixels in y and x dimensions respectively.
To find out a car position relatively to the center of the lane I use bottom points of the lane lines functions , so now I know left and right border of the lane relatively to the car. Distance from the center of the image to lane border will define relative car position to the center of the lane. You can see lane curvature radius and relative car position being displayed on the result video output of the pipeline.
        Here is an example of the videostream image with the lane curvature and lane position being displayed in the top left corner:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 645.50px; height: 363.09px;">![](images/image00.png)</span>

###### <span class="c15">CRITERIA: Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.</span>

<span class="c0">Code for this step contained in the # 8 code cells of the iPython Notebook.</span>

<span>To project lane area back to the original image I use a backward warp function</span> <span class="c1">cv2.warpPerspective</span><span> (with inverted coefficients) of the result lane projection. And then I use</span> <span class="c1">cv2.addWeighted</span><span class="c0">  function to highlight lane area on the original image.</span>

<span>Output files are in the</span> <span class="c13">‘output_images/fit_poly_[image_name].jpg’</span><span class="c0"> files.</span>

<span class="c0">Here is an example of original test image with projected lane area:</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 657.69px; height: 374.17px;">![](images/image02.png)</span>

## <span class="c12"></span>

* * *

## <span class="c12"></span>

## <span class="c12">Pipeline (video)</span>

###### <span class="c15">CRITERIA: Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)</span>

<span class="c0">Processing of videos implementation code contained in the # 11 - 17 code cells of the IPython Notebook.</span>

<span>Output files are in the</span> <span class="c13">‘output_videos’</span><span class="c0"> folder.</span>

<span class="c31">[Resulting video](./output_videos/project_video_result.mp4)</span>

## <span class="c12">Discussion</span>

###### <span class="c15">CRITERIA: Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?</span>

<span class="c0">Most time consuming issues are include fine-tuning perspective transform points coordinates (I ended up using points coordinates provided with the example writeup file for the project). Also it is worth mentioning that thresholding range values also have to be picked up manually, and it is not an ordinary task, because even small difference make a big change for the resulting lane lines image.</span>

<span class="c0">Most likely my pipeline will fail on the videos with not clearly visible lane lines . As you can see, challenge videos was handled not correctly with my pipeline. I will try to improve my pipeline in meantime because overall it is a very profound and rewarding task.</span>

<span class="c0">To make my pipeline more robust I need to implement more sophisticated lane lines search algorithm. Including advanced window search algorithm and implement outlier rejection and use a low-pass filter to smooth the lane detection over frames.</span>

<span class="c0"></span>

<span class="c0"></span>