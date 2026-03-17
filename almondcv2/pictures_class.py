import cv2
import numpy as np
import os
from aux_functions import midpoint, pol2cart, cart2pol, calculate_vertical_symmetry, calculate_horizontal_symmetry, smoothing_masks, watershed
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
import pandas as pd
import traceback


class Pictures():
    def __init__(self, working_directory, input_folder,info_file,fruit, project_name,
                 binary_masks=False, blurring_binary_masks=False, blur_binary_masks_value=5, binary_pixel_size=250, threshold_binarization=254):
        """
        Initializes the Pictures class.

        This constructor sets up the project structure, including directories 
        and file paths required for processing images and storing results.

        Parameters:
            working_directory (str): The main directory where results will be stored.
            input_folder (str): The folder containing input images.
            info_file (str): The information file related to the images.
            fruit (str): The type of fruit analyzed.
            project_name (str): The name of the project.
            binary_masks (bool, optional): Enables binary mask generation. Defaults to False.
            blurring_binary_masks (bool, optional): Enables blurring for binary masks. Defaults to False.
            blur_binary_masks_value (int, optional): The blur kernel size for binary masks. Defaults to 5.
            binary_pixel_size (int, optional): The pixel size used for binary processing. Defaults to 250.
            threshold_binarization (int, optional): The threshold value for binarization after blurring. Defaults to 254.

        Returns:
            None
        """
        self.working_directory=working_directory
        self.input_folder=input_folder
        self.info_file=info_file
        self.fruit=fruit
        self.binary_masks=binary_masks
        self.blurring_binary_masks=blurring_binary_masks
        self.blur_binary_masks_value=blur_binary_masks_value
        self.binary_pixel_size=binary_pixel_size
        self.threshold_binarization=threshold_binarization

        self.project_name=project_name
        self.results_directory=os.path.join(working_directory,f"Results_{self.project_name}")
        os.makedirs(self.results_directory, exist_ok=True)

        self.path_export_1=os.path.join(self.results_directory, "results_morphology.txt")
        self.path_export_2=os.path.join(self.results_directory, "results_general.txt")
        self.path_export_3=os.path.join(self.results_directory, "pic_results")
        self.path_export_4=os.path.join(self.results_directory, "binary_masks")
        self.path_export_5=os.path.join(self.results_directory,"binary_masks_info_table.txt")
        self.path_export_outlier=os.path.join(self.results_directory,"outlier_table.txt")
        self.path_export_error=os.path.join(self.results_directory,"errors_table.txt")
        self.path_export_outlier_folder=os.path.join(self.results_directory, "outliers_pics")
        os.makedirs(self.path_export_3, exist_ok=True)
        os.makedirs(self.path_export_4, exist_ok=True)

    def set_postsegmentation_parameters(self,  segmentation_input, sahi=True, smoothing=False, kernel_smoothing=5,
                                         smoothing_iterations=2, watershed=False, kernel_watershed=5, threshold_watershed=0.7, watershed_iterations=3):
        
        """
        Sets the parameters for post-segmentation processing.

        This method configures different post-segmentation techniques such as 
        smoothing and watershed transformation, depending on whether SAHI (Sliced 
        Aided Hyper Inference) is enabled.

        Parameters:
            segmentation_input (str): The input for segmentation processing.
            sahi (bool, optional): Whether to use SAHI-based segmentation. Defaults to True.
            smoothing (bool, optional): Enables smoothing for segmentation. Defaults to False.
            kernel_smoothing (int, optional): Kernel size for smoothing operation. Defaults to 5.
            smoothing_iterations (int, optional): Number of iterations for smoothing. Defaults to 2.
            watershed (bool, optional): Enables watershed segmentation. Defaults to False.
            kernel_watershed (int, optional): Kernel size for watershed processing. Defaults to 5.
            threshold_watershed (float, optional): Threshold value for watershed segmentation. Defaults to 0.7.
            watershed_iterations (int, optional): Number of iterations for watershed segmentation. Defaults to 3.

        Returns:
            None
        """
        
        self.sahi=sahi
        self.segmentation_input=segmentation_input
        if self.sahi==True:
            pass
        else:
            self.smoothing=smoothing
            self.smoothing_kernel=kernel_smoothing
            self.smooting_iterations=smoothing_iterations
            self.watershed=watershed
            self.kernel_watershed=kernel_watershed
            self.threshold_watershed=threshold_watershed
            self.watershed_iterations=watershed_iterations

    
    def measure_almonds(self, margin=100, spacing=30, limit_area_pixels_min=600, line_thickness=1, text_size=0.75, text_thickness=2,
                         text_color=(255,255,255), line_color=(0,255,0)):

        self.margin=margin
        self.spacing=spacing
        self.limit_area_pixels_min=limit_area_pixels_min
        self.text_size=text_size
        self.line_thickness=line_thickness
        self.text_thickness=text_thickness
        self.text_color=text_color
        self.line_color=line_color

        """
        Measures almonds and exports results.

        This method sets measurement parameters for almond analysis, including 
        margin size, spacing, and minimum area threshold for valid detections. 
        It processes the detected almonds and generates output files, such as 
        CSV reports and images with annotations.

        Parameters:
            margin (int, optional): The margin size around the almonds in pixels. Defaults to 100.
            spacing (int, optional): The spacing between detected almonds in pixels. Defaults to 30.
            limit_area_pixels_min (int, optional): The minimum area in pixels for an almond to be considered valid. Defaults to 600.
            line_thickness (int, optional): Thickness of the annotation lines in output images. Defaults to 1.
            text_size (float, optional): Font size for text annotations. Defaults to 0.75.
            text_thickness (int, optional): Thickness of the text in annotations. Defaults to 2.
            text_color (tuple, optional): Color of the text annotations in BGR format. Defaults to (255, 255, 255) (white).
            line_color (tuple, optional): Color of the annotation lines in BGR format. Defaults to (0, 255, 0) (green).

        Outputs:
            - Saves measurement results in CSV format:
                - Morphological data: `results_morphology.txt`
                - General statistics: `results_general.txt`
                - Binary masks information: `binary_masks_info_table.txt`
            - Exports images:
                - Processed almond images to `pic_results/`
                - Binary mask images to `binary_masks/`
                - Annotated images with measurements and contours

        Returns:
            None
        """
        

        #region Init table results 
        morphology_table=pd.DataFrame()
        morphology_table = pd.DataFrame(columns=["Project_name","Sample_picture","Fruit_name", "Fruit_number",
                                                 "Length","Width","Width_25","Width_50","Width_75","Area", "Perimeter","Hull_area",
                                                 "Solidity","Circularity","Ellipse_Ratio","L","a","b","Symmetry_v", "Symmetry_h","Shoulder_symmetry"])
        general_table=pd.DataFrame()
        general_table = pd.DataFrame(columns=["Name_picture","Sample_number","ID","Weight","Session","Shell","Pixelmetric","Sample_picture","N_fruits"])
        #endregion

        #region Loop over the pictures 
        error_list=[]
        for picture in self.segmentation_input:
            try:
                #region Post-segmentation processing
                if self.sahi==True: #  SAHI approach # Transform contour format to opencv format
                    name_pic=os.path.basename(picture[1])
                    mask_contours_list=picture[0].object_prediction_list
                    almonds=cv2.imread(picture[1])
                    sahi_contours_list=[]
                    for contour in mask_contours_list:
                        contour=contour.mask.segmentation[0]
                        array_contour=np.array(contour)
                        array_contour=array_contour.reshape(-1,2)
                        contour_pixels = array_contour.astype(np.int32)
                        contour = contour_pixels.reshape((-1, 1, 2))
                        sahi_contours_list.append(contour)
                    self.mask_contours_list=sahi_contours_list
                    
                else:
                    # For slice_predict_reconstruct approach simply define variables and find_contours
                    name_pic=os.path.basename(picture[1])
                    mask=picture[0]

                    if self.smoothing==True:

                        mask=smoothing_masks(mask=mask,smoothing_iterations=self.smooting_iterations, smoothing_kernel=self.smoothing_kernel)[0]

                    if self.watershed==True:
                        mask, rect_kernel=smoothing_masks(mask=mask,smoothing_iterations=self.smooting_iterations, smoothing_kernel=self.smoothing_kernel)
                        mask=watershed(mask=mask, rect_kernel=rect_kernel, iterations=self.watershed_iterations,
                                       kernel_watershed=self.kernel_watershed, threshold_watershed=self.threshold_watershed)

                    self.mask_contours_list, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    almonds=cv2.imread(picture[1])
                    


                        
                

                # Obtain pixelmetric values
                pixelmetric = self.info_file.loc[self.info_file['Sample_picture'] == name_pic, 'Pixelmetric'].values
                pixelmetric=float(pixelmetric)
                #endregion
            
                #region Prepare rectangles size for contours

                height_pic, width_pic = almonds.shape[:2]
                max_height = max(cv2.boundingRect(cnt)[3] for cnt in self.mask_contours_list)
                max_width = max(cv2.boundingRect(cnt)[2] for cnt in self.mask_contours_list)

                # Define spacing
                spacing_x = max_width + spacing  
                spacing_y = max_height + spacing
                
                num_rows = max(1, int((height_pic - 2 * self.margin) / spacing_y))
                num_columns = int((len(self.mask_contours_list) + num_rows - 1) // num_rows)
                width = num_columns * spacing_x + 2 * margin
                
                # Centres of the rectangles
                x_coords = [margin + i * spacing_x for i in range(num_columns)]
                y_coords = [margin + i * spacing_y for i in range(num_rows)]
                centers = [(x, y) for y in y_coords for x in x_coords]

                #endregion

                #region Prepare images bases for output pictures
                image_base= np.zeros((height_pic, width, 3), dtype=np.uint8)
                measure_pic = image_base.copy()
                elipse_pic=image_base.copy()
                circ_pic=image_base.copy()
                widths_pic=image_base.copy()
                #endregion
 

                #region Loop over the contours of a picture
                # Contour list
                all_almond_contours=[]
                count=1
                for contour in self.mask_contours_list:
                    try:
                        #region Conditions to avoid errors for fake contours
                        if cv2.contourArea(contour) < self.limit_area_pixels_min:
                            continue

                        # Functions fail if a contour has fewer than 5 points; this usually happens when it is too small or irregular.
                        if  len(contour) < 5:
                            continue
                        #endregion

                        #region Length, width , dist_max, order almonds 
                        
                        #Fit an ellipse to determine the orientation and turn the almond
                        (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
                        rotation_angle= 180-angle
                        M = cv2.moments(contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cnt_norm = contour-[cx, cy]

                        coordinates = cnt_norm[:, 0, :]
                        xs, ys = coordinates[:, 0], coordinates[:, 1]
                        thetas, rhos = cart2pol(xs, ys)
                                                
                        thetas_deg = np.rad2deg(thetas)
                        thetas_new_deg = (thetas_deg + rotation_angle) % 360
                        thetas_new = np.deg2rad(thetas_new_deg)

                        xs, ys = pol2cart(thetas_new, rhos)
                        cnt_norm[:, 0, 0] = xs
                        cnt_norm[:, 0, 1] = ys
                        cnt_rotated = cnt_norm + [cx, cy]
                        cnt_rotated = cnt_rotated.astype(np.int32)
                        
                        # Calculate displacement                     

                        cx_cen, cy_cen = centers[count-1]
                        dx = cx_cen - cx  # x displacement
                        dy = cy_cen - cy  # y displacement
                        # Rotate 
                        cnt_rotated= cnt_rotated - [cx, cy] + [cx_cen, cy_cen]
                        cnt_rotated = cnt_rotated.astype(np.int32)
                        cx_initial,cy_initial=[cx, cy]
                        cx, cy=[cx_cen, cy_cen]

                        
                        # Most distant point to rotate the almonds according to the tip 

                        dist_max = 0
                        most_distant_point = None

                        for point in cnt_rotated:
                            # Current point coordinates
                            x, y = point[0]
                            # Euclidean distance between point and centroid
                            dist_pc = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            
                            # Update most distant point
                            if dist_pc > dist_max:
                                dist_max = dist_pc
                                most_distant_point = (x, y)
                        turn=0
                        if most_distant_point[1] < cy:
                            # Normalize contour
                            contour_centered = cnt_rotated - [cx, cy]

                            # Rotate 180º 
                            contour_rotated = contour_centered * [-1, -1]

                            # Return the contour to the original centre
                            contour_rotated = contour_rotated + [cx, cy]
                            cnt_rotated = contour_rotated.astype(np.int32)
                            turn=180
                        
                        #Filled mask
                        filled=almonds.copy()


                        # Create rotation matrix
                        
                        M_rotation = cv2.getRotationMatrix2D((cx_initial, cy_initial), turn-rotation_angle, 1.0)

                        # Add traslation to rotation matrix 
                        M_rotation[0, 2] += dx  # X displacement
                        M_rotation[1, 2] += dy  # Y displacement
                        image_transformed = cv2.warpAffine(filled, M_rotation, (image_base.shape[1], image_base.shape[0]))
                        
                        # Create mask
                        mask_transformed = np.zeros(image_transformed.shape[:2], dtype=np.uint8)
                        # Draw element in the mask
                        mask_transformed=cv2.drawContours(mask_transformed, [cnt_rotated], -1, 255, thickness=cv2.FILLED)

                        filled_almond = cv2.bitwise_and(image_transformed, image_transformed , mask=mask_transformed)
                        filled_almond = filled_almond[:, :image_base.shape[1], :]

                        
                        #Prepare output picture for lenght and width
                        measure_pic = cv2.bitwise_or(measure_pic, filled_almond)
                        measure_pic=cv2.drawContours(measure_pic, [cnt_rotated], 0, (0, 255, 0), 1)

                        box= cv2.boundingRect(cnt_rotated)
                        (xb,yb, w, h)= box
                        measure_pic=cv2.rectangle(measure_pic,(xb,yb),(xb+w,yb+h),self.line_color,self.line_thickness)
                        box= ((xb+(w/2),yb+(h/2)),(w,h), angle-180)
                        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box = np.array(box, dtype="int")
                        box = perspective.order_points(box)
                        
                        (tl, tr, br, bl) = box
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)
                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)
                        
                        
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        
                        #To decide which is lenght and which width
                        if dA < dB:
                            dB_temp= dB
                            dB=dA
                            dA=dB_temp
                        else:
                            pass

                        dimA = float(dA / pixelmetric)
                        dimB = float(dB / pixelmetric)
                        cv2.putText(measure_pic, f"N: {count}", (int(tltrX), int(tltrY+10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(measure_pic, "L:{:.2f}mm".format(dimA), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(measure_pic, "W:{:.2f}mm".format(dimB), (int(trbrX - 100), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        
                        #endregion
                        
                        #region Width 25, 50 y 75 
                        widths_pic=cv2.bitwise_or(widths_pic, filled_almond)

                        # Create a mask for width
                        mask_width = np.zeros(image_base.shape[:2], dtype=np.uint8)   

                        # Draw the contour filled 
                        cv2.drawContours(mask_width, [cnt_rotated], -1, 255, thickness=cv2.FILLED)  

                        # Crop the ROI to calculate the widths
                        ROI = mask_width[yb:yb+h, xb:xb+w]
                        
                        h_25=int(dA*0.25)
                        h_50=int(dA*0.50)
                        h_75=int(dA*0.75)

                        width_25=cv2.countNonZero(ROI[h_25:h_25 + 1, :])/pixelmetric
                        width_50=cv2.countNonZero(ROI[h_50:h_50 + 1, :])/pixelmetric
                        width_75=cv2.countNonZero(ROI[h_75:h_75 + 1, :])/pixelmetric
                        
                        green = (0, 255, 0)
                        # Draw lines at different heights
                        cv2.line(widths_pic, (xb, yb + h_25), (xb + w, yb + h_25), green, self.line_thickness)

                        cv2.line(widths_pic, (xb, yb + h_50), (xb + w, yb + h_50), green, self.line_thickness)

                        cv2.line(widths_pic, (xb, yb + h_75), (xb + w, yb + h_75), green, self.line_thickness)

                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_25)), (int(tltrX - 40), int(yb + h_25)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_50)), (int(tltrX - 40), int(yb + h_50)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(widths_pic, "{:.2f}mm".format(float(width_75)), (int(tltrX - 40), int(yb + h_75)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        
                        # Shoulder
                        
                        # For calculating the shoulder 
                        row = mask_width[yb + int(h * 0.25), :]  # Extract the complet row at 25%

                        start_col = np.argmax(row == 255)  # First white pixel
                        end_col = len(row) - np.argmax(row[::-1] == 255)  # Last white pixel
                        # Verify that pixels are consistent
                        if start_col < end_col:

                            ROI_shoulder = mask_width[yb:yb + int(h * 0.25), start_col:end_col]  # Obtain the ROI of the superior part
                            shoulder=calculate_vertical_symmetry(ROI_shoulder) #Calculate symmetry between the left and right part

                        else:
                            ROI_shoulder = None  
                            print("Shoulder_error")

                        #endregion

                        #region Symmetry x and y 
                        
                        symmetry_v=calculate_vertical_symmetry(ROI)
                        symmetry_h=calculate_horizontal_symmetry(ROI)

                        #endregion

                        #region Area, perimeter, hull-area and solidity
                        area=(cv2.contourArea(cnt_rotated))/(pixelmetric**2)
                        perimeter= (cv2.arcLength(cnt_rotated, True))/(pixelmetric)
                        hull = cv2.convexHull(cnt_rotated)
                        area_hull= (cv2.contourArea(hull))/(pixelmetric**2)
                        solidity = area/area_hull
                        #endregion

                        #region Circulatity and ellipse 
                        circ_pic = cv2.bitwise_or(circ_pic, filled_almond)
                        elipse_pic = cv2.bitwise_or(elipse_pic, filled_almond)
                        circularity=4*np.pi*(area/perimeter**2)
                        circularity = float(circularity)
                        (x,y),radius = cv2.minEnclosingCircle(cnt_rotated)
                        center = (int(x),int(y))
                        radius = int(radius)
                        # circ_pic=cv2.circle(circ_pic,center,radius,self.line_color,self.line_thickness)
                        # cv2.putText(circ_pic, "Circ: {:.2f}".format(circularity), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)

                        ellipse = cv2.fitEllipse(cnt_rotated)
                        ellipse_rat= ellipse[1][0]/ellipse[1][1]
                        elipse_pic= cv2.ellipse(elipse_pic, ellipse, self.line_color,self.line_thickness)
                        cv2.putText(elipse_pic, "Elip: {:.2f}".format(ellipse_rat), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        
                        #endregion

                        #region Color Average


                        lab_pic= cv2.cvtColor(almonds, cv2.COLOR_BGR2LAB)
                        mask_contour = np.zeros_like(lab_pic)
                        mask_contour=cv2.drawContours(mask_contour, [contour], -1, 255, -1)
                        mask_contour= cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)

                        color_pic=cv2.bitwise_and(lab_pic, lab_pic, mask=mask_contour)
                        l, a, b= cv2.split(color_pic)
                        
                        ml=(np.mean(l[mask_contour != 0]))*100/255
                        ma=(np.mean(a[mask_contour != 0]))-128
                        mb=(np.mean(b[mask_contour != 0]))-128
                        #endregion

                        #region binary_masks
                        
                        #Save contours for segmentation mask presentation in output picture
                        all_almond_contours.append(contour)


                        #Export binary masks
                        if self.binary_masks==True:

                                mask_binary = np.ones(image_base.shape[:2], dtype=np.uint8) * 255  # white_mask
                                # Draw element contour filled in black
                                cv2.drawContours(mask_binary, [cnt_rotated], -1, 0, thickness=cv2.FILLED)  # Rellenar con 0 (negro)
                                
                               # Prepare ROI using bounding box

                                ROI = mask_binary[yb:yb + h, xb:xb + w]

                                
                                if ROI.shape[1] > ROI.shape[0]: 
                                    scale_factor = self.binary_pixel_size / ROI.shape[1]
                                    new_w, new_h = self.binary_pixel_size, int(ROI.shape[0] * scale_factor)
                                else:  
                                    scale_factor = self.binary_pixel_size / ROI.shape[0]
                                    new_w, new_h = int(ROI.shape[1] * scale_factor), self.binary_pixel_size
                                    
                                # Resize using nearest-neighbor interpolation
                                ROI_resized = cv2.resize(ROI, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT)


                                # Create a white background mask
                                out_pic = np.ones((self.binary_pixel_size, self.binary_pixel_size), dtype=np.uint8) * 255

                                # Calculate coordinates to center the ROI
                                x_offset = (self.binary_pixel_size - new_w) // 2
                                y_offset = (self.binary_pixel_size - new_h) // 2

                                # Insert ROI
                                out_pic[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = ROI_resized

                                if self.blurring_binary_masks is True:
                                    # Add blurring
                                    out_pic = cv2.GaussianBlur(out_pic, (self.blur_binary_masks_value, self.blur_binary_masks_value), 0)
                                    _, out_pic_binary  = cv2.threshold(out_pic , self.threshold_binarization, 255, cv2.THRESH_BINARY)
                                else:
                                    _, out_pic_binary = cv2.threshold(out_pic, self.threshold_binarization, 255, cv2.THRESH_BINARY)

                                #region Flip binary mask according to tip orientation

                                # Find all the black pixels 
                                black_pixels = np.column_stack(np.where(out_pic_binary == 0))

                                if len(black_pixels) > 0:
                                    # Find the lowest black pixels
                                    max_y = np.max(black_pixels[:, 0])  

                                    # Filter the lowest black pixels 
                                    pixels_last_row = black_pixels[black_pixels[:, 0] == max_y]

                                    # Find the pixels in the most right part 
                                    pixel_most_right = pixels_last_row[np.argmax(pixels_last_row[:, 1])]

                                if pixel_most_right[1] > self.binary_pixel_size/2:
                                    out_pic_binary  = cv2.flip(out_pic_binary, 1)

                                cv2.imwrite(f'{self.path_export_4}/{name_pic}_{count}.jpg', out_pic_binary)
                                #endregion

                        #endregion
                    
                    except Exception as e:
                        print(f"Error with picture {name_pic}", f"Fruit number: {count}", e)
                        error_list.append([name_pic, count, e])
                        traceback.print_exc()
                        row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,None, None, None, None, None,None, None,None, None, None, None, None, None, None, None, None, None]],
                        columns=morphology_table.columns)
                        morphology_table = pd.concat([morphology_table, row], ignore_index=True)
                        count=count+1
                        
                                

                    #Write_results
                    # row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,dimA, dimB, width_25[0], width_50[0], width_75[0],area[0], perimeter[0],area_hull[0], solidity[0], circularity, ellipse_rat, ml, ma, mb, symmetry_h, symmetry_v, shoulder]],
                    #                 columns=morphology_table.columns)
                    row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,dimA, dimB, width_25, width_50, width_75,area, perimeter,area_hull, solidity, circularity, ellipse_rat, ml, ma, mb, symmetry_h, symmetry_v, shoulder]],
                                    columns=morphology_table.columns)
                    
                    morphology_table = pd.concat([morphology_table, row], ignore_index=True)

                    count=count+1
                #endregion end loop over contours
                

                #region Exporting picture results 

                row_general = self.info_file.loc[self.info_file['Sample_picture'] == name_pic]
                row_general=row_general.values.flatten().tolist()
                row_general.append(count-1)

                row_general=pd.DataFrame([row_general], columns=general_table.columns)
                general_table=pd.concat([general_table,row_general], ignore_index=True)

                
                #Draw the contours for green masking
                almonds_mask= np.zeros((height_pic, width_pic), dtype=np.uint8)
                cv2.drawContours(almonds_mask, all_almond_contours, -1, 255, thickness=cv2.FILLED)
                


                #Put a transparent  green mask over the elements
                green_color = np.zeros_like(almonds)
                green_color[:] = [0, 255, 0]  

                
                alpha = 0.05  
                colored_mask = cv2.bitwise_and(green_color, green_color, mask=almonds_mask )
                original_weighted = cv2.multiply(almonds, 1 - alpha)

                almonds_masked= cv2.add(original_weighted, colored_mask)

                output_pic=np.concatenate((almonds_masked, measure_pic, widths_pic, circ_pic, elipse_pic), axis=1)
                cv2.imwrite(f"{self.path_export_3}/rs_{name_pic}", output_pic)
                #endregion 

            except Exception as e:
                print(f"Error with picture {name_pic}", e)
                traceback.print_exc()
                error_list.append([name_pic, "General", e])
                
        #endregion loop over picture
        #region Export session results    
        morphology_table = pd.merge(morphology_table, general_table[['Sample_picture', 'ID']], left_on='Sample_picture', right_on='Sample_picture', how='left')
        if self.binary_masks is True:
            binary_table = morphology_table[['Sample_picture', 'ID', 'Fruit_number']]
            binary_table['Binary_mask_picture'] = binary_table['Sample_picture'] + '_' + binary_table['Fruit_number'].astype(str)
            binary_table = binary_table[['Binary_mask_picture', 'ID']]
            binary_table.to_csv(self.path_export_5, mode='w', header=True, index=False, sep='\t')
            self.binary_table=binary_table

        morphology_table.to_csv(self.path_export_1, mode='w', header=True, index=False, sep='\t')
        general_table.to_csv(self.path_export_2, mode='w', header=True, index=False, sep='\t')

        self.morphology_table=morphology_table
        self.general_table=general_table
        #endregion 


    def measure_general(self, margin=100, spacing=30, limit_area_pixels_min=600, line_thickness=1, text_size=0.75, text_thickness=2,
                         text_color=(255,255,255), line_color=(0,255,0)):
        self.margin=margin
        self.spacing=spacing
        self.limit_area_pixels_min=limit_area_pixels_min
        self.text_size=text_size
        self.line_thickness=line_thickness
        self.text_thickness=text_thickness
        self.text_color=text_color
        self.line_color=line_color

        """
        Measures general (any fruit or seed) and exports results.

        This method sets measurement parameters for fruit analysis, including 
        margin size, spacing, and minimum area threshold for valid detections. 
        It processes the detected fruits and generates output files, such as 
        CSV reports and images with annotations.

        Parameters:
            margin (int, optional): The margin size around the fruits in pixels. Defaults to 100.
            spacing (int, optional): The spacing between detected fruits in pixels. Defaults to 30.
            limit_area_pixels_min (int, optional): The minimum area in pixels for an fruit to be considered valid. Defaults to 600.
            line_thickness (int, optional): Thickness of the annotation lines in output images. Defaults to 1.
            text_size (float, optional): Font size for text annotations. Defaults to 0.75.
            text_thickness (int, optional): Thickness of the text in annotations. Defaults to 2.
            text_color (tuple, optional): Color of the text annotations in BGR format. Defaults to (255, 255, 255) (white).
            line_color (tuple, optional): Color of the annotation lines in BGR format. Defaults to (0, 255, 0) (green).

        Outputs:
            - Saves measurement results in CSV format:
                - Morphological data: `results_morphology.txt`
                - General statistics: `results_general.txt`
                - Binary masks information: `binary_masks_info_table.txt`
            - Exports images:
                - Processed fruit images to `pic_results/`
                - Binary mask images to `binary_masks/`
                - Annotated images with measurements and contours

        Returns:
            None
        """
        
        #region Init table results 
        morphology_table=pd.DataFrame()
        morphology_table = pd.DataFrame(columns=["Project_name","Sample_picture","Fruit_name", "Fruit_number",
                                                 "Length","Width","Area", "Perimeter","Hull_area",
                                                 "Solidity","Circularity","Ellipse_Ratio","L","a","b"])
        general_table=pd.DataFrame()
        general_table = pd.DataFrame(columns=["Name_picture","Sample_number","ID","Session","Pixelmetric","Sample_picture","N_fruits"])
        #endregion

        #region Loop over the pictures 
        error_list=[]
        for picture in self.segmentation_input:
            try:
                #region Post-segmentation processing
                if self.sahi==True: #  SAHI approach # Transform contour format to opencv format
                    name_pic=os.path.basename(picture[1])
                    mask_contours_list=picture[0].object_prediction_list
                    fruits=cv2.imread(picture[1])
                    sahi_contours_list=[]
                    for contour in mask_contours_list:
                        contour=contour.mask.segmentation[0]
                        array_contour=np.array(contour)
                        array_contour=array_contour.reshape(-1,2)
                        contour_pixels = array_contour.astype(np.int32)
                        contour = contour_pixels.reshape((-1, 1, 2))
                        sahi_contours_list.append(contour)
                    self.mask_contours_list=sahi_contours_list
                    
                else:
                    # For slice_predict_reconstruct approach simply define variables and find_contours
                    name_pic=os.path.basename(picture[1])
                    mask=picture[0]

                    if self.smoothing==True:

                        mask=smoothing_masks(mask=mask,smoothing_iterations=self.smooting_iterations, smoothing_kernel=self.smoothing_kernel)[0]

                    if self.watershed==True:
                        mask, rect_kernel=smoothing_masks(mask=mask,smoothing_iterations=self.smooting_iterations, smoothing_kernel=self.smoothing_kernel)
                        mask=watershed(mask=mask, rect_kernel=rect_kernel, iterations=self.watershed_iterations,
                                       kernel_watershed=self.kernel_watershed, threshold_watershed=self.threshold_watershed)

                        self.mask_contours_list, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        fruits=cv2.imread(picture[1])
                        

                # Obtain pixelmetric values
                pixelmetric = self.info_file.loc[self.info_file['Sample_picture'] == name_pic, 'Pixelmetric'].values[0]
                
                #endregion
            
                #region Prepare rectangles size for contours

                height_pic, width_pic = fruits.shape[:2]
                max_height = max(cv2.boundingRect(cnt)[3] for cnt in self.mask_contours_list)
                max_width = max(cv2.boundingRect(cnt)[2] for cnt in self.mask_contours_list)

                # Define spacing
                spacing_x = max_width + spacing  
                spacing_y = max_height + spacing
                
                num_rows = max(1, int((height_pic - 2 * self.margin) / spacing_y))
                num_columns = int((len(self.mask_contours_list) + num_rows - 1) // num_rows)
                width = num_columns * spacing_x + 2 * margin
                
                # Centres of the rectangles
                x_coords = [margin + i * spacing_x for i in range(num_columns)]
                y_coords = [margin + i * spacing_y for i in range(num_rows)]
                centers = [(x, y) for y in y_coords for x in x_coords]

                #endregion

                #region Prepare images bases for output pictures
                image_base= np.zeros((height_pic, width, 3), dtype=np.uint8)
                measure_pic = image_base.copy()
                elipse_pic=image_base.copy()
                circ_pic=image_base.copy()
                #endregion
 

                #region Loop over the contours of a picture
                # Contour list
                all_fruit_contours=[]
                count=1
                for contour in self.mask_contours_list:
                    try:
                        #region Conditions to avoid errors for fake contours
                        if cv2.contourArea(contour) < self.limit_area_pixels_min:
                            continue

                        # Functions fail if a contour has fewer than 5 points; this usually happens when it is too small or irregular.
                        if  len(contour) < 5:
                            continue
                        #endregion
                         #Save contours for segmentation mask presentation in output picture
                        all_fruit_contours.append(contour)
                        #region Length, width , dist_max, order fruit
                        
                        #Fit an ellipse to determine the orientation and turn the fruit
                        M = cv2.moments(contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        
                        # Calculate displacement                     

                        cx_cen, cy_cen = centers[count-1]
                        dx = cx_cen - cx  # x displacement
                        dy = cy_cen - cy  # y displacement
                        # Rotate 
                        contour= contour - [cx, cy] + [cx_cen, cy_cen]
                        contour = contour.astype(np.int32)
                        cx_initial,cy_initial=[cx, cy]
                        cx, cy=[cx_cen, cy_cen]

                                                
                        #Filled mask
                        filled=fruits.copy()


                        # Aplicar traslación a la imagen sin rotación
                        M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
                        image_transformed = cv2.warpAffine(filled, M_translation, (image_base.shape[1], image_base.shape[0]))

                        # Crear máscara
                        mask_transformed = np.zeros(image_transformed.shape[:2], dtype=np.uint8)
                        mask_transformed = cv2.drawContours(mask_transformed, [contour], -1, 255, thickness=cv2.FILLED)

                        filled_fruit = cv2.bitwise_and(image_transformed, image_transformed , mask=mask_transformed)
                        filled_fruit = filled_fruit[:, :image_base.shape[1], :]

                        
                        #Prepare output picture for lenght and width
                        measure_pic = cv2.bitwise_or(measure_pic, filled_fruit)
                        measure_pic=cv2.drawContours(measure_pic, [contour], 0, (0, 255, 0), 1)
                        
                        box= cv2.boundingRect(contour)
                        (xb,yb, w, h)= box
                        measure_pic=cv2.rectangle(measure_pic,(xb,yb),(xb+w,yb+h),self.line_color,self.line_thickness)
                        box = np.array([[xb, yb],
                            [xb + w, yb],
                            [xb + w, yb + h],
                            [xb, yb + h]
                        ], dtype="int")
                        box = perspective.order_points(box)
                        
                        (tl, tr, br, bl) = box
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)
                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)
                        
                        
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        

                        dimA = float(dA / pixelmetric)
                        dimB = float(dB / pixelmetric)
                        cv2.putText(measure_pic, f"N: {count}", (int(tltrX), int(tltrY+10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(measure_pic, "L:{:.2f}mm".format(dimA), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        cv2.putText(measure_pic, "W:{:.2f}mm".format(dimB), (int(trbrX - 100), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        
                        #endregion
                    

                        #region Area, perimeter, hull-area and solidity
                        area=(cv2.contourArea(contour))/(pixelmetric**2)
                        perimeter= (cv2.arcLength(contour, True))/(pixelmetric)
                        hull = cv2.convexHull(contour)
                        area_hull= (cv2.contourArea(hull))/(pixelmetric**2)
                        solidity = area/area_hull
                        #endregion

                        #region Circulatity and ellipse 
                        circ_pic = cv2.bitwise_or(circ_pic, filled_fruit)
                        elipse_pic = cv2.bitwise_or(elipse_pic, filled_fruit)
                        circularity=4*np.pi*(area/perimeter**2)
                        circularity = float(circularity)
                        (x,y),radius = cv2.minEnclosingCircle(contour)
                        center = (int(x),int(y))
                        radius = int(radius)
                        circ_pic=cv2.circle(circ_pic,center,radius,self.line_color,self.line_thickness)
                        cv2.putText(circ_pic, "Circ: {:.2f}".format(circularity), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)

                        ellipse = cv2.fitEllipse(contour)
                        ellipse_rat= ellipse[1][0]/ellipse[1][1]
                        elipse_pic= cv2.ellipse(elipse_pic, ellipse, self.line_color,self.line_thickness)
                        cv2.putText(elipse_pic, "Elip: {:.2f}".format(ellipse_rat), (int(tltrX - 50), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color, self.text_thickness)
                        
                        #endregion

                        #region Color Average


                        lab_pic= cv2.cvtColor(fruits, cv2.COLOR_BGR2LAB)
                        mask_contour = np.zeros_like(lab_pic)
                        mask_contour=cv2.drawContours(mask_contour, [contour], -1, 255, -1)
                        mask_contour= cv2.cvtColor(mask_contour, cv2.COLOR_BGR2GRAY)

                        color_pic=cv2.bitwise_and(lab_pic, lab_pic, mask=mask_contour)
                        l, a, b= cv2.split(color_pic)
                        
                        ml=(np.mean(l[mask_contour != 0]))*100/255
                        ma=(np.mean(a[mask_contour != 0]))-128
                        mb=(np.mean(b[mask_contour != 0]))-128
                        #endregion

                        #region binary_masks
                        



                        #Export binary masks
                        if self.binary_masks==True:

                                mask_binary = np.ones(image_base.shape[:2], dtype=np.uint8) * 255  # white_mask
                                # Draw element contour filled in black
                                cv2.drawContours(mask_binary, [contour], -1, 0, thickness=cv2.FILLED)  # Rellenar con 0 (negro)
                                
                                # Prepare ROI using bounding box
                                ROI = mask_binary[yb:yb + h, xb:xb + w]

                                
                                if ROI.shape[1] > ROI.shape[0]: 
                                    scale_factor = self.binary_pixel_size / ROI.shape[1]
                                    new_w, new_h = self.binary_pixel_size, int(ROI.shape[0] * scale_factor)
                                else:  
                                    scale_factor = self.binary_pixel_size / ROI.shape[0]
                                    new_w, new_h = int(ROI.shape[1] * scale_factor), self.binary_pixel_size

                                # Resize using nearest-neighbor interpolation
                                ROI_resized = cv2.resize(ROI, (new_w, new_h), interpolation=cv2.INTER_NEAREST_EXACT)


                                # Create a white background mask
                                out_pic = np.ones((self.binary_pixel_size, self.binary_pixel_size), dtype=np.uint8) * 255

                                # Calculate coordinates to center the ROI
                                x_offset = (self.binary_pixel_size - new_w) // 2
                                y_offset = (self.binary_pixel_size - new_h) // 2

                                # Insert ROI
                                out_pic[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = ROI_resized

                                if self.blurring_binary_masks is True:
                                    # Add blurring
                                    out_pic = cv2.GaussianBlur(out_pic, (self.blur_binary_masks_value, self.blur_binary_masks_value), 0)
                                    _, out_pic_binary  = cv2.threshold(out_pic , self.threshold_binarization, 255, cv2.THRESH_BINARY)
                                else:
                                    _, out_pic_binary = cv2.threshold(out_pic, self.threshold_binarization, 255, cv2.THRESH_BINARY)


                                cv2.imwrite(f'{self.path_export_4}/{name_pic}_{count}.jpg', out_pic_binary)
                                

                        #endregion
                    
                    except Exception as e:
                        
                        print(f"Error with picture {name_pic}", f"Fruit number: {count}", e)
                        error_list.append([name_pic, count, e])
                        traceback.print_exc()
                        row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,None, None, None, None, None,None,  None, None, None, None, None]],
                        columns=morphology_table.columns)
                        morphology_table = pd.concat([morphology_table, row], ignore_index=True)
                        count=count+1
                        
                                

                    #Write_results
                    row =pd.DataFrame([[self.project_name,name_pic, self.fruit, count,dimA, dimB, area, perimeter,area_hull, solidity, circularity, ellipse_rat, ml, ma, mb]],
                                    columns=morphology_table.columns)
                    
                    morphology_table = pd.concat([morphology_table, row], ignore_index=True)
                    
                    count=count+1
                    
                #endregion end loop over contours
                

                #region Exporting picture results 

                row_general = self.info_file.loc[self.info_file['Sample_picture'] == name_pic]
                row_general=row_general.values.flatten().tolist()
                row_general.append(count-1)

                row_general=pd.DataFrame([row_general], columns=general_table.columns)
                general_table=pd.concat([general_table,row_general], ignore_index=True)

                
                #Draw the contours for green masking
                fruits_mask= np.zeros((height_pic, width_pic), dtype=np.uint8)
                cv2.drawContours(fruits_mask, all_fruit_contours, -1, 255, thickness=cv2.FILLED)
                


                #Put a transparent  green mask over the elements
                green_color = np.zeros_like(fruits)
                green_color[:] = [0, 255, 0]  

                
                alpha = 0.05  
                colored_mask = cv2.bitwise_and(green_color, green_color, mask=fruits_mask )
                original_weighted = cv2.multiply(fruits, 1 - alpha)

                fruits_masked= cv2.add(original_weighted, colored_mask)

                output_pic=np.concatenate((fruits_masked, measure_pic, circ_pic, elipse_pic), axis=1)
                cv2.imwrite(f"{self.path_export_3}/rs_{name_pic}", output_pic)
                #endregion 

            except Exception as e:
                print(f"Error with picture {name_pic}", e)
                traceback.print_exc()
                error_list.append([name_pic, "General", e])
                
        #endregion loop over picture
        #region Export session results    
        morphology_table = pd.merge(morphology_table, general_table[['Sample_picture', 'ID']], left_on='Sample_picture', right_on='Sample_picture', how='left')
        if self.binary_masks is True:
            binary_table = morphology_table[['Sample_picture', 'ID', 'Fruit_number']]
            binary_table['Binary_mask_picture'] = binary_table['Sample_picture'] + '_' + binary_table['Fruit_number'].astype(str)
            binary_table = binary_table[['Binary_mask_picture', 'ID']]
            binary_table.to_csv(self.path_export_5, mode='w', header=True, index=False, sep='\t')
            self.binary_table=binary_table

        morphology_table.to_csv(self.path_export_1, mode='w', header=True, index=False, sep='\t')
        general_table.to_csv(self.path_export_2, mode='w', header=True, index=False, sep='\t')

        self.morphology_table=morphology_table
        self.general_table=general_table
        #endregion 
