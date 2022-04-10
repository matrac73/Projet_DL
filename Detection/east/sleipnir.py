import cv2, os, datetime, sys, ntpath, shutil, cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from shapely.geometry import Polygon
from screeninfo import get_monitors

class sleipnir(object):
    """
    OCR module for SN scanning, includes methods to retrain a model on specific data and evaluate model performances.
    
    Simon Beckouche simon.beckouche.ext@safrangroup.com
    SLS Digital HUB delivery team
    """
    def __init__(self, verbose=False, logging=True, log_path = os.path.abspath('log/'), write_output_images = True, show_ground_truth_boxes=True, show_output_boxes=True):
        """
        Initialize an instance
        
        Inputs
        verbose: boolean controling display of messages in prompt
        logging: boolean controling logging of messages in text file
        """
        self.timestamp = lambda : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # timestamp method
        self.verbose = verbose # enable prompt messages
        self.logging = logging # enable logging to file
        self.log_path = os.path.join(log_path, 'log_' + self.timestamp().replace(' ', '_').replace(':', '') + '.txt') # path for log file
        self.true_labels = []
        self.output_labels = []
        self.eval_script = []
        self.lambd = 10 # used to balance error terms
        self.max_error = -np.log(0.0001) + self.lambd * 1 # used for cases when no ROI is detected in an image
        self.angle_thresh = np.radians(5) # angle treshold for detection, in radian
        self.pos_thresh = .10 # centroid position threshold for detection as fraction of largest polygon edge length
        self.area_thresh = .06 # area ratio threshold for detection as fraction of largest polygon edge length
        self.write_output_images = write_output_images # flags to control if images with added detected ROIs should be writen on disk
        self.show_ground_truth_boxes = show_ground_truth_boxes # controls if ground truth ROIs are added to output images
        self.show_output_boxes = show_output_boxes # controls if output ROIs are added to output images
        if self.logging:
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            with open(self.log_path, "w+") as logfile:
                logfile.write("%s\n" % (msg))

    def get_true_labels(self, true_labels_path):
        self.true_labels = []
        for file in os.listdir(true_labels_path): # list of files in folder
            filename, extension = os.path.splitext(file)
            if extension.lower() == '.txt':
                self.true_labels.append(os.path.join(true_labels_path, file))
        self.log_message('Found ' + str(len(self.true_labels)) + ' ground truth label files')
        
    def get_eval_script(self, eval_script_path):
        if os.path.exists(eval_script_path):
            self.eval_script = eval_script_path
            msg = 'Found evaluation script'
        else:
            self.eval_script = []
            msg = 'Evaluation script not found'
        self.log_message(msg)
        
    def get_images(self, source_dir=None):
        """
        Find image files in test data path
        """
        files = []
        if source_dir is None:
            source_dir = self.input_images_path
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(source_dir):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        return files
    
    def get_txt(self, source_dir=None):
        """
        Find image files in test data path
        """
        files = []
        if source_dir is None:
            source_dir = self.input_images_path
        exts = ['txt']
        for parent, dirnames, filenames in os.walk(source_dir):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        return files
        
    def log_message(self,msg):
        timed_msg = self.timestamp() + ' - ' + msg
        if self.verbose:
            print(timed_msg)
        if self.logging:
            with open(self.log_path, "a+") as logfile:
                logfile.write("%s\n" % (timed_msg))
    
    def get_input_images(self,input_images_path):
        self.input_images_path = input_images_path
        self.input_images = []
        image_extensions = ['.jpg', '.jpeg', '.png']
        for file in os.listdir(input_images_path): # list of files in folder
            filename, extension = os.path.splitext(file)
            if extension.lower() in image_extensions:
                self.input_images.append(os.path.join(input_images_path, file))
        self.log_message('Found ' + str(len(self.input_images)) + ' input images')
        
    def get_model_checkpoint(self, model_checkpoint_path):
        if os.path.exists(os.path.join(model_checkpoint_path, 'checkpoint')):
            self.model_checkpoint_path = model_checkpoint_path
            msg = 'Found model checkpoint'
        else:
            self.model_checkpoint_path = ''
            msg = 'Model checkpoint not found'
        self.log_message(msg)
        
    def get_output_dir(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_dir = output_path
        msg = 'Output directory set'
        self.log_message(msg)
    
    def add_polygons_to_image(self, image_input, image_output, poly_str_list, color = 'red'):
        """
        Add list of polygons to image
        """
        im = cv2.imread(image_input)[:, :, ::-1] # reading image
        if color == 'red':
            color_tuple = (0, 0, 255)
        elif color == 'blue':
            color_tuple = (255, 200, 0)
        elif color == 'green':
            color_tuple = (0, 255, 0)
        elif color=='yellow':
            color_tuple = (0, 215, 255)
        elif color=='purple':
            color_tuple = (238, 130, 238)
        for poly_str in poly_str_list:
            box = self.polyarray_from_str(poly_str)
            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color_tuple, thickness=2)
        cv2.imwrite(image_output, im[:, :, ::-1])
        
    def replace_output_images(self):
        """
        Replaces output images with originals before adding ROIs
        """
        input_images = self.get_images() # reading content of input folder
        nim = 0
        for image_path in input_images:
            path_root, full_filename = os.path.split(image_path) # splitting containig folder and full file name
            shutil.copy(image_path, os.path.join(self.output_dir, full_filename)) # moving image to output directory
            nim += 1
        msg = 'Moved ' + str(nim) + ' images to output directory'
        self.log_message(msg)
                
    def add_true_labels_to_images(self):
        """
        Add ground truth labels to processed images
        """
        image_list = self.get_images(self.output_dir) # listing images in output folder
        label_list = self.true_labels
        npoly = 0 # polygon counter
        for image_path in image_list:
            path_root, full_filename = os.path.split(image_path) # splitting containig folder and full file name
            filename, extension = os.path.splitext(full_filename) # splitting file name and extension
            if filename + '.txt' in [ntpath.basename(label) for label in label_list]: # checking if a label file was generated
                true_label_index = [ntpath.basename(label) for label in label_list].index(filename + '.txt')
                true_label = label_list[true_label_index]
                with open(true_label) as true_f:
                    true_poly_txt = true_f.readlines() # content of ground truth label text file
                fully_detected_poly_str_list = []
                partially_detected_poly_str_list = []
                undetected_poly_str_list = []
                for poly_str in true_poly_txt: # filtering ROIs detected fully, partially, and missed
                    if self.fully_detected_true_labels[npoly]: # case where ROI has been fully detected
                        fully_detected_poly_str_list.append(poly_str)
                    elif self.partially_detected_true_labels[npoly]: # case where ROI has been partially detected
                        partially_detected_poly_str_list.append(poly_str)
                    else: # case where ROI was missed
                        undetected_poly_str_list.append(poly_str)
                    npoly += 1
                output_image = os.path.join(self.output_dir, full_filename)
                self.add_polygons_to_image(image_path, output_image, fully_detected_poly_str_list, color='green')
                self.add_polygons_to_image(image_path, output_image, partially_detected_poly_str_list, color='yellow')
                self.add_polygons_to_image(image_path, output_image, undetected_poly_str_list, color='red')
                
    def add_output_labels_to_images(self):
        """
        Add output truth labels to processed images
        """
        image_list = self.get_images(self.output_dir) # listing images in output folder
        label_list = self.output_labels
        npoly = 0 # polygon counter
        for image_path in image_list:
            path_root, full_filename = os.path.split(image_path) # splitting containig folder and full file name
            filename, extension = os.path.splitext(full_filename) # splitting file name and extension
            if filename + '.txt' in [ntpath.basename(label) for label in label_list]: # checking if a label file was generated
                output_label_index = [ntpath.basename(label) for label in label_list].index(filename + '.txt')
                output_label = label_list[output_label_index]
                with open(output_label) as output_f:
                    output_poly_txt = output_f.readlines() # content of output label text file
                false_positive_poly_str_list = []
                true_positive_poly_str_list = []
                for poly_str in output_poly_txt: # filtering ROIs detected fully, partially, and missed
                    if self.false_positive_output_labels[npoly]: # case where ROI is false positive
                        false_positive_poly_str_list.append(poly_str)
                    else: # case where ROI is true positive
                        true_positive_poly_str_list.append(poly_str)
                    npoly += 1
                output_image = os.path.join(self.output_dir, full_filename)
                self.add_polygons_to_image(image_path, output_image, true_positive_poly_str_list, color='blue')
                self.add_polygons_to_image(image_path, output_image, false_positive_poly_str_list, color='purple')

    def get_output_labels(self, label_path = None):
        if label_path is None:
            label_path = self.output_dir
        self.output_labels = []
        for file in os.listdir(self.output_dir): # list of files in folder
            filename, extension = os.path.splitext(file)
            if extension.lower() == '.txt':
                self.output_labels.append(os.path.join(self.output_dir, file))     
        msg = 'Found ' + str(len(self.output_labels)) + ' output label files'
        self.log_message(msg)
    
    def clean_folder(self, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                msg = ('Failed to delete %s. Reason: %s' % (file_path, e))
                self.log_message(msg)
    
    def save_anchor_point_RBOX(event,x,y,flags,param):
        """
        Callback function for image labeling
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            drawing_color_ROI = [0, 0, 255] # using red as color for ROIs
            drawing_color_anchors = [255, 0, 0] # using blue as color for Anchors
            # Unpacking parameters
            img = param['img']
            dspl = param['dspl']
            anchors = param['anchors']
            anchors[-1].append([x,y]) # adding additional corner to ROI
            cv2.circle(img,(x,y),2,drawing_color_anchors,-1)
            cv2.imshow(dspl,img)
            if len(anchors[-1]) == 3: # case where the last corner of an ROI has been saved
                x1 = anchors[-1][0][0]
                x2 = anchors[-1][1][0]
                x3 = anchors[-1][2][0]
                y1 = anchors[-1][0][1]
                y2 = anchors[-1][1][1]
                y3 = anchors[-1][2][1]
                xv = x2 - x1
                yv = y2 - y1
                s = np.sqrt(xv**2+yv**2)
                d = ((x3-x1)*xv+(y3-y1)*yv)/s
                xh = int(x1 + xv*d/s) # calcul de la projection du point sur la droite pour fair un rectangle
                yh = int(y1 + yv*d/s)
                anchors[-1][1] = [xh,yh]
                anchors[-1].append([x1-(xh-x3),y1-(yh-y3)])
                label = input() # saving ROI label
                anchors[-1].append(label)
                poly_pts = np.int32(np.asarray(anchors[-1][:4]))
                cv2.polylines(img, [poly_pts], True, drawing_color_ROI)
                cv2.imshow(dspl,img)
                anchors.append([]) # initializing new ROI
    
    def save_anchor_point_trapeze(event,x,y,flags,param):
        """
        Callback function for image labeling
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            drawing_color = [0, 0, 255] # using red as color for ROIs
            # Unpacking parameters
            img = param['img']
            dspl = param['dspl']
            anchors = param['anchors']
            anchors[-1].append([x,y]) # adding additional corner to ROI
            cv2.circle(img,(x,y),2,drawing_color,-1)
            cv2.imshow(dspl,img)
            if len(anchors[-1]) == 4: # case where the last corner of an ROI has been saved
                label = input() # saving ROI label
                anchors[-1].append(label)
                poly_pts = np.int32(np.asarray(anchors[-1][:4]))
                cv2.polylines(img, [poly_pts], True, drawing_color)
                cv2.imshow(dspl,img)
                print(anchors)
                anchors.append([]) # initializing new ROI
                
    def save_anchor_point_rectangle(event,x,y,flags,param):
        """
        Callback function for image labeling
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            drawing_color = [0, 0, 255] # using red as color for ROIs
            # Unpacking parameters
            img = param['img']
            dspl = param['dspl']
            anchors = param['anchors']
            anchors[-1].append([x,y]) # adding additional corner to ROI
            cv2.circle(img,(x,y),2,drawing_color,-1)
            cv2.imshow(dspl,img)
            if len(anchors[-1]) == 2: # case where the last corner of an ROI has been saved
                # Reorder the point's coordinates
                x1 = anchors[-1][0][0]
                y1 = anchors[-1][0][1]
                x2 = anchors[-1][1][0]
                y2 = anchors[-1][1][1]
                anchors[-1] = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                # Passing Label
                label = input() # saving ROI label
                anchors[-1].append(label)
                poly_pts = np.int32(np.asarray(anchors[-1][:4]))
                cv2.polylines(img, [poly_pts], True, drawing_color)
                cv2.imshow(dspl,img)
                print(anchors)
                anchors.append([]) # initializing new ROI


    def polygon_from_str(self, poly_str):
        """
        Create a shapely polygon object from gt or dt line.
        this function makes the shapely polygon object to be used in the later function. 
        input here is each line in "img_*.txt" files of the dataset
        """
        polygon_points = [float(o) for o in poly_str.split(',')[:8]]
        polygon_points = np.array(polygon_points).reshape(4, 2)
        polygon = Polygon(polygon_points).convex_hull
        return polygon
    
    def polyarray_from_str(self, poly_str):
        """
        Create a numpy array polygon from gt or dt line.
        """
        fields = poly_str.split(',') # extracting coma separated values in poly string
        X = [float(fields[i]) for i in [0,2,4,6]] # extracting corner Y coordinates
        Y = [float(fields[i]) for i in [1,3,5,7]] # extracting corner X coordinates
        polyarray = np.array(list(zip(X,Y))) # zipping and converting to numpy array
        return polyarray
    
    def polygon_iou(self, poly_str1, poly_str2):
        """
        Intersection over union between two polygon strings
        """
        poly1 = self.polygon_from_str(poly_str1)
        poly2 = self.polygon_from_str(poly_str2)
        if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
            iou = -np.log(0.0001)
        else:
            try:
                inter_area = poly1.intersection(poly2).area
                union_area = poly1.area + poly2.area - inter_area
                iou = -np.log(float(inter_area) / union_area)
            except shapely.geos.TopologicalError:
                msg = 'shapely.geos.TopologicalError occured, iou set to 0'
                self.log_message(msg)
                iou = -np.log(0.0001)
        return iou

    def polygon_angle(self, poly_str1, poly_str2):
        """
        Angular error between two polygon represented as strings
        """
        fields1 = poly_str1.split(',') # extracting coma separated values in first poly string
        Y1 = [float(fields1[i]) for i in [0,2,4,6]] # extracting corner Y coordinates
        X1 = [float(fields1[i]) for i in [1,3,5,7]] # extracting corner X coordinates
        d1 = np.array([X1[1] - X1[0], Y1[1] - Y1[0]]) # direction of first poly
        fields2 = poly_str2.split(',') # extracting coma separated values in second poly string
        Y2 = [float(fields2[i]) for i in [0,2,4,6]] # extracting corner Y coordinates
        X2 = [float(fields2[i]) for i in [1,3,5,7]] # extracting corner X coordinates
        d2 = np.array([X2[1] - X2[0], Y2[1] - Y2[0]]) # direction of second poly
        cos_angle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        Langle = 1 - cos_angle
        return Langle
    
    def polygon_distance(self, poly_str1, poly_str2):
        """
        Compute distance between polygon strings as combination of angular and IOU
        """
        Laabb = self.polygon_iou(poly_str1, poly_str2)
        Langle = self.polygon_angle(poly_str1, poly_str2)
        Ltotal = Laabb + self.lambd * Langle
        return Ltotal
    
    def find_closest_polygon(poly_str1, poly_str_list):
        """
        Find highest score neighbor of polygon string in list of polygon string
        """
        poly1 = self.polyarray_from_str(poly_str1)
        min_dist = np.Inf # initializing distance as infinite
        min_index = -1 # initializing score minimizing index
        for poly_str2 in poly_str_list:
            poly2 = self.polyarray_from_str(poly_str2)
            dist = self.polygon_distance(poly_str1, poly_str2)
            if dist < min_dist: # case where new highest score neighbor has been found
                min_dist = dist # storing new minimum distance
                min_index = poly_str_list.index(poly_str2) # storing index of closest neighbor
        return min_dist, min_index
    
    def check_polygon_intersection(self, poly_str1, poly_str2):
        """
        Checks if two polygons intersect or not
        """
        poly1 = self.polygon_from_str(poly_str1)
        poly2 = self.polygon_from_str(poly_str2)
        return poly1.intersects(poly2)
    
    def check_polygon_identity(self, poly_str1, poly_str2):
        """
        Checks if two polygons are close enough to be considered identical
        """
        
        Langle = self.polygon_angle(poly_str1, poly_str2) # computing angle related loss
        angle_delta = np.arccos(1 - Langle) - np.pi/2
        poly1 = self.polygon_from_str(poly_str1) # converting polygon to shapely object
        poly2 = self.polygon_from_str(poly_str2) # converting polygon to shapely object
        polyarray1 = self.polyarray_from_str(poly_str1) # converting polygon to numpy array
        polyarray2 = self.polyarray_from_str(poly_str2) # converting polygon to numpy array
        areas = [poly1.area, poly2.area] # areas of both polygons
        area_ratio = min(areas) / max(areas) # computing smallest area over largest area
        largest_edge = max([np.linalg.norm(polyarray1[1] - polyarray1[0]), np.linalg.norm(polyarray1[2] - polyarray1[1]), # length of longest edge among both polygons
                  np.linalg.norm(polyarray2[1] - polyarray2[0]), np.linalg.norm(polyarray2[2] - polyarray2[1])])
        centroid1 = np.array(poly1.centroid) # extracting poly1 centroid
        centroid2 = np.array(poly2.centroid) # extracting poly2 centroid
        centroid_delta = np.linalg.norm(centroid1 - centroid2) # computing centroid distance
        centroid_thresh = largest_edge * self.pos_thresh # threshold used for centroid delta
        area_thresh = np.power(1 - self.area_thresh, 2)
        identical_polygons = angle_delta < self.angle_thresh and area_ratio > area_thresh and centroid_delta < centroid_thresh
        return identical_polygons    
    
    def remove_empty_label_lines(self):
        """
        Removes empty lines in output label files
        """
        for output_label in self.output_labels:
            with open(output_label) as output_f:
                    output_poly_txt = output_f.readlines() # content of output label text file
            new_poly_txt = [] # initializing new list of ROIs
            for poly_str in output_poly_txt:
                if poly_str.strip('\n'): # case where line is empty after removing line break
                    new_poly_txt.append(poly_str) # passing non empty lines
            file_path, file_full_filename = os.path.split(output_label)
            filename, file_extension = os.path.splitext(file_full_filename)
            temp_filename = file_path + filename + '_temp' + file_extension # temporary filename of new label
            with open(temp_filename, 'w') as f: # generating cleaned label file
                for poly_str in new_poly_txt:
                    f.write("%s\n" % poly_str.rstrip('\n'))
            shutil.move(temp_filename, output_label)

    def defragment_label_polygons(self):
        """
        Checks for fragmented polygons in every output label and fuse them together
        """
        nfusion = 0 # counter for number of polygons fused
        for output_label in self.output_labels:
            with open(output_label) as output_f:
                    output_poly_txt = output_f.readlines() # content of output label text file
            length_before_fusion = len(output_poly_txt)
            fused_poly_txt = self.fuse_connected_polygons(output_poly_txt) # fusing intersecting polygons
            nfusion += length_before_fusion - len(fused_poly_txt) # incrementing counter
            file_path, file_full_filename = os.path.split(output_label)
            filename, file_extension = os.path.splitext(file_full_filename)
            temp_filename = file_path + filename + '_temp' + file_extension # temporary filename of new label
            with open(temp_filename, 'w') as f: # generating cleaned label file
                for poly_str in fused_poly_txt:
                    f.write("%s\n" % poly_str.rstrip('\n'))
            shutil.move(temp_filename, output_label) # overwriting previous label with new file
        if nfusion > 0:
            msg = 'Fused ' + str(nfusion) + ' fragmented output ROIs together'
        else:
            msg = 'No fragmented output ROI detected'
        self.log_message(msg)
    
    def fuse_connected_polygons(self, poly_str_list):
        """
        Fuse polygons that intersects
        """
        all_fused = False
        
        while all_fused == False :
            all_fused = True
            for poly_str1 in poly_str_list: # looping over every polygon string
                poly1 = self.polygon_from_str(poly_str1) # converting first poly string to shapely polygon
                for poly_str2 in poly_str_list[poly_str_list.index(poly_str1) + 1:]: # looping over tail of list
                    poly2 = self.polygon_from_str(poly_str2) # converting second poly string to shapely polygon
                    if poly1.intersects(poly2):
                        all_fused = False
                        break

            if all_fused == False:
                print("A")
                for poly_str1 in poly_str_list: # looping over every polygon string
                    poly1 = self.polygon_from_str(poly_str1) # converting first poly string to shapely polygon
                    for poly_str2 in poly_str_list[poly_str_list.index(poly_str1) + 1:]: # looping over tail of list
                        poly2 = self.polygon_from_str(poly_str2) # converting second poly string to shapely polygon
                        if poly1.intersects(poly2): # case where polys intersect each others

                            # Computing new bounding box
                            poly1 = self.polyarray_from_str(poly_str1) # converting string poly to array
                            poly2 = self.polyarray_from_str(poly_str2) # converting string poly to array

                            main_dir = poly1[1] - poly1[0] + poly2[1] - poly2[0] # length direction defined with second and third corners

                            # empêcher une composante de main d'être nulle
                            if main_dir[0] == 0:
                                main_dir[0] = 0.1
                            if main_dir[1] == 0:
                                main_dir[1] = 0.1

                            main_dir /= np.linalg.norm(main_dir) # making vector unitary
                            sec_dir = np.array([-main_dir[1], main_dir[0]]) # computing orthogonal direction
                            main_dir = np.sign(main_dir.dot([0, 1])) * main_dir # making first direction pointing toward increasing X for corner ordering
                            sec_dir = np.sign(sec_dir.dot([1, 0])) * sec_dir # making second direction pointing toward increasing Y for corner ordering
                            minX = min(poly1.dot(main_dir).min(), poly2.dot(main_dir).min())
                            maxX = max(poly1.dot(main_dir).max(), poly2.dot(main_dir).max())
                            minY = min(poly1.dot(sec_dir).min(), poly2.dot(sec_dir).min())
                            maxY = max(poly1.dot(sec_dir).max(), poly2.dot(sec_dir).max())
                            p1 = minX * main_dir + minY * sec_dir # top left corner
                            p2 = maxX * main_dir + minY * sec_dir # bottom left corner
                            p3 = maxX * main_dir + maxY * sec_dir # bottom right corner
                            p4 = minX * main_dir + maxY * sec_dir # top right corner
                            p1str = np.round(p1).astype(int).astype(str) # converting to integer string
                            p2str = np.round(p2).astype(int).astype(str) # converting to integer string
                            p3str = np.round(p3).astype(int).astype(str) # converting to integer string
                            p4str = np.round(p4).astype(int).astype(str) # converting to integer string
                            bounding_poly = p1str[0] + ',' + p1str[1] + ',' + p2str[0] + ',' + p2str[1] + ',' +  \
                            p3str[0] + ',' + p3str[1] + ',' + p4str[0] + ',' + p4str[1]
                            poly_str_list.remove(poly_str1) # removing fused poly from list
                            poly_str_list.remove(poly_str2) # removing fused poly from list
                            poly_str_list.append(bounding_poly) # adding new fused bounding box to list
                            break
        return poly_str_list
            
    def compute_false_positive_rate(self):
        """
        Computes rate of output ROIs that do not intersect any ground truth ROI
        """ 
        self.false_positive_rate = 0 # initializing rate at 0
        self.false_positive_output_labels = [] # list storing booleans indicating if output ROIs are false positive
        npoly = 0 # number of processed polygons
#         print(self.output_labels)
        for output_label in self.output_labels: # iterating over every output label
            true_labels_filename_list = [ntpath.basename(label) for label in self.true_labels] # list of output labels filenames
            tmp1 = ntpath.basename(output_label)
            tmp = true_labels_filename_list.index(tmp1)
            true_label = self.true_labels[tmp] # selecting associated true label
            with open(output_label) as output_f:
                    output_poly_txt = output_f.readlines() # content of output label text file
            with open(true_label) as true_f:
                    true_poly_txt = true_f.readlines() # content of true label text file
            for output_poly_str in output_poly_txt:
                is_false_positive = True # initializing polygon as false positive
                for true_poly_str in true_poly_txt:
                    try:
                        is_false_positive = is_false_positive and not self.check_polygon_intersection(output_poly_str, true_poly_str)
                    except:
                        pass
                self.false_positive_rate += is_false_positive
                self.false_positive_output_labels.append(is_false_positive)
                npoly += 1
        if npoly != 0:
            self.false_positive_rate /= npoly # normalizing rate            
        msg = 'False detection rate over ' + str(npoly) + ' computed ROIs ' + str(round(100*self.false_positive_rate, 1)) + '%'
        self.log_message(msg)
        return(self.false_positive_rate)
    
    def process_images(self):
        msg = 'Cleaning up output folder'
        self.log_message(msg)
        self.clean_folder(self.output_dir)
        msg = 'Processing images...'
        self.log_message(msg)
        run_command = sys.executable + ' ' + self.eval_script + ' --test_data_path=' + self.input_images_path + ' --checkpoint_path=' + self.model_checkpoint_path + \
                 ' --output_dir=' + self.output_dir + ' --gpu_list=0'
        a = os.system(run_command)
        msg = 'Processing complete'
        self.log_message(msg)
                
    def process_results(self):
        self.get_output_labels(self.output_dir) # import output labels path
        self.remove_empty_label_lines() # remove empty line in output labels
        self.defragment_label_polygons() # fuse connected polygons in output labels
        self.compute_average_error() # compute error and detection rates
        self.compute_false_positive_rate() # compute rate of false positive output labels
        if self.show_ground_truth_boxes or self.show_output_boxes:
            self.replace_output_images()
            msg = 'Adding ROI boxes to output images'
            self.log_message(msg)
        if self.show_ground_truth_boxes: # adding ground truth ROIs to output images as green boxes
            self.add_true_labels_to_images()
        if self.show_output_boxes: # adding output ROIs to output images as green boxes
            self.add_output_labels_to_images()
            
    def compute_average_error(self):
        """
        Computes average accuracy between detected ROIs and ground truth ROIs
        """
        self.avg_err = 0 # initializing average error
        self.partial_detect_rate = 0 # initializing counter of partially detected ROIs
        self.full_detect_rate = 0 # initializing counter of fully detected ROIs
        self.partially_detected_true_labels = [] # list storing booleans indicating if ground truth ROIs have been partially detected or not
        self.fully_detected_true_labels = [] # list storing booleans indicating if ground truth ROIs have been fully detected or not
        npoly = 0 # number of processed polygons
        for true_label in self.true_labels: # looping over every ground truth label files
            true_label_filename = ntpath.basename(true_label) # extracting basename to find associated file in list of ground truth labels
            output_labels_filename_list = [ntpath.basename(label) for label in self.output_labels] # list of output labels filenames
            if true_label_filename in output_labels_filename_list: # case where at least one ROI is available in output label files
                empty_label = False # marking image as not empty as at least one ROI was detected
                output_label_index = output_labels_filename_list.index(true_label_filename) # finding path of true label of corresponding filename
                output_label = self.output_labels[output_label_index] # extracting full path of associated ground truth label
                with open(output_label) as output_f:
                    output_poly_txt = output_f.readlines() # content of output label text file
            else:
                empty_label = True # marking image as empty to use maximum error for every ground truth ROIs in contains
            with open(true_label) as true_f:
                true_poly_txt = true_f.readlines() # content of true label text file
            # Measuring detection rate for ground truth ROIs
            for true_poly_str in [line for line in true_poly_txt if len(line) > 8]: # looking for closest neighbor in ground truth label file excluding line shorter than 8 characters
                    best_dist = np.Inf # initializing distance to infinite
                    partial_intersection = False # initializing partial intersection of current ground truth poly as false
                    total_detection = False # initializing total detection of current ground truth poly as false
                    if not empty_label:
                        for output_poly_str in [line for line in output_poly_txt if len(line) > 8]: # looping over all outout labels files excluding line shorter than 8 characters
                                current_dist = self.polygon_distance(output_poly_str, true_poly_str)
                                best_dist = min(best_dist, current_dist)
                                partial_intersection = partial_intersection or self.check_polygon_intersection(true_poly_str, output_poly_str)
                                total_detection = total_detection or self.check_polygon_identity(true_poly_str, output_poly_str)
                    else: # applying maximum penalties
                        best_dist = self.max_error
                        partial_intersection = False # marking ground truth ROI as not partially not detected
                        total_detection = False # marking ROI as not fully detected
                    self.partially_detected_true_labels.append(partial_intersection)
                    self.fully_detected_true_labels.append(total_detection)
                    self.avg_err += best_dist # adding score between output label and its closest neighbor to average error
                    self.partial_detect_rate += partial_intersection # adding partial intersection of current ground truth polygon
                    self.full_detect_rate += total_detection
                    npoly += 1 # incrementning polygon counter
        self.avg_err /= max(npoly, 1) # normalizing average accuracy
        self.partial_detect_rate /= max(npoly, 1) # normalizing partial detection rate
        self.full_detect_rate /= max(npoly, 1) # normalizing full detection rate
        err_msg = 'Average error over ' + str(npoly) + ' computed ROIs ' + str(round(self.avg_err,1))
        self.log_message(err_msg)
        partial_detect_msg = 'Partial detection rate for ground truth ROIs ' + str(round(100*self.partial_detect_rate, 1)) + '%'
        self.log_message(partial_detect_msg)
        full_detect_msg = 'Full detection rate for ground truth ROIs ' + str(round(100*self.full_detect_rate, 1)) + '%'
        self.log_message(full_detect_msg)
        return(self.avg_err, self.partial_detect_rate, self.full_detect_rate)
    
    def create_dataset_transcription(self,save_path):
        """
        Isolate RoI from images based on the ground truth label.
        """
#         print(self.output_dir)
#         print(self.get_images(self.output_dir))
        image_list = self.get_images(self.output_dir) # listing images in output folder
#         print(image_list)
        label_list = self.true_labels
#         print(label_list)
        for image_path in image_list:
            path_root, full_filename = os.path.split(image_path) # splitting containig folder and full file name
            filename, extension = os.path.splitext(full_filename) # splitting file name and extension
            img = cv2.imread(image_path) # loading image
            
            if filename + '.txt' in [ntpath.basename(label) for label in label_list]: # checking if a label file was generated
                true_label_index = [ntpath.basename(label) for label in label_list].index(filename + '.txt')
                true_label = label_list[true_label_index]
                with open(true_label) as true_f:
                    true_poly_txt = true_f.readlines() # content of ground truth label text file
                count_roi = 0
                for roi in true_poly_txt:
                    roi = roi.strip() # removing the \n from the strings
                    roi = roi.split(',') # creating a list of coordinates from the string
                    fichier = open(save_path+'/'+filename+'_'+str(count_roi)+'.gt.txt', "w") # create the .txt
                    fichier.write(str(roi[-1])) # fill the .txt
                    fichier.close() # close the file
                    roi = list(map(int,roi[:-1])) # getting integers instead of string
                    cnt = np.array([[[roi[2*i],roi[2*i+1]]] for i in [0,1,2,3]]) # modifying structure of the data in order to match the following code expectations
                    rect = cv2.minAreaRect(cnt) # rectangle minimizing the area
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    width = int(rect[1][0])
                    height = int(rect[1][1])
                    src_pts = box.astype("float32")
                    dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(img, M, (width, height))
                    if width < height:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(save_path+'/'+filename+'_'+str(count_roi)+'.jpg',warped) # save the image of roi
                    count_roi += 1

    def add_right_roi_to_dataset(self,output_path,input_path):
        """
        Isolate RoI which are right sided in order to start training our models
        """
        image_list = self.get_images(self.output_dir) # listing images in the input folder
        for image_path in image_list:
            path_root, full_filename = os.path.split(image_path)
            filename, extension = os.path.splitext(full_filename)
            img = cv2.imread(image_path) # loading image
            plt.figure()
            plt.imshow(img) # showing image
            plt.show()
            bon_sens = input("Est ce que l'image est droite ?") # asking the user if the image is right sided
            if bon_sens == 'n':
                print('image pas ajouté') # do not copy
            else :
                print('image ajouté !')
                cv2.imwrite(output_path+'/'+full_filename,img) # copy image
                shutil.copyfile(input_path+'/'+filename+'.txt', output_path+'/'+filename+'.gt.txt') # copy .txt
            plt.close()
        
    def setting_up_labeling(self,data_path):

        img_exts = ['.jpeg'] # image formats
        img_list = []
        label_list = []
        file_list = os.listdir(data_path) # files in current folder
        print('Labeling data in ', data_path)
        for file in file_list:
            filename, file_extension = os.path.splitext(file) # extracting file extension
            if file_extension.lower() in img_exts: # checking if file is an image
                img_list.append(os.path.join(data_path,file))
                if os.path.exists(os.path.join(data_path,filename + '.txt')): # case where label is available for image
                    label_list.append(os.path.join(data_path,filename + '.txt'))
        Nimg = len(img_list) # number of images
        Nlabel = len(label_list) # number of labels
        print('Found ',Nimg, 'valid images including', Nlabel, 'labeled images')
        return img_list
        
    