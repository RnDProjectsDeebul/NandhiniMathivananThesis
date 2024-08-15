#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv_bridge

import copy
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray

########
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats 
import pandas as pd
import time

class Visualizer:
    """ """

    def __init__(self):
        """ Constructor of Visualizer class

        - define the class variables (ROS publisher and subscribers etc.)

        """
         ####nmathi2s####
        self.model = torch.jit.load("/home/nandhini/thesis/Grasp/Gaussian/final_gaussian.pt")
      
        self.scaler = StandardScaler()
        data = [[640, 480], [1, 1]]
        self.scaler.fit(data)

        # publisher for transform point cloud
        self.pc_pub = rospy.Publisher("/transformed_point_cloud", PointCloud2, queue_size=10)

        # publisher for pose array
        self.pose_array_pub = rospy.Publisher("/pose_array", PoseArray, queue_size=10)

        self.rgb_topic = '/camera/color/image_raw'
        self.pc_topic = 'input_pointcloud_topic'

        self.bridge = cv_bridge.CvBridge()

        self.rgb_image = None
        self.pc = None

        # Subscribers with depth information
        
        #self.image_sub = message_filters.Subscriber(self.rgb_topic, Image)
        #self.pc_sub = message_filters.Subscriber(self.pc_topic, PointCloud2)
        #sync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 10, 0.2, allow_headerless=True)
        #sync.registerCallback(self.perceive)
        
        # subscribe to rgb 
        # Set up your subscriber and define its callback
        #############
        #rospy.Subscriber(self.rgb_topic, Image, self.image_callback)
        ##############
        # Publisher
        self.image_pub = rospy.Publisher(
            '/image_with_keypoints', Image, queue_size=100)

        
    def image_callback(self, img_msg) -> bool:
        # process image
        self.rgb_image = img_msg

        # Convert the image message to
        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough')
       
        cv_image = cv2.resize(cv_image, (640, 480))
        cv_image = cv_image.transpose(2, 1, 0)
        print(cv_image.shape)
        
        
        key_x, key_y, entropy = self.test_validation(self.model, cv_image)
        final_result = self.process_data(key_x, key_y, entropy, cv_image)
        print('table', final_result)



        final_img = cv_image.transpose(2,1,0)
        print(final_img.shape)
        even = final_result[final_result.index % 2 ==0] 
        
        odd = final_result[final_result.index % 2 !=0] 
        

    
        centers = []

        for i in range (len(even)):
        #     print(i)

            x1,y1 = even["pred_coords"].iloc[0][0], even["pred_coords"].iloc[0][1]
            
            x2,y2 = odd["pred_coords"].iloc[0][0], odd["pred_coords"].iloc[0][1]
            print('x1,y1', x1,y1)
            print('x2,y2', x2,y2)


            cv2.circle(final_img, (int(x1), int(y1)), 5, (0, 255, 0), -1)
            cv2.circle(final_img, (int(x2), int(y2)), 5, (0, 255, 0), -1)
            x,y = (x1+x2)/2, (y1+y2)/2
            top_left, top_right, bottom_left, bottom_right = self.calculate_box_corners(x, y, 50)
            print('top_left', top_left)
            cv2.rectangle(final_img, (int(top_left[0]),int(top_left[1])), (int(bottom_right[0]),int(bottom_right[1])), color = (0,255,0), thickness = 2)


            



        # #cv_image = cv2.UMat(cv_image)
        # for keypoint in keypoints:
        #     x, y = int(keypoint[0]), int(keypoint[1])
        #     #cv2.circle(cv_image, (x, y), 5, circle_color, circle_thickness)
        #     #plt.scatter(x,y,c = 'red', marker = 'o', s = 50)
       

        image_with_points_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
 

        vis_img_msg = self.bridge.cv2_to_imgmsg(image_with_points_rgb)

        print("*"*25)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename= "/home/nandhini/thesis_robot/bag_files/images/Gaussian/"+timestr+".png"
        cv2.imwrite(filename, image_with_points_rgb)

        self.image_pub.publish(vis_img_msg)

        rospy.sleep(1)

        return top_left, top_right, bottom_left, bottom_right


    def perceive(self, img_msg, pc_msg) -> bool:
        # unsubscribe from rgb and depth
        self.image_sub.unregister()
        self.pc_sub.unregister()

        # process image
        self.rgb_image = img_msg
        self.pc = pc_msg

        # Convert the image message to
        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough')
        
       

        vis_img_msg = self.bridge.cv2_to_imgmsg(cv_image)

        self.image_pub.publish(vis_img_msg)

    
    # def test_validation (self, model, image):
          
    #     with torch.no_grad():
    #         input_tensor = torch.from_numpy(image).unsqueeze(0).float() 
    #         input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    #         keypoints, variance = model(input_tensor)
        
    #         even_indices = keypoints[:, 1::2]  
    #         odd_indices= keypoints[:, ::2] 
      

    #         coordinates = torch.stack((odd_indices, even_indices), dim=2)
           
           
    #         coordinates = self.scaler.inverse_transform(coordinates[0])
    #         print('IT WORKED')

        return coordinates

    def test_validation(self, model, image):
    
        pred_x = []
        pred_y = [] 
        ent = []


        with torch.no_grad():
                
            input_tensor = torch.from_numpy(image).unsqueeze(0).float() 
            input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs, variance = model(input_tensor)
            #outputs, alpha, beta = model(input_tensor)
            # outputs,loga,logb,logl = model(input_tensor)
            # alpha = torch.exp(loga)
            # beta = torch.exp(logb)
            # lamda = torch.exp(logl)

            # sigma_square = beta / (alpha +1 + (1/2))
            # variance = sigma_square / lamda
            
            
            

            out_x = [[outputs[j][i].item() for i in range(len(outputs[0])) if i%2 == 0] for j in range(len(outputs))]
            out_y = [[outputs[j][i].item() for i in range(len(outputs[0])) if i%2 != 0] for j in range(len(outputs))]
    
            pred_x.extend(out_x)
            pred_y.extend(out_y)
                

            #std = np.sqrt(variance)
        
            print('outputs', outputs)
            #print('std', std)
            print(outputs.shape)       
            
            for j in range(len(outputs)):
                    
                o = outputs[j]
                #s = std[j]
                # s = alpha[j]
                # b = beta[j]
                s = variance[j]
               
                entropy = []
                
                for k in range(int(len(o)/4)):
                    coordinates = o[int(len(o)/4)*k : int(len(o)/4)*(k+1) ]
                    print('c',coordinates)
                    std_dev = s[int(len(s)/4)*k : int(len(s)/4)*(k+1) ]
                    print('s',std_dev)
                    # beta_dev = b[int(len(b)/4)*k : int(len(b)/4)*(k+1) ]
                    en = scipy.stats.cauchy(loc=coordinates, scale=std_dev).entropy()
                   
                    #en = scipy.stats.gennorm(loc=coordinates, scale=std_dev, beta = beta_dev).entropy()

                    print('entropy',en)
                    entropy.append(np.average(en))
                ent.append(entropy)

        
        return pred_x,pred_y,ent
    
    def process_data(self,pred_x, pred_y, ent, img):

        modified_entropy = []

        for k in range(len(ent)):

            for value in ent[k]:
                 modified_entropy.extend([value, value])

        final = pd.DataFrame(columns=['predicted_value_x', 'predicted_value_y', 'entropy'])




        for i in range(len(img)):
            print('predicted' , pred_x)
            for j in range(len(pred_x[0])):
                print("i:", i)
                print("j:", j)
                print("len(pred_x[0]):", len(pred_x[0]))
                print("len(modified_entropy):", len(modified_entropy))          
                
    
                dictionary = pd.Series(data={

                'entropy': modified_entropy[ j], 
                'predicted_value_x': pred_x[0][j],
                'predicted_value_y': pred_y[0][j]

            })
               # print(df.head())
                final = final.append(dictionary, ignore_index=True)
                print('final', final)


        final.to_csv('/home/nandhini/thesis/Grasp/cauchy/temp_robot.csv')
        final['pred_coords'] = final.apply(lambda row: [row['predicted_value_x'], row['predicted_value_y']], axis=1)
        pred_coordinates = final['pred_coords'].tolist()

        scaler = StandardScaler()
        sample_data = [[640, 480], [1, 1]]
        scaler.fit(sample_data)
        inverse_transformed_pred = scaler.inverse_transform(pred_coordinates)
        final['pred_coords'] = inverse_transformed_pred.tolist()


        final1 = final.sort_values('entropy').reset_index(drop=True)

        final1.to_csv('/home/nandhini/thesis/Grasp/cauchy/thesis_robot.csv')

        return final1
    
    def calculate_box_corners(self, center_x, center_y, side_length):

        half_length = side_length / 2

        top_left = (center_x - half_length, center_y - half_length)
        top_right = (center_x + half_length, center_y - half_length)
        bottom_left = (center_x - half_length, center_y + half_length)
        bottom_right = (center_x + half_length, center_y + half_length)

        return top_left, top_right, bottom_left, bottom_right


def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("visualizer")

    visualizer = Visualizer()

    rospy.loginfo('visualizer is initialized')

    rospy.spin()


if __name__ == '__main__':
    main()




