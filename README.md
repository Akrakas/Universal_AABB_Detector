# Universal_AABB_Detector

![alt text](./DEMO/Demo.gif)

This program is an Axis Aligned Bounding Box extractor currently trained to detect faces in pictures.  
The picture is first divided in many small overlapping thumbnails. The resolution of these thumbnails is reduced to 12 by 12 pixels and fed to a convolutionnal neural network whose task is to detect if they are part of a face or not.  
The thumbnails that pass this test are then translated using another neural network toward the center of the face, and then resized with yet another network.  
The resulting thumbnails are then combined with a clustering algorithm to extract the bounding boxes of the face.  

To do :  
Use more steps to eliminate false positives.  

It could technically be used to detect anything given the right database.  
