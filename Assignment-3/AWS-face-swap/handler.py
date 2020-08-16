print('starting handler python file execution')
try:
    print('importing starts')
    import unzip_requirements
except ImportError:
    print("Error while importing unzip_requirements")
    pass

import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder
import faceBlendCommon as fbc
import dlib
import numpy as np
import cv2
print("Import ends..")

s3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'model-mobilenetv2'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobileNetV2.pt'
#LANDMARK_68_DETECTOR_PATH = os.environ['LANDMARK_68_DETECTOR_PATH'] if 'LANDMARK_68_DETECTOR_PATH' in os.environ else 'shape_predictor_68_face_landmarks.dat'

print('Downloading model...')

s3 = boto3.client('s3')

#try:
#    if os.path.isfile('MODEL_PATH')!=True:
#        obj = s3.get_object(Bucket=s3_BUCKET, Key=LANDMARK_68_DETECTOR_PATH)
#        print("creating Bytestream")
#        landmark_dat = io.BytesIO(obj['Body'].read())
#        print("Loading Model...")
#except Exception as e:
#    print(repr(e))
#    raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def swap_face(event,context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print("Body loaded...")

        picture1 = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print(picture1.content)
        jpg_as_np1=np.frombuffer(picture1.content, dtype=np.uint8)
        
        picture2 = decoder.MultipartDecoder(body, content_type_header).parts[1]
        print(picture2.content)
        jpg_as_np2=np.frombuffer(picture2.content, dtype=np.uint8)
         
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        img2=cv2.imdecode(jpg_as_np2, flags=1)
        faceRects2=detector(img2,0)
        print("Number of faces in the image = ",len(faceRects2))
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        img1=cv2.imdecode(jpg_as_np1, flags=1)
        faceRects1=detector(img1,0)
        print("Number of faces in the image = ",len(faceRects1))
        
        points2 = fbc.getLandmarks(detector, predictor, img2)
        points2=np.array(points2)
        img2 = np.float32(img2)/255.0
        
        points1 = fbc.getLandmarks(detector, predictor, img1)
        points1=np.array(points1)
        img1 = np.float32(img1)/255.0
        
        h=300
        w=300
        img2, points2 = fbc.normalizeImagesAndLandmarks((h,w), img2, points2)
        img2 = np.uint8(img2*255)
        
        img1, points1 = fbc.normalizeImagesAndLandmarks((h,w), img1, points1)
        img1 = np.uint8(img1*255)
        
        im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        im2Display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1Warped = np.copy(img2)
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # Read array of corresponding points
        points1 = fbc.getLandmarks(detector, predictor, img1)
        points2 = fbc.getLandmarks(detector, predictor, img2)
        
        imTemp = im2Display.copy()
        
        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

        # Create convex hull lists
        hull1 = []
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull1.append(points1[hullIndex[i][0]])
            hull2.append(points2[hullIndex[i][0]])
            
        imTemp = im2Display.copy()
        numPoints = len(hull2)
        
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype) 
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        # Find Centroid
        m = cv2.moments(mask[:,:,1])
        center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
        
        sizeImg2 = img2.shape    
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = fbc.calculateDelaunayTriangles(rect, hull2)

        # If no Delaunay Triangles were found, quit
        if len(dt) == 0:
            quit()
            
        imTemp1 = im1Display.copy()
        imTemp2 = im2Display.copy()

        tris1 = []
        tris2 = []
        for i in range(0, len(dt)):
            tri1 = []
            tri2 = []
            for j in range(0, 3):
                tri1.append(hull1[dt[i][j]])
                tri2.append(hull2[dt[i][j]])

            tris1.append(tri1)
            tris2.append(tri2)
        
        
        for i in range(0, len(tris1)):
            fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])
 
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
        print(output)
 
        filename = picture1.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename)<4:
            filename = picture1.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {"statusCode":200, "headers": {"Content-Type":"application/json","Access-Control-Allow-Origin":"*","Access-Control-Allow-Credentials":True}, "body":json.dumps({"file":filename.replace('"',''),"predicted":(output.tolist())})
        }
    except Exception as e:
        print(repr(e))
        return {"statusCode":500, "headers": {"Content-Type":"application/json","Access-Control-Allow-Origin":"*","Access-Control-Allow-Credentials":True}, "body":json.dumps({"error":repr(e)})
        }


