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

def align_face(event,context):
    try:
        content_type_header = event['headers']['content-type']
        #print(event['body'])
        body = base64.b64decode(event["body"])
        print("Body loaded...")

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        #prediction = get_prediction(image_bytes=picture.content)
        print(type(picture.content))
        jpg_as_np=np.frombuffer(picture.content, dtype=np.uint8)
         
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        #im=cv2.imread(image_bytes=picture.content)
        im=cv2.imdecode(jpg_as_np, flags=1)
        faceRects=detector(im,0)
        print("Number of faces in the image = ",len(faceRects))
        
        points = fbc.getLandmarks(detector, predictor, im)
        points=np.array(points)
        im = np.float32(im)/255.0
        
        h=300
        w=300
        imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), im, points)
        imNorm = np.uint8(imNorm*255)
 
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename)<4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {"statusCode":200, "headers": {"Content-Type":"application/json","Access-Control-Allow-Origin":"*","Access-Control-Allow-Credentials":True}, "body":json.dumps({"file":filename.replace('"',''),"predicted":(imNorm.tolist())})
        }
    except Exception as e:
        print(repr(e))
        return {"statusCode":500, "headers": {"Content-Type":"application/json","Access-Control-Allow-Origin":"*","Access-Control-Allow-Credentials":True}, "body":json.dumps({"error":repr(e)})
        }


