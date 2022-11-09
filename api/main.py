from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf
import cv2

app = FastAPI()
MODEL = tf.keras.models.load_model('../savedModels/1')
@app.get("/ping")

async def bing():
    return "Hello, the server is alive"





def display_image(image):
   print(image.shape)
   print(image)
   cv2.imshow("test image", image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
    
def read_files_as_images(iamge_bytes, mask_bytes):
   image_file = cv2.imdecode(np.frombuffer(iamge_bytes, dtype = np.uint8), cv2.IMREAD_ANYCOLOR)
   mask_file = cv2.imdecode(np.frombuffer(mask_bytes, dtype = np.uint8), cv2.IMREAD_GRAYSCALE)
   display_image(mask_file)

   return image_file, mask_file

def Image_mask_input_preprocess(image_file, mask_file):
    x = np.zeros((1, 256, 256, 3), dtype=np.float32)
    y = np.zeros((1, 256, 256, 1), dtype=np.float32)
    image = cv2.resize(image_file, (256, 256))
    mask = cv2.resize(mask_file, (256, 256))
    mask = np.reshape(mask, (256, 256, 1))
    x[0] = image
    y[0] = mask     
    image = image / 255
    mask = mask / 255
    return image, mask

@app.post("/predict")
async def predect(files: list[UploadFile]= File(...)):
    uploaded_image_bytes = await files[0].read()
    uploaded_mask_bytes = await files[1].read()
    image_file, mask_file = read_files_as_images(uploaded_image_bytes,uploaded_mask_bytes) 
    pass

    image_batch = Image_mask_input_preprocess(image_file, mask_file) #preparing image as a batch by adding a diminsion to be provided to the model.predict
    prediction_res = MODEL.predict(image_batch)
    return prediction_res
    pass
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)





    


            


