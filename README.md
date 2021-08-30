# raspberrypi-image-classifier-tf-lite

## To classify image
``` 
  python3 classify_image.py -m model.tflite -l labels.txt -i smile.png
```

## Pi camera
```
  python3 classify_picamera.py -model model.tflite -labels labels.txt
```
