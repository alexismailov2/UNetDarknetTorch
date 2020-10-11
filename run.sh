#!/usr/bin/env bash
echo "Generate needed configuration of unet for example: 3 input colors, 2 classes, 3 levels, 8 features"
./build_host/train_unet_darknet2d --generate-custom-unet=3,2,3,8
echo "Generate needed configuration of unet for example: 3 input colors, 2 classes, 2 levels, 8 features"
./build_host/train_unet_darknet2d --generate-custom-unet=3,2,2,8
echo "Generate classic configuration of unet for example: 1 input colors, 2 classes, 4 levels, 64 features"
./build_host/train_unet_darknet2d --generate-custom-unet=1,2,4,64

echo "Create test random train dataset with circles and rectangles"
./build_host/train_unet_darknet2d --test-dataset-create=./dataset/train/imgs/,./dataset/train/masks/,256,256,30,0,255,0,255,0,0
echo "Create test random valid dataset with circles and rectangles"
./build_host/train_unet_darknet2d --test-dataset-create=./dataset/valid/imgs/,./dataset/valid/masks/,256,256,30,0,255,0,255,0,0

xdg-open ./dataset/train/imgs
xdg-open ./dataset/train/masks
xdg-open ./dataset/valid/imgs
xdg-open ./dataset/valid/masks

echo "Convert exist dataset to needed size and crop"
./build_host/train_unet_darknet2d --convert-images="64,64,128,128,./dataset/train/imgs/,./dataset/train/converted_imgs/"
./build_host/train_unet_darknet2d --convert-images="64,64,128,128,./dataset/train/masks/,./dataset/train/converted_masks/"
xdg-open ./dataset/train/converted_imgs
xdg-open ./dataset/train/converted_masks

echo "Start training here we can also downscale image during the train"
./build_host/train_unet_darknet2d \
 --model-darknet=./model/_unet3c2cl2l8f.cfg \
 --epochs=200 \
 --checkpoints-output=./checkpoints_128x128_3c2cl2l8f \
 --train-directories=./dataset/train/imgs,./dataset/train/masks \
 --valid-directories=./dataset/valid/imgs,./dataset/valid/masks \
 --colors-to-class-map="rectangle,0,255,0,disk,255,0,0" \
 --selected-classes-and-thresholds="rectangle,0.3,disk,0.3" \
 --batch-count=3 \
 --size-downscaled=128,128 \
 --grayscale=no \

echo "Start training here we can also downscale image during the train"
./build_host/train_unet_darknet2d \
 --model-darknet=./model/_unet3c2cl3l8f.cfg \
 --epochs=200 \
 --checkpoints-output=./checkpoints_128x128_3c2cl3l8f \
 --train-directories=./dataset/train/imgs,./dataset/train/masks \
 --valid-directories=./dataset/valid/imgs,./dataset/valid/masks \
 --colors-to-class-map="rectangle,0,255,0,disk,255,0,0" \
 --selected-classes-and-thresholds="rectangle,0.3,disk,0.3" \
 --batch-count=3 \
 --size-downscaled=128,128 \
 --grayscale=no \
