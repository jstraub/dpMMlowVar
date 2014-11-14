path = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'

cand = ['bathroom_0004_47_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bookstore_0002_100_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0004_149_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0003_174_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0004_178_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0007_191_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'office_0000_215_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'furniture_store_0001_227_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'furniture_store_0002_241_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'furniture_store_0002_246_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'furniture_store_0002_248_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'classroom_0023_326_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0002_362_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0006_375_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0008_380_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0008_383_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0010_390_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'home_office_0012_395_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'laundry_room_0001_405_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'playroom_0005_435_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0134_521_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0037_545_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0038_550_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bathroom_0012_657_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bathroom_0028_691_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bathroom_0041_720_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'kitchen_0019_747_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0040_954_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0058_1001_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0095_1108_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0095_1109_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'bedroom_0103_1131_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0025_1201_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0040_1246_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0041_1250_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0048_1275_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0052_1290_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'living_room_0072_1329_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0008_1361_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0010_1366_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0022_1401_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0023_1402_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0036_1447_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png',
'dining_room_0036_1449_K_1-base_DPvMFmeans-T_100-delta_12.0-nu_10003.0-lambda_-1.17364817767lbls.png']

candName = ['bathroom_0004_47',
'bookstore_0002_100',
'living_room_0004_149',
'bedroom_0003_174',
'bedroom_0004_178',
'bedroom_0007_191',
'office_0000_215',
'furniture_store_0001_227',
'furniture_store_0002_241',
'furniture_store_0002_246',
'furniture_store_0002_248',
'classroom_0023_326',
'home_office_0002_362',
'home_office_0006_375',
'home_office_0008_380',
'home_office_0008_383',
'home_office_0010_390',
'home_office_0012_395',
'laundry_room_0001_405',
'playroom_0005_435',
'bedroom_0134_521',
'dining_room_0037_545',
'dining_room_0038_550',
'bathroom_0012_657',
'bathroom_0028_691',
'bathroom_0041_720',
'kitchen_0019_747',
'bedroom_0040_954',
'bedroom_0058_1001',
'bedroom_0095_1108',
'bedroom_0095_1109',
'bedroom_0103_1131',
'living_room_0025_1201',
'living_room_0040_1246',
'living_room_0041_1250',
'living_room_0048_1275',
'living_room_0052_1290',
'living_room_0072_1329',
'dining_room_0008_1361',
'dining_room_0010_1366',
'dining_room_0022_1401',
'dining_room_0023_1402',
'dining_room_0036_1447',
'dining_room_0036_1449']

import cv2, re ,os
import numpy as np
import fnmatch                                                  

for i,ca in enumerate(cand):
  print candName[i]
  for file in os.listdir(path):
    if fnmatch.fnmatch(file,candName[i]+'_K_4*.png'):
      Ispkm4 = cv2.imread(path+file)
    if fnmatch.fnmatch(file,candName[i]+'_K_6*.png'):
      Ispkm6 = cv2.imread(path+file)
  for file in os.listdir('/data/vision/fisher/data1/nyu_depth_v2/extracted/'):
    if fnmatch.fnmatch(file,candName[i]+'_rgb.png'):
      Irgb = cv2.imread('/data/vision/fisher/data1/nyu_depth_v2/extracted/'+file)

  I = cv2.imread(path+ca)
  cv2.imshow('candidate',I)
  cv2.imshow('candidate rgb',Irgb)
  cv2.imshow('candidate spkm4',Ispkm4)
  cv2.imshow('candidate spkm6',Ispkm6)
  cv2.waitKey(0)
