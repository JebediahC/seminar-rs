# unzip data

SOURCE_DATA_PATH=/src/24305557.zip
DATA_DIR=/workspace/


 # unzip
 unzip $SOURCE_DATA_PATH -d $DATA_DIR

 mkdir dl-model
 mkdir images-and-shpfiles
 mkdir gf-7-building-3bands
 mkdir gf-7-building-4bands

 unzip GF-7\ Building\ \(3Bands\).zip -d gf-7-building-3bands
 unzip GF-7\ Building\ \(4Bands\).zip -d gf-7-building-4bands
 unzip 'DL Model.zip' -d dl-model
 unzip 'Images and Shpfiles.zip' -d images-and-shpfiles