# FinalProject-Group4

### Content
Fruits were planted in the shaft of a low speed motor (3 rpm) and a short movie of 20 seconds was recorded.

A Logitech C920 camera was used for filming the fruits. This is one of the best webcams.

Behind the fruits we placed a white sheet of paper as background.

However due to the variations in the lighting conditions, the background was not uniform and we wrote a dedicated algorithm which extract the fruit from the background. This algorithm is of flood fill type: we start from each edge of the image and we mark all pixels there, then we mark all pixels found in the neighborhood of the already marked pixels for which the distance between colors is less than a prescribed value. We repeat the previous step until no more pixels can be marked.

All marked pixels are considered as being background (which is then filled with white) and the rest of pixels are considered as belonging to the object.

The maximum value for the distance between 2 neighbor pixels is a parameter of the algorithm and is set (by trial and error) for each movie.

### Dataset properties
Training set: 15506 images.

Testing set: 5195 images.

Number of classes: 33 fruits.

Image size: 100x100 pixels.

Filename format: image_index_100.jpg (e.g. 32_100.jpg) or r_image_index_100.jpg (e.g. r_32_100.jpg). "r" stands for rotated fruit. "100" represents image size (100x100 pixels).

Different varieties of the same fruit are shown having different labels (e.g. Apple Red 1, Apple Red 2).
