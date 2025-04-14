# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import random
import skimage
import skimage.transform as trans
import skimage.io as io

'''
def random_crop(image, mask, max_border = 20):
  """
  Randomly crops the image.
  - Randomly select number of pixels to remove from each edge
  - Crop
  - Resize back to original size
  """
  target_size = (image.shape[0], image.shape[1]) # Assumes channels last, assumes mask is the same size

  remove_v = (int(random.random()*max_border), int(random.random()*max_border))
  remove_h = (int(random.random()*max_border), int(random.random()*max_border))

  remove_img = (remove_v, remove_h, (0,0))
  remove_mask = (remove_v, remove_h)
   
  image = trans.resize(skimage.util.crop(image, remove_img), target_size)
  mask = trans.resize(skimage.util.crop(mask, remove_mask), target_size)

  return image, mask
'''

# Image axes are (vert, horiz)
def random_flip_horiz(image, mask):
  if (random.random() < 0.5):
    return image[:, ::-1, :], mask[:, ::-1] # Use numpy operation to reverse the values along the second axis
  else:
    return image, mask


def random_flip_vert(image, mask):
  if (random.random() < 0.5):
    return image[::-1, :, :], mask[::-1, :] # Use numpy operation to reverse the values along the first axis
  else:
    return image, mask


def random_rotate(image, mask):
  angle = int(random.random() * 4) * 90
  return trans.rotate(image, angle, preserve_range=True), trans.rotate(mask, angle, preserve_range=True)

'''
def random_scale(image, factor, offset):
  """
  Randomly adjusts the mean and sd of the image
  TODO
  """
  # Compute rand_factor and rand_offset and add to 1
  # update image
  scale_factor = 1 + (random.random() * 2 * factor) - factor # 1 +/- scale_factor
  random_offset = (random.random() * 2 * offset) - offset # +/- offset
  #print("AUGMENTATION: FACTOR={} OFFSET={}".format(scale_factor, random_offset))
  #print("BEFORE:")
  #print(image)
  new_image = image * scale_factor + random_offset
  # print("AFTER:")
  # print(new_image)
  return new_image
'''  

def distort_image(image, mask, max_to_remove=20):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
  """
  # image, mask = random_crop(image, mask, max_to_remove) # MAY be mucking up labels: try without...
  # image, mask = random_crop(image, mask, 50) # Try more aggressive cropping...
  image, mask = random_flip_horiz(image, mask)
  image, mask = random_flip_vert(image, mask)
  # image, mask = random_rotate(image, mask) # PROBLEM: DIVIDING THE VALUES BY 255! NEED FIX THE DATATYPE

  """
  # Don't do - bad idea...
  # Randomly move mean and SD (brightness/contrast augmentation)
  # FACTOR = 0.5
  # OFFSET = 0.2

  # Conservative settings
  # FACTOR = 0.1
  # OFFSET = 0.1
  
  # Aggressive settings
  FACTOR = 0.5
  OFFSET = 0.5

  image = random_scale(image, FACTOR, OFFSET)
  """
  
  # Randomly distort the colors.
  # TODO
  # distorted_image = distort_color(distorted_image)
         
  return image, mask

def test():
  filename = 'D:/landslides/train/visual/Blue_Duck_wv_18SEP_pansh_nztm_rect_resample_12-14_000_000.tif'
  img = io.imread(filename)
  mask = io.imread(filename,as_gray = True)
  io.imsave('c:/brent/temp/original_image.png', img)
  io.imsave('c:/brent/temp/original_mask.png', mask)
  for i in range(10):
    distorted_img, distorted_mask = distort_image(img, mask)
    io.imsave('c:/brent/temp/distorted_{}_img.png'.format(i), distorted_img)
    io.imsave('c:/brent/temp/distorted_{}_msk.png'.format(i), distorted_mask)
  
if __name__ == '__main__':
    # test()
    print("Augmentation loaded - nothing to see here...")