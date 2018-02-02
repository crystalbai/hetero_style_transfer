Skip to content
This repository
Search
Pull requests
Issues
Marketplace
Explore
 @crystalbai
 Sign out
 Watch 101
  Star 1,206  Fork 371 fzliu/style-transfer
 Code  Issues 25  Pull requests 1  Projects 0  Wiki  Insights
Branch: master Find file Copy pathstyle-transfer/scripts/download_models.sh
8457157  on Oct 7, 2015
 Frank Liu Added demo script.
0 contributors
RawBlameHistory     
15 lines (13 sloc)  765 Bytes
#!/bin/bash

if [ "$#" == 0 ] || [ "$1" == "googlenet" ]; then
    curl "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" -o models/googlenet/bvlc_googlenet.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "caffenet" ]; then
    curl "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel" -o models/caffenet/bvlc_reference_caffenet.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "vgg16" ]; then
    curl "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel" -o models/vgg16/VGG_ILSVRC_16_layers.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "vgg19" ]; then
    curl "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel" -o models/vgg19/VGG_ILSVRC_19_layers.caffemodel
fi
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About