# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.


__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__copyright__ = "MFA Fauzi, et al. 2015 (https://doi.org/10.1007/978-3-319-19156-0_17)"
__version__ = "1.0.0"

##==== inside a folder containing Dockerfile, run: sudo docker build -t cytomine/s_python_classifypncell ====##

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import os
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
#from csbdeep.utils import Path, normalize
#from stardist import random_label_cmap
#from stardist.models import StarDist2D
from cytomine import cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job
from cytomine.models.ontology import Ontology, Term, RelationTerm, TermCollection

from PIL import Image
import argparse
import json
import logging

import cv2
import math


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
        conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")
        base_path = "{}".format(os.getenv("HOME")) # Mandatory for Singularity
        working_path = os.path.join(base_path,str(conn.job.id))

        ontology = Ontology("classPNcells"+str(conn.parameters.cytomine_id_project)).save()
        term_P = Term("PositiveCell", ontology.id, "#FF0000").save()
        term_N = Term("NegativeCell", ontology.id, "#00FF00").save()

        # Get all the terms of our ontology
        terms = TermCollection().fetch_with_filter("ontology", ontology.id)
        print(terms)
        
        # #Loading pre-trained Stardist model
        # np.random.seed(17)
        # lbl_cmap = random_label_cmap()
        # #Stardist H&E model downloaded from https://github.com/mpicbg-csbd/stardist/issues/46
        # #Stardist H&E model downloaded from https://drive.switch.ch/index.php/s/LTYaIud7w6lCyuI
        # model = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')   #use local model file in ~/models/2D_versatile_HE/

        #Select images to process
        images = ImageInstanceCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)
        list_imgs = []
        if conn.parameters.cytomine_id_images == 'all':
            for image in images:
                list_imgs.append(int(image.id))
            else:
                list_imgs = [int(id_img) for id_img in conn.parameters.cytomine_id_images.split(',')]

        #Go over images
        for id_image in conn.monitor(list_imgs, prefix="Running PN classification on image", period=0.1):

            # #Dump ROI annotations in img from Cytomine server to local images
            # #conn.job.update(status=Job.RUNNING, progress=0, statusComment="Fetching ROI annotations...")
            roi_annotations = AnnotationCollection()
            roi_annotations.project = conn.parameters.cytomine_id_project
            roi_annotations.term = conn.parameters.cytomine_id_roi_term
            roi_annotations.image = id_image #conn.parameters.cytomine_id_image
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            print(roi_annotations)

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            for roi in roi_annotations:
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                print("----------------------------Cells------------------------------")
                roi_geometry = wkt.loads(roi.location)
                print("ROI Geometry from Shapely: {}".format(roi_geometry))
                print("ROI Bounds")
                print(roi_geometry.bounds)
                minx=roi_geometry.bounds[0]
                miny=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_png_filename=os.path.join(roi_path+'/'+str(roi.id)+'.png')
                print("roi_png_filename: %s" %roi_png_filename)
                roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=True)
                #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)

                # im=Image.open(roi_png_filename)

                J = cv2.imread(roi_png_filename,cv2.IMREAD_UNCHANGED)
                J = cv2.cvtColor(J, cv2.COLOR_BGRA2RGBA) 
                [r, c, h]=J.shape

                if r < c:
                    blocksize=r
                else:
                    blocksize=c
                print("blocksize:",blocksize)
                rr=np.zeros((blocksize,blocksize))
                cc=np.zeros((blocksize,blocksize))

                zz=[*range(1,blocksize+1)]
                print("zz:", zz)
                for i in zz:
                     rr[i-1,:]=zz
                print("rr shape:",rr.shape)

                zz=[*range(1,blocksize+1)]
                for i in zz:
                    cc[:,i-1]=zz
                print("cc shape:",cc.shape)


                cc1=np.asarray(cc)-16.5
                rr1=np.asarray(rr)-16.5
                cc2=np.asarray(cc1)**2
                rr2=np.asarray(rr1)**2
                rrcc=np.asarray(cc2)+np.asarray(rr2)

                weight=np.sqrt(rrcc)
                weight2=1./weight
                print("weight2 shape:",weight2.shape)
                coord=[c/2,r/2]
                halfblocksize=blocksize/2

                y=round(coord[1])
                x=round(coord[0])

                # Convert the RGB image to HSV
                Jalpha=J[:,:,3]
                Jalphaloc=Jalpha/255
                Jrgb = cv2.cvtColor(J, cv2.COLOR_RGBA2RGB)
                Jhsv = cv2.cvtColor(Jrgb, cv2.COLOR_RGB2HSV_FULL)
                Jhsv = Jhsv/255
                Jhsv[:,:,0]=Jhsv[:,:,0]*Jalphaloc
                Jhsv[:,:,1]=Jhsv[:,:,1]*Jalphaloc
                Jhsv[:,:,2]=Jhsv[:,:,2]*Jalphaloc

                print("Jhsv size:",Jhsv.shape)
                print("Jhsv class:",Jhsv.dtype)

                currentblock = Jhsv[0:blocksize,0:blocksize,:]
                print(currentblock.dtype)
                currentblockH=currentblock[:,:,0]
                currentblockV=1-currentblock[:,:,2]
                hue=sum(sum(currentblockH*weight2))
                val=sum(sum(currentblockV*weight2))
                print("hue:", hue)
                print("val:", val)


                if val<15:
                   cellclass=2
                else:
                    if hue<30 or val>40:
                       cellclass=1
                    else:
                       cellclass=2


                if cellclass==1:
                    print("Positive (H: ", str(hue), ", V: ", str(val))
                    id_terms=term_P
                elif cellclass==2:
                    print("Negative (H: ", str(hue), ", V: ", str(val))
                    id_terms=term_N

                        # First we create the required resources
        

                cytomine_annotations = AnnotationCollection()
                cytomine_annotations.append(Annotation(location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=conn.parameters.cytomine_id_project,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)

                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()

                conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

                if __name__ == "__main__":
                    main(sys.argv[1:])

                    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
                        run(cyto_job, cyto_job.parameters)
