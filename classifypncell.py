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

from __future__ import print_function, unicode_literals, absolute_import, division


##==== inside a folder containing Dockerfile, run: sudo docker build -t cytomine/s_python_classifypncell ====##

import sys
import numpy as np
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob
from tifffile import imread
#from csbdeep.utils import Path, normalize
#from stardist import random_label_cmap
#from stardist.models import StarDist2D
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, Ontology, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, JobData, Project, ImageInstance, Property, OntologyCollection, Term, RelationTerm, TermCollection
#from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection
#from cytomine.models.property import Tag, TagCollection, PropertyCollection
#from cytomine.utilities.software import parse_domain_list, setup_classify, stringify

from argparse import ArgumentParser
from PIL import Image
import argparse
import json
import logging
import logging.handlers
import shutil
import time

import cv2
import math

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__copyright__ = "PN Classification: MFA Fauzi, et al. 2015 (https://doi.org/10.1007/978-3-319-19156-0_17)"
__version__ = "0.1.0"



def run(cyto_job, parameters):
    logging.info("----- PN-Classificationh v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    threshold_set=parameters.cytomine_th_set
    roi_type=parameters.cytomine_roi_type
    write_hv=parameters.cytomine_write_hv

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()
    
    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "PN_classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        #Go over images
        for id_image in list_imgs2:    
            print('Current image:', id_image)
            roi_annotations = AnnotationCollection()
            roi_annotations.project = id_project
            roi_annotations.term = parameters.cytomine_id_cell_term
            roi_annotations.image = id_image #conn.parameters.cytomine_id_image
            roi_annotations.job = parameters.cytomine_id_annotation_job
            roi_annotations.user = parameters.cytomine_id_user_job
            roi_annotations.showWKT = True
            roi_annotations.fetch()
            # print(roi_annotations)

            hue_all=[]
            val_all=[]
            class_positive = 0
            class_negative = 0

            #Go over ROI in this image
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            roi_numel=len(roi_annotations)
            x=range(1,roi_numel)
            increment=np.multiply(10000,x)            
            job.update(status=Job.RUNNING, progress=50, statusComment="Running PN classification on nuclei...")
            for i, roi in enumerate(roi_annotations):
                
                for inc in increment:
                    if i==inc:
                        shutil.rmtree(roi_path, ignore_errors=True)
                        import gc
                        gc.collect()
                        print("i==", inc)
                
                #Get Cytomine ROI coordinates for remapping to whole-slide
                #Cytomine cartesian coordinate system, (0,0) is bottom left corner                
                print("----------------------------Cells------------------------------")
                roi_geometry = wkt.loads(roi.location)
                minx=roi_geometry.bounds[0]
                miny=roi_geometry.bounds[3]
                #Dump ROI image into local PNG file
                # roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')
                
                
                ## --- ROI types: crop or alpha ---
                if roi_type==1: #alpha
                    roi.dump(dest_pattern=roi_png_filename,mask=True)
                elif roi_type==2: #crop
                    roi.dump(dest_pattern=roi_png_filename)
                    
               
                J = cv2.imread(roi_png_filename,cv2.IMREAD_UNCHANGED)
                J = cv2.cvtColor(J, cv2.COLOR_BGRA2RGBA) 
                [r, c, h]=J.shape
                # print("J: ",J)
                

                if r < c:
                    blocksize=r
                else:
                    blocksize=c
                # print("blocksize:",blocksize)
                rr=np.zeros((blocksize,blocksize))
                cc=np.zeros((blocksize,blocksize))

                zz=[*range(1,blocksize+1)]
                # print("zz:", zz)
                for i in zz:
                     rr[i-1,:]=zz
                # print("rr shape:",rr.shape)

                zz=[*range(1,blocksize+1)]
                for i in zz:
                    cc[:,i-1]=zz
                # print("cc shape:",cc.shape)
 
 
                cc1=np.asarray(cc)-16.5
                rr1=np.asarray(rr)-16.5
                cc2=np.asarray(cc1)**2
                rr2=np.asarray(rr1)**2
                rrcc=np.asarray(cc2)+np.asarray(rr2)

                weight=np.sqrt(rrcc)
                weight2=1./weight
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

                currentblock = Jhsv[0:blocksize,0:blocksize,:]
                currentblockH=currentblock[:,:,0]
                currentblockV=1-currentblock[:,:,2]
                hue=sum(sum(currentblockH*weight2))
                val=sum(sum(currentblockV*weight2))
#                 print("hue:", hue)
#                 print("val:", val)
                hue_all.append(hue)
                val_all.append(val)

                if threshold_set==1:
                    #--- Threshold values (modified-used on v0.1.25 and earlier)---
                    # FP case (positive as negative): Hue 4.886034191808089 Val 14.45894207427296 
                    if hue<5:#mod1<2; mod2<5
                       cellclass=2
                    elif val<15:
                       cellclass=1
                    else:
                        if hue<30 or val>40:
                           cellclass=2
                        else:
                           cellclass=1
                    #--------------------------------------------------------------
                elif threshold_set==2:
                    #--- Threshold values (original-used on v2 from 18 April 2022)---
                    if val>50:
                       cellclass=2
                    else:
                        if hue>70:
                            cellclass=1
                        else:
                            cellclass=2
                    #----------------------------------------------------------------


                if cellclass==1:#negative
                    id_terms=parameters.cytomine_id_negative_term
                    class_negative=class_negative+1
                elif cellclass==2:#positive
                    id_terms=parameters.cytomine_id_positive_term  
                    class_positive=class_positive+1

                cytomine_annotations = AnnotationCollection()
                annotation=roi_geometry
                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#parameters.cytomine_id_image,
                                                       id_project=parameters.cytomine_id_project,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)

                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()
                
            end_classify_time=time.time()
            
            if write_hv == 1:
                cytomine_annotations = AnnotationCollection()    
                cytomine_annotations.project = project.id
                cytomine_annotations.image = id_image
                cytomine_annotations.job = job.id
                cytomine_annotations.user = user
                cytomine_annotations.showAlgo = True
                cytomine_annotations.showWKT = True
                cytomine_annotations.showMeta = True
                cytomine_annotations.showGIS = True
                cytomine_annotations.showTerm = True
                cytomine_annotations.annotation = True
                cytomine_annotations.fetch()
                hue_all.reverse()
                val_all.reverse()

                ## --------- WRITE Hue and Value values into annotation Property -----------
                job.update(status=Job.RUNNING, progress=80, statusComment="Writing classification results on CSV...")
                for i, annotation in enumerate(cytomine_annotations):
                    f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(annotation.id,annotation.image,annotation.project,job.id,annotation.term,annotation.user,annotation.area,annotation.perimeter,str(hue_all[i]),str(val_all[i]),annotation.location))
                    Property(Annotation().fetch(annotation.id), key='Hue', value=str(hue_all[i])).save()
                    Property(Annotation().fetch(annotation.id), key='Val', value=str(val_all[i])).save()
                    Property(Annotation().fetch(annotation.id), key='ID', value=str(annotation.id)).save()
                ##---------------------------------------------------------------------------
            
            f.write("\n")
            f.write("Image ID;Class Positive;Class Negative;Total Nuclei;Execution Time \n")
            f.write("{};{};{};{};{}\n".format(id_image,class_positive,class_negative,class_positive+class_negative,end_classify_time-start_time))
            

        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "PN_classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)
        
