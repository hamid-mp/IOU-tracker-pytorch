# -------------------------- Note ------------------
# modfiy the Max_ID




import torch



class IOU_tracker_Pytorch:

    def __init__(self, sigma_l:float, sigma_iou:float):
        

        self.sigma_l = sigma_l
        self.sigma_iou = sigma_iou
        self.max_ID = None
        self.first_frame = True
        self.previous_det = []

    def _io(self, bbox1:torch.tensor, bbox2:torch.tensor):
        '''
        (x1, y1, x2, y2) = bbox ====> (x1,y1) => bottom left     (x2,y2)=> top right
        calculate intersection over union (IOU) between two bounding box
        '''
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        xx1 = torch.max(x0_1, x0_2)
        yy1 = torch.max(y0_1, y0_2)

        xx2 = torch.min(x1_1, x1_2)
        yy2 = torch.min(y1_1, y1_2)

        w = xx1 - xx2
        h = yy2 - yy1
        
        w = torch.clamp(w, min=0)
        h = torch.clamp(h, min=0)

        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        
        intersection = w * h

        unions = area1 + area2 - intersection

        return intersection / unions




    def _iou1(self, bbox1:torch.tensor, bbox2:torch.tensor):
        '''
        (x1, y1, x2, y2) = bbox ====> (x1,y1) => top left     (x2,y2)=> bottom right

        calculate intersection over union (IOU) between two bounding box
        '''
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2


        xx1 = torch.max(x0_1, x0_2)
        yy1 = torch.max(y0_1, y0_2)
        xx2 = torch.min(x1_1, x1_2)
        yy2 = torch.min(y1_1, y1_2)


        w = xx2 - xx1
        h = yy2 - yy1
        if w > 0 and h > 0:
            intersection = w * h
        else:
            intersection  = 0 

        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)

        unions = area1 + area2 - intersection
        return intersection / unions


    def _filter_low_conf(self, detection:torch.tensor): 
        '''
        remove the detections with low confidence scores => chance to be a False Positive

        input:
            detection (torch.tensor) : output of yolov5Face 
        output:
            detections (list) : a list which contains detections that has conf scores higher than a threshold (sima_l)
        '''
        detections = [det for det in detection if det[4] > self.sigma_l]

        return detections


    def _remain_id_dets(self, remain_dets:list): 
        '''
        assign new ID to new detections  which is not updated into previous detections
        
        remain_dets (list) : this is a list containing the detections which is not updated in current frame detections
        max_ID : is the maximum ID available from previous frame
        '''


        remain_tracks = []
        if len(remain_dets) > 0 :
            for det in remain_dets:
                
                self.max_ID += 1

                remain_tracks.append([self.max_ID, det])

        return remain_tracks , self.max_ID


    def _update_previous_dets(self, detections:list): #-------------------------------- 2
        '''
        Here we first check if this is the first detection or not 
        if it is the first, all detections will get an ID and append to the final_list

        if it is not the first frame, so we compare each new detection of current frame with previous ones
        and if the iou between them was greater than iou_threshold, we assign previous ID to new detection


        input:
            detections (list) : list of current frame detections

        outputs:
            updated_list (list) : a list that is updated with the new dets and previous IDs
            remain_dets (list)  : a list which contains the current dets that are not updated and should get new IDs
        '''
        updated_list = []
        remain_dets = detections.copy() #copy new detections and we remove its elements if they replace with previous

        for i, new_det in enumerate(detections):
            all_iou = []
            id_list = []    
            for old_det in self.previous_det:
                all_iou.append(self._iou1(new_det[:4], old_det[1][:4]))
                id_list.append(old_det[0]) 
            max_ = max(all_iou)
            max_index = all_iou.index(max_)

            if max_ >= self.sigma_iou:
                updated_list.append([ id_list[max_index] , new_det ]) 

                remain_dets = [i for i in remain_dets if  any(i!= new_det)]


        return updated_list, remain_dets


    def _process(self, detections:torch.tensor):
        '''
        Here we put all methods together

        input:
            detections (torch.tensor) : detections of yolov5face
        
        output:
            final_dets (list) : list of tracks with IDs
        '''
        
        filterd_dets = self._filter_low_conf(detections)

        if  self.first_frame:   #this is for first frame and each detection gets an ID
            
            self.first_frame = False

            for id , det in enumerate(filterd_dets, start=1):
                self.previous_det.append([id, det])
                final_tracks = self.previous_det
            
            self.max_ID = id

        else:
      

            updated, remain = self._update_previous_dets(filterd_dets)

            final_remains, max_ = self._remain_id_dets(remain)


            final_tracks = updated + final_remains
            self.previous_det = final_tracks
            self.max_ID = max_
        return final_tracks


'''
tracker = IOU_tracker_Pytorch(sigma_iou=0.5, sigma_l=0)

P = torch.tensor([
[1, 1, 3, 3, 0.95],
[1, 1, 3, 4, 0.93],
[1, 0.9, 3.6, 3, 0.98],
[1, 0.9, 3.5, 3, 0.97]
])
P1 = torch.tensor([
[1.1, 0.9, 2.8, 2.5, 0.85],
[1.4, 1.1, 3.2, 3.5, 0.73],
[1.5, 0.9, 3.2, 3, 0.88],
[1, 0.7, 3, 3.5, 0.92],
[5,6,34,43, 0.95]])

P2 = torch.tensor([
[1.1, 0.9, 2.8, 2.5, 0.85],
[1.4, 1.1, 3.2, 3.5, 0.73],
[1.5, 0.9, 3.2, 3, 0.88],
[1, 0.7, 3, 3.5, 0.92],
[5,6,34,43, 0.95],
[1234,1242342,42355,345345,0.88]])
dets = [P, P1, P2]
for det in dets:
    tracker._process(det)
'''