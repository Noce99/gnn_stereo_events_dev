import os
import cv2
import h5py
import hdf5plugin
import numpy as np
from typing import Dict, Tuple, List, Union
from collections import deque
from sklearn.neighbors import NearestNeighbors


WIDTH = 640
HEIGHT = 480


class Event:
    p                : bool
    x                : np.uint16
    y                : np.uint16
    t                : np.uint32
    left_right       : str
    arr              : np.ndarray
    arr_for_distance : np.ndarray

    def __init__(self,
        p          : bool,
        x          : np.uint16,
        y          : np.uint16,
        t          : np.uint32,
        left_right : str
    ):
        self.p          = p
        self.x          = x
        self.y          = y
        self.t          = t
        self.left_right = left_right

        self.arr =              np.array([self.p, self.t, self.x, self.y]) # [polarity, time, x, y]
        self.arr_for_distance = np.array([self.p, self.t, self.y])         # [polarity, time, y]

    def __str__(self):
        return f"<{self.left_right}>[{self.x}, {self.y}] {{{self.t}}} {self.p}"


class DsecEventIterator:
    left_right : str

    events_h5 : h5py.File
    rectify_map : h5py.File
    i : int
    len : int

    def __init__(self,
        sequence_folder_path : str,
        left_right : str
    ):
        self.left_right = left_right

        event_h5_path =         os.path.join(sequence_folder_path, "events", left_right, "events.h5")
        rectify_map_h5_path =   os.path.join(sequence_folder_path, "events", left_right, "rectify_map.h5")
        self.events_h5 =        h5py.File(event_h5_path)
        self.rectify_map =      h5py.File(rectify_map_h5_path, "r")["rectify_map"][()]
        self.i =                0
        self.len =              self.events_h5[f"events/p"].len()
        self.tmp_len =          self.events_h5[f"events/t"][self.len-1] - self.events_h5[f"events/t"][0]

    def __iter__(self):
        return self

    def __next__(self) -> Event:
        if self.i >= self.len:
            raise StopIteration
        
        to_return = Event(
            self.events_h5[f"events/p"][self.i] == np.uint(1),
            self.events_h5[f"events/x"][self.i],
            self.events_h5[f"events/y"][self.i],
            self.events_h5[f"events/t"][self.i],
            self.left_right,
        )
        self.i += 1
        return to_return
    
    def __len__(self):
        return self.len


class DsecStereoEventIterator:
    left_iter :     DsecEventIterator
    right_iter :    DsecEventIterator

    last_left :     Event
    last_right :    Event

    def __init__(self,
        sequence_folder_path : str,
    ):
        self.left_iter =  iter(DsecEventIterator( sequence_folder_path, left_right="left"  ))
        self.right_iter = iter(DsecEventIterator( sequence_folder_path, left_right="right" ))

        self.last_left =  next( self.left_iter  )
        self.last_right = next( self.right_iter )

    def __iter__(self):
        return self

    def __next__(self) -> Event:
        if self.last_left.t < self.last_right.t:
            to_return = self.last_left
            self.last_left =  next( self.left_iter  )
        else:
            to_return = self.last_right
            self.last_right =  next( self.right_iter  )
        
        return to_return
    
    def __len__(self):
        return len(self.left_iter) + len(self.right_iter)

class EventSet:
    time_window_us : int
    events : deque[np.ndarray] # [polarity, time, x, y]

    def __init__(self, time_window_us: int):
        self.time_window_us = time_window_us
        self.events               = deque()
        self.events_for_distance  = deque()
    
    def add(self, event : Event):
        # Remove events that are too old
        while self.events and event.arr[1] - self.events[0][1] >= self.time_window_us:
            self.events.popleft()
            self.events_for_distance.popleft()

        # Add the new event
        self.events.append(event.arr)
        self.events_for_distance.append(event.arr_for_distance)
    
    def get_np(self):
        return np.array(self.events)
    
    def get_np_for_distance(self):
        return np.array(self.events_for_distance)

    def __len__(self):
        return len(self.events)
    
class EventGraph:
    event : Event
    set : EventSet

    vertices : Union[np.ndarray, None]
    edges : Union[np.ndarray, None]

    def __init__(self, event: Event, set : EventSet, neighbors : int):
        self.set = set
        self.event = event

        points = self.set.get_np_for_distance()

        
        if points.shape[0] > 0:
            nbrs = NearestNeighbors(n_neighbors=min(points.shape[0], neighbors), algorithm='auto', metric='euclidean')
            nbrs.fit(points)
            
            # Find neighbors for the new point
            indices = nbrs.kneighbors(event.arr_for_distance.reshape(1, -1), return_distance=False)
            indices = indices[0]

            vertices = [self.event.arr]
            points = self.set.get_np()
            for i in indices:
                vertices.append(points[i])
            
            vertices = np.array(vertices)

            edges = np.zeros((2, len(indices)), dtype=np.uint8)
            edges[0, :] = np.arange(1, len(indices)+1, dtype=np.uint8)

            self.vertices = vertices
            self.edges = edges
        else:
            self.vertices = None
            self.edges = None




def stereo_event_graph_creation(neighbors, sequence_folder_path = "/media/enrico/Enrico_Datasets/dsec/train/interlaken_00_c"):
    stereo_event_iter = iter(
        DsecStereoEventIterator(
            sequence_folder_path = sequence_folder_path
        )
    )

    set = EventSet(
        time_window_us=100
    )

    for event in stereo_event_iter:
        if event.left_right == "left":
            set.add(event)
        else:
            graph = EventGraph(event, set, neighbors = neighbors)


def stereo_event_graph_creation_with_gui(time_window_us, neighbors, sequence_folder_path = "/media/enrico/Enrico_Datasets/dsec/train/interlaken_00_c"):
    back_ground = np.ones((HEIGHT, WIDTH*2, 3)) * 255
    image = np.ones((HEIGHT, WIDTH*2, 3)) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # Black
    text_thickness = 2

    processed_left_events = 0
    processed_right_events = 0

    def draw(event_grap : EventGraph):
        image[:, :] = back_ground[:, :]
        def draw_event_graph(event_graph : EventGraph):
            if event_grap.vertices is not None:
                if event_graph.event.p:
                    image[event_graph.event.y, event_graph.event.x] = (255, 0, 0)
                else:
                    image[event_graph.event.y, event_graph.event.x] = (0, 0, 255)

                for i in range(1, len(event_graph.vertices)):
                    cv2.line(
                        image, 
                        (event_graph.event.x + WIDTH, event_graph.event.y),
                        (event_graph.vertices[i][2], event_graph.vertices[i][3]),
                        (0, 125, 125),
                        1
                    )

        draw_event_graph(event_grap)

        # Bottom left corner
        bottom_left_text = f"Left Events = {processed_left_events}"
        bottom_left_pos = (10, HEIGHT - 10)  # 10 pixels from left and bottom
        cv2.putText(image, bottom_left_text, bottom_left_pos, font, font_scale, font_color, text_thickness)

        # Bottom right corner
        bottom_right_text = f"Right Events = {processed_right_events}"
        text_size = cv2.getTextSize(bottom_right_text, font, font_scale, text_thickness)[0]
        bottom_right_pos = (2*WIDTH - text_size[0] - 10, HEIGHT - 10)  # 10 pixels from right and bottom
        cv2.putText(image, bottom_right_text, bottom_right_pos, font, font_scale, font_color, text_thickness)

        cv2.imshow("Stereo Event Graph", image)
        cv2.waitKey(10)
        
        

    stereo_event_iter = iter(
        DsecStereoEventIterator(
            sequence_folder_path = sequence_folder_path
        )
    )

    set = EventSet(
        time_window_us=time_window_us
    )

    last_timestamp = 0
    for event in stereo_event_iter:

        if event.left_right == "left":
            processed_left_events += 1
            set.add(event)
            if event.p:
                back_ground[event.y, event.x] = (255, 0, 0)
            else:
                back_ground[event.y, event.x] = (0, 0, 255)
        else:
            processed_right_events += 1
            if event.p:
                back_ground[event.y, event.x + WIDTH] = (255, 0, 0)
            else:
                back_ground[event.y, event.x + WIDTH] = (0, 0, 255)
            if processed_right_events % 50 == 0:
                draw(EventGraph(event, set, neighbors))

        if event.t - last_timestamp > time_window_us:
            back_ground[:, :] = (255, 255, 255)
            last_timestamp = event.t
        
    