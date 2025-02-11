"""우리가 만들고 싶은 틀"""

from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker

import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json

from inpainting import inpainting_result_output


def clean():

    return None, None, None, None, None, None, [[], []]

def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"]
    )
    
    prompt = {
        "points_coord":click_stack[0],
        "points_mode":click_stack[1],
        "multimask":"True",
    }

    return prompt

def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    
    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
   
    return first_frame, first_frame, first_frame, ""


def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker


def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""


def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    
    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]

    print("Undo!")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask":"True",
        }

        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click( 
                                                      origin_frame=origin_frame, 
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame


def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, evt:gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")

    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        # TODO：add everything positive points
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    # get click prompts for sam to predict mask
    click_prompt = get_click_prompt(click_stack, point)

    # Refine acc to prompt
    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_stack


def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)


    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


def tracking_objects(Seg_Tracker, input_video, frame_num=0):
    print("Start tracking !")
    output_video, output_mask=tracking_objects_in_video(Seg_Tracker, input_video, frame_num)
    
    return output_video, output_mask


def show_inpainting_result(input_video):
    return inpainting_result_output(input_video)

def seg_track_app():

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()

    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">🖌INPAINTERZ🩵</span>
            </div>
            '''
        )

        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row():
            # video input
            with gr.Column(scale=0.5):

                tab_video_input = gr.Tab(label="Video type input")
                with tab_video_input:
                    input_video = gr.Video(label='Input video').style(height=450)
 

                input_first_frame = gr.Image(label='Segment result of first frame',interactive=True).style(height=450)


                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_mode = gr.Radio(
                                    choices=["Positive",  "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    interactive=True)

                        # args for modify and tracking 
                        click_undo_but = gr.Button(
                                    value="Undo",
                                    interactive=True
                                    )
                        
                        click_reset_but = gr.Button(
                                    value="Reset",
                                    interactive=True
                                            )

                
                    with gr.Row():
                        with gr.Column(scale=0.5):
                            
                            with gr.Accordion("SegTracker Args", open=False):
                                points_per_side = gr.Number(
                                    label = "points_per_side (1~100)",
                                    value = 50,
                                    interactive=True
                                )

                                sam_gap = gr.Number(
                                    label='sam_gap (1~9999)',
                                    value = 2023,
                                    interactive=True
                                )

                                max_obj_num = gr.Number(
                                    label='max_obj_num (50~300)',
                                    value = 255,
                                    interactive=True
                                )
                        
                        with gr.Accordion("aot advanced options", open=False):
                            aot_model = gr.Dropdown(
                                label="aot_model",
                                choices = [
                                    "r50_deaotl"
                                ],
                                value = "r50_deaotl",
                                interactive=True,
                                )
                            long_term_mem = gr.Number(
                                label="long term memory gap (1~9999)", 
                                value=2023,
                                interactive=True
                                )
                            max_len_long_term = gr.Number(
                                label="max len of long term memory (1~9999)", 
                                value=2023,
                                interactive=True    
                                )
  


                    
                with gr.Column():
                   
                    reset_button = gr.Button(
                        value="Reset",
                        interactive=True,
                        )   
                    track_for_video = gr.Button(
                        value="Start Tracking!",
                        interactive=True,
                        )

            with gr.Column(scale=0.5):
                with gr.Tab(label='Masked video output'):
                    output_video = gr.Video(label='Output masked video').style(height=450)
                
                inpainting_button = gr.Button(
                    value="Start Inpainting!",
                    interactive=True,
                    )
                
                with gr.Tab(label='inpainting video output'):
                    inpainting_video = gr.Video(label='Output inpainting video').style(height=450)
                
                output_mask = gr.File(label="Output masks download")

                                
    ##########################################################
    ######################  back-end #########################
    ##########################################################

        # listen to the input_video to get the first frame of video
        input_video.change(
            fn=get_meta_from_video,
            inputs=[
                input_video
            ],
            outputs=[
                input_first_frame, 
                origin_frame, 
                # drawing_board, 
                # grounding_caption
            ]
        )

        
        #-------------- Input compont -------------
        tab_video_input.select(
            fn = clean,
            inputs=[],
            outputs=[
                input_video,
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                # drawing_board,
                click_stack,
            ]
        )

        # ------------------- Interactive component -----------------

        # listen to the tab to init SegTracker
        
        tab_click.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack, 
                # grounding_caption
            ],
            queue=False,
        )

        
        # # Interactively modify the mask acc click
        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker, 
                origin_frame, 
                point_mode, 
                click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ]
        )

        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
            ],
            outputs=[
                output_video, output_mask
            ]
        )



        # ----------------- Reset and Undo ---------------------------

        # Rest 
        reset_button.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ],
            queue=False,
            show_progress=False
        ) 


        click_reset_but.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame
            ],
            outputs=[
                Seg_Tracker, input_first_frame, click_stack
            ],
            queue=False,
            show_progress=False
        ) 


        # Undo click
        click_undo_but.click(
            fn = undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker, origin_frame, click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
               Seg_Tracker, input_first_frame, click_stack
            ]
        )
        
        inpainting_button.click(
            fn = inpainting_result_output,
            inputs=[
                input_video
            ],
            outputs=[
                inpainting_video
            ]
        )

        with gr.Tab(label='Video example'):
                gr.Examples(
                    examples=[
                        os.path.join(os.path.dirname(__file__), "sam/assets", "MountainBiking.mp4"),
                        os.path.join(os.path.dirname(__file__), "sam/assets", "LM350h.mp4"),
                        os.path.join(os.path.dirname(__file__), "sam/assets", "MG4car.mp4"),
                        os.path.join(os.path.dirname(__file__), "sam/assets", "logo.mp4"),
                        ],
                    inputs=[input_video],
                )
     
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)


if __name__ == "__main__":
    seg_track_app()