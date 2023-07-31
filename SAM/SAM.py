from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import streamlit as st
import numpy as np
import cv2 as cv
import torch
import os


class SamPredictorAV:

    def __init__(self, image=None):

        if image is not None:
            self.image = image
        else:
            self.image = cv.imread("SAM/room_test.jpg")

        self.masks = None
        self.box = np.array([])
        self.streamlit_layout()

    def get_all_masks(self):
        CHECKPOINT_PATH = os.path.join("SAM", "weights", "sam_vit_h_4b8939.pth")
        print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

        mask_generator = SamAutomaticMaskGenerator(sam)
        image_bgr = cv.imread("room_test.jpg")
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        sam_result = mask_generator.generate(image_rgb)
        mask_annotator = sv.MaskAnnotator()

        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        # sv.plot_images_grid(
        #     images=[image_bgr, annotated_image],
        #     grid_size=(1, 2),
        #     titles=['source image', 'segmented image']
        # )

        self.masks = [
            mask['segmentation']
            for mask
            in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        ]

        return annotated_image

    def get_masks_from_box(self):
        CHECKPOINT_PATH = os.path.join("SAM", "weights", "sam_vit_h_4b8939.pth")
        print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

        mask_predictor = SamPredictor(sam)

        image_bgr = self.image
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        mask_predictor.set_image(image_rgb)

        masks, scores, logits = mask_predictor.predict(
            box=self.box,
            multimask_output=True
        )

        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]

        source_image = box_annotator.annotate(scene=self.image.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        st.image(source_image)
        st.image(segmented_image)

        for mask in masks:
            numpy_image = np.array(mask, dtype=np.uint8)
            cv2_image = numpy_image * 255
            st.image(cv2_image)

            # st.download_button(
            #     label="Download image",
            #     data=???,
            #     file_name="mask.png",
            #     mime="image/png"
            # )
            st.divider()


        # sv.plot_images_grid(
        #     images=[source_image, segmented_image],
        #     grid_size=(1, 2),
        #     titles=['source image', 'segmented image']
        # )

    @staticmethod
    def draw_rectangle_on_image(image, x, y, width, height, color=(0, 255, 0), thickness=2):
        image_with_rectangle = image.copy()
        cv.rectangle(image_with_rectangle, (x, y), (x + width, y + height), color, thickness)
        return image_with_rectangle

    def streamlit_layout(self):

        if self.image is not None:

            # Get image dimensions for Streamlit display
            image_height, image_width, _ = self.image.shape

            # User inputs for rectangle position and size
            st.sidebar.header("Rectangle Parameters")
            x = st.sidebar.slider("X Coordinate", min_value=0, max_value=image_width, value=(0, image_width))
            y = st.sidebar.slider("Y Coordinate", min_value=0, max_value=image_height, value=(0, image_height))
            width = x[1] - x[0]
            height = y[1] - y[0]

            # Draw rectangle on the image
            image_with_rectangle = self.draw_rectangle_on_image(cv.cvtColor(self.image, cv.COLOR_BGR2RGB), x[0], y[0], width, height)

            # Display the original and modified images
            # st.image(image, caption="Original Image", use_column_width=True)
            st.image(image_with_rectangle, caption="Image with Rectangle", use_column_width=True)

            self.box = np.array([x[0], y[0], width + x[0], height+y[0]])
            print(self.box)
            if st.button('Segment'):
                self.get_masks_from_box()


SamPredictorAV(cv.imread("room_test.jpg"))
