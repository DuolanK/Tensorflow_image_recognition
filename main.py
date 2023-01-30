from pixellib.instance import instance_segmentation

def object_detection():
    segment_image = instance_segmentation()
    segment_image.load_model("/home/brohaj/PycharmProjects/imagerecognition/mask_rcnn_coco.h5")

    segment_image.segmentImage(
        image_path="2.jpg",
        show_bboxes=True,
        output_image_name="output.jpg"
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    object_detection()
