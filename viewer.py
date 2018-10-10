from mentalitystorm.observe import ImageViewer

view_latent1 = ImageViewer('latent1', (320, 480), channels=[0, 1, 2])
view_latent2 = ImageViewer('latent2', (320, 480), channels=[3])
view_input1 = ImageViewer('input1', (320, 480), channels=[0, 1, 2])
view_input2 = ImageViewer('input2', (320, 480), channels=[3, 4, 5])


def view_image(model, input, output):
    view_input1.update(input[0].data)
    view_input2.update(input[0].data)
    view_latent1.update(output[0].data)
    view_latent2.update(output[0].data)


decode_viewer1 = ImageViewer('decoded1', (320, 480), channels=[0, 1, 2])
decode_viewer2 = ImageViewer('decoded2', (320, 480), channels=[3])


def view_decode(model, input, output):
    image = model.decode(output)
    decode_viewer1.update(image)
    decode_viewer2.update(image)