import dill
import cv2
import numpy as np
import pygame as pg
from matplotlib import pyplot as plt

MODEL_PATH = "../models/mnist-86p.pkl"

model = dill.load(open(MODEL_PATH, "rb"))
model.summary()

pg.init()
screen = pg.display.set_mode((280, 280))
pg.display.set_caption("MNIST Example")
clock = pg.time.Clock()

drawing = False
previous_point = None


def process(image):
    image = image[:, :, 0]
    image = cv2.resize(image, (28, 28)) / 255.0

    image = image.T.flatten()

    res = model.forward(image)
    print(np.argmax(res))


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit(0)
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                quit(0)
            elif event.key == pg.K_BACKSPACE:
                screen.fill((0, 0, 0))
            elif event.key == pg.K_RETURN:
                process(pg.surfarray.array3d(screen))
        elif event.type == pg.MOUSEBUTTONDOWN:
            drawing = True
            pg.draw.circle(screen, (255, 255, 255), event.pos, 20)
            previous_point = event.pos
        elif event.type == pg.MOUSEBUTTONUP:
            drawing = False
            previous_point = None
        elif event.type == pg.MOUSEMOTION:
            if drawing:
                pg.draw.line(screen, (255, 255, 255), previous_point, event.pos, 40)
                pg.draw.circle(screen, (255, 255, 255), event.pos, 20)
                pg.draw.circle(screen, (255, 255, 255), previous_point, 20)
                previous_point = event.pos

    pg.display.update()
    clock.tick(60)
