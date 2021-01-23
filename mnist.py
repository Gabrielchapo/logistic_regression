import os
import sys
import time
import numpy as np
import pandas as pd
import pygame.font, pygame.event, pygame.draw
from LogisticRegression import LogisticRegression

try:
    df = pd.read_csv(sys.argv[1])
except:
    exit("Error: Something went wrong with the dataset")
tmp = df["label"].tolist()
Y = np.zeros((len(tmp), 10))
for i,x in enumerate(tmp):
    Y[i][x] = 1
X = df.drop(columns=["label"]).to_numpy()

model = LogisticRegression()

model.fit(X, Y, 0.90, 1000, verbose=1)
model.save()
pygame.init()

black = (0, 0, 0)
gray = (200, 200, 215)
blue_gray = (45, 45, 60)
white = (255, 255, 255)
orange = (255, 128, 10)
bright_orange = (255, 170, 50)
green = (40, 255, 15)

width, height = 914, 612
size = [width, height]
half_width, half_height = width/2, height/2
input_field = (392, 392)
edge_buffer = (10, 50)
scale_size = 17.6

x_coordinates =[]
y_coordinates = []
pixel_colors = []

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Model trained on the MNIST database")

loading_background = pygame.Surface(size)
loading_background.fill(gray)
backdrop = pygame.Surface(size)
backdrop.fill(gray)
background = pygame.Surface(input_field)
background.fill(white)
background2 = pygame.Surface((492, 492))
background2.fill(white)

font = pygame.font.SysFont("Agency FB", 40)
font_small = pygame.font.SysFont("Agency FB", 32)

clock = pygame.time.Clock()

def calculate_image(background):

    scaledBackground = pygame.transform.smoothscale(background, (28, 28))
    image = pygame.surfarray.array3d(scaledBackground)
    to_predict = 253 - image
    image = abs(1-image/253)
    image = np.mean(image, 2)
    to_predict = np.mean(to_predict, 2)
    pixelate(image)
    pixelate(to_predict)
    to_predict = to_predict.transpose()
    to_predict = to_predict.ravel()
    image = image.transpose()
    image = image.ravel()
    return image, to_predict

def calculate_prediction(input_draw):

    reshaped_input = input_draw.reshape(1,784)
    prediction = model.predict(reshaped_input)
    pred = np.argmax(prediction)
    display_prediction(pred)

def display_prediction(prediction):

    display_prediction = "Prediction: %s" %(prediction)
    font = pygame.font.SysFont("Agency FB", 36)
    initialize_prediction = font.render(display_prediction, 1, (white))
    pygame.draw.rect(screen, orange, (edge_buffer[0], input_field[1] + edge_buffer[1] + 10, input_field[0], 90))
    screen.blit(initialize_prediction, (edge_buffer[0] + 5, input_field[1] + edge_buffer[1] + 20))

def pixelate(image):

    size = 28
    image = image.ravel()
    image = (255-image*255)
    for column in range(size):
        for row in range(size):
            # 0 - size**2
            index = row*size + column
            base_rgb = int(image[index])

            x_coordinates.append(row)
            y_coordinates.append(column)
            pixel_colors.append(base_rgb)


def scanner():

    changeY = 0
    speed = 0.003
    coordinate_x = 1
    coordinate_y = 1
    px = 1

    for x in range(int(input_field[0]/2)):

        screen.blit(background,(edge_buffer))
        pygame.draw.rect(screen, green, (edge_buffer[0], edge_buffer[1] + changeY, input_field[0], 5))
        changeY += 2

        if changeY % 14 == 0:
            if (changeY/14) % 2 == 0:

                coordinate_x += 1
                coordinate_y += 1
                px += 1
            else:
                coordinate_x -= 1
                coordinate_y -= 1
                px -= 1
            for i in range(14):
                gray_scaled = (pixel_colors[px], pixel_colors[px], pixel_colors[px])
                pygame.draw.rect(screen, gray_scaled, (x_coordinates[coordinate_x]*scale_size + input_field[0] + 2*edge_buffer[0],
                y_coordinates[coordinate_y]*scale_size + edge_buffer[1], scale_size, scale_size))
                coordinate_x += 2
                coordinate_y += 2
                px += 2

        pygame.draw.rect(screen, gray, (edge_buffer[0], edge_buffer[1] + input_field[1], input_field[0], 10))
        time.sleep(speed)
        pygame.display.flip()
    coordinate_x = 783
    coordinate_y = 783
    px = 783

    for x in range(int(input_field[0]/2)):

        screen.blit(background,(edge_buffer))
        pygame.draw.rect(screen, green, (edge_buffer[0], edge_buffer[1] + changeY, input_field[0], 5))
        changeY -= 2

        if changeY % 14 == 0:
            if (changeY/14) % 2 == 0:
                coordinate_x += 1
                coordinate_y += 1
                px += 1
            else:
                coordinate_x -= 1
                coordinate_y -= 1
                px -= 1
            for i in range(14):
                gray_scaled = (pixel_colors[px], pixel_colors[px], pixel_colors[px])
                pygame.draw.rect(screen, gray_scaled, (x_coordinates[coordinate_x]*scale_size + input_field[0] + 2*edge_buffer[0],
                y_coordinates[coordinate_y]*scale_size + edge_buffer[1], scale_size, scale_size))
                coordinate_x -= 2
                coordinate_y -= 2
                px -= 2

        pygame.draw.rect(screen, gray, (edge_buffer[0], edge_buffer[1] + input_field[1], input_field[0], 10))
        time.sleep(speed)
        pygame.display.flip()

    x_coordinates[:] = []
    y_coordinates[:] = []
    pixel_colors[:] = []

def create_button(btn_label, surface, color, new_color, locationX, locationY, width, height):

    initialize_btn_label = font_small.render(btn_label, 1, white)
    initialize_btn_label_dim = initialize_btn_label.get_rect().width, initialize_btn_label.get_rect().height
    pygame.draw.rect(surface, color, (locationX, locationY, width, height))
    mouse = pygame.mouse.get_pos()

    if locationX + width > mouse[0] > locationX and locationY + height> mouse[1] > locationY:
        pygame.draw.rect(surface, new_color, (locationX, locationY, width, height))

        if pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Classify":
            image, to_predict = calculate_image(background)
            scanner()
            calculate_prediction(to_predict)
        elif pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Clear":
            background.fill(white)
            screen.blit(background2, (2*edge_buffer[0] + input_field[0], edge_buffer[1]))
            display_prediction('Unknown')
    else:
        pygame.draw.rect(surface, color, (locationX, locationY, width, height))

    surface.blit(initialize_btn_label, (locationX + width/2 - initialize_btn_label_dim[0]/2,
        locationY + height/2 - initialize_btn_label_dim[1]/2))

def draw_line(surface, color, start, end, radius):
    
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start[0]+i/distance*dx)
        y = int(start[1]+i/distance*dy)
        pygame.draw.circle(surface, color, (x - edge_buffer[0], y - edge_buffer[1]), radius)

def draw_interface():

    screen.fill(gray)
    display_prediction('Unknown')

    label_input = "Input"
    label_pixelated = "Pixelated"
    label_mnist = "Model trained on the MNIST database"

    initialize_label_input = font.render(label_input, 1, black)
    initialize_label_pixelated = font.render(label_pixelated, 1, black)
    initialize_label_mnist = font.render(label_mnist, 1, white)

    screen.blit(initialize_label_input, (edge_buffer[0] + 10, 10))
    screen.blit(initialize_label_pixelated, (input_field[0] + 2*edge_buffer[0] + 10, 10))
    pygame.draw.rect(screen, blue_gray, (edge_buffer[0], size[1] - 60, size[0] - 2*edge_buffer[0], 50))
    screen.blit(initialize_label_mnist, (size[0]/2 - initialize_label_mnist.get_rect().width/2, size[1] - 50))
    screen.blit(background2, (2*edge_buffer[0] + input_field[0], edge_buffer[1]))

    pygame.display.flip()

def main():

    last_pos = (0, 0)
    line_width = 9

    draw_interface()

    image = None
    continue_on = True
    while continue_on:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                continue_on = False

            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    draw_line(background, black, event.pos, last_pos, line_width)
                last_pos = event.pos

            screen.blit(background, (edge_buffer[0], edge_buffer[1]))
            create_button("Classify", screen, orange, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 10, 120, 45)
            create_button("Clear", screen, orange, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 55, 120, 45)
            pygame.display.flip()

if __name__ == "__main__":
    main()