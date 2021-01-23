import sys
import numpy as np
import pygame.font, pygame.event, pygame.draw
from LogisticRegression import LogisticRegression

model = LogisticRegression()
model.load()

pygame.init()

black = (0, 0, 0)
gray = (200, 200, 215)
blue_gray = (45, 45, 60)
white = (255, 255, 255)
orange = (255, 128, 10)
bright_orange = (255, 170, 50)

width, height = 412, 512
size = [width, height]
input_field = (392, 392)
edge_buffer = (10, 10)

x_coordinates =[]
y_coordinates = []
pixel_colors = []

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Model trained on the MNIST database")

backdrop = pygame.Surface(size)
backdrop.fill(gray)
background = pygame.Surface(input_field)
background.fill(white)

font = pygame.font.SysFont("Agency FB", 40)
font_small = pygame.font.SysFont("Agency FB", 32)

def calculate_image(background):

    scaledBackground = pygame.transform.smoothscale(background, (28, 28))
    image = pygame.surfarray.array3d(scaledBackground)
    to_predict = 253 - image
    image = abs(1 - image / 253)
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
            index = row*size + column
            base_rgb = int(image[index])
            x_coordinates.append(row)
            y_coordinates.append(column)
            pixel_colors.append(base_rgb)

def create_button(btn_label, surface, color, new_color, locationX, locationY, width, height):

    initialize_btn_label = font_small.render(btn_label, 1, white)
    initialize_btn_label_dim = initialize_btn_label.get_rect().width, initialize_btn_label.get_rect().height
    pygame.draw.rect(surface, color, (locationX, locationY, width, height))
    mouse = pygame.mouse.get_pos()

    if locationX + width > mouse[0] > locationX and locationY + height> mouse[1] > locationY:
        pygame.draw.rect(surface, new_color, (locationX, locationY, width, height))

        if pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Classify":
            image, to_predict = calculate_image(background)
            calculate_prediction(to_predict)
        elif pygame.mouse.get_pressed() == (1, 0, 0) and btn_label == "Clear":
            background.fill(white)
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

def main():

    last_pos = (0, 0)
    line_width = 9

    screen.fill(gray)
    display_prediction('Unknown')
    pygame.display.flip()

    image = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    draw_line(background, black, event.pos, last_pos, line_width)
                last_pos = event.pos
            screen.blit(background, (edge_buffer[0], edge_buffer[1]))
            create_button("Classify", screen, blue_gray, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 10, 120, 45)
            create_button("Clear", screen, blue_gray, bright_orange, input_field[0] - 110, input_field[1] + edge_buffer[1] + 55, 120, 45)
            pygame.display.flip()

if __name__ == "__main__":
    main()