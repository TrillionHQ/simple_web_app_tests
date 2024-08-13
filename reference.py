import cv2
import pygame
import math
import mediapipe as mp
from mediapipe.tasks.python import vision, core
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
import numpy as np
from PIL import Image
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from scipy.spatial.transform import Rotation as R


hand_edges = (
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

def calculate_finger_rotation(points_world, finger_name='INDEX'):
    if finger_name == 'INDEX':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 5, 6, 5, 13
    elif finger_name == 'MIDDLE':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 9, 10, 5, 13
    elif finger_name == 'RING':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 13, 14, 5, 13
    elif finger_name == 'PINKY':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 17, 18, 5, 13
    else:  # == THUMB
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 2, 3, 5, 9

    world_mcp1 = points_world[idx_mcp1][:3] / points_world[idx_mcp1][3]
    world_mcp2 = points_world[idx_mcp2][:3] / points_world [idx_mcp2][3]
    world_pip = points_world[idx_pip][:3] / points_world [idx_pip][3]
    world_mcp = points_world[idx_mcp][:3] / points_world [idx_mcp][3]

    world_base_vector = world_mcp2 - world_mcp1
    world_finger_vector = world_pip - world_mcp

    world_base_vector_normalized = world_base_vector / np.linalg.norm(world_base_vector)
    world_finger_vector_normalized = world_finger_vector / np.linalg.norm(world_finger_vector)

    rotation_matrix_world = R.from_rotvec(np.cross(world_base_vector_normalized, world_finger_vector_normalized)).as_matrix()
    world_quaternion = R.from_matrix(rotation_matrix_world).as_quat()

    return world_quaternion

def calculate_finger_rotation_cone(points_world, finger_name='INDEX'):
    if finger_name == 'INDEX':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 5, 6, 5, 13
    elif finger_name == 'MIDDLE':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 9, 10, 5, 13
    elif finger_name == 'RING':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 13, 14, 5, 13
    elif finger_name == 'PINKY':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 17, 18, 5, 13
    else:  # == THUMB
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 2, 3, 5, 9

    world_mcp1 = points_world[idx_mcp1][:3] / points_world[idx_mcp1][3]
    world_mcp2 = points_world[idx_mcp2][:3] / points_world[idx_mcp2][3]
    world_pip = points_world[idx_pip][:3] / points_world[idx_pip][3]
    world_mcp = points_world[idx_mcp][:3] / points_world[idx_mcp][3]

    world_base_vector = world_mcp2 - world_mcp1
    world_finger_vector = world_pip - world_mcp

    world_base_vector_normalized = world_base_vector / np.linalg.norm(world_base_vector)
    world_finger_vector_normalized = world_finger_vector / np.linalg.norm(world_finger_vector)

    rotation_matrix_world = np.eye(3)
    rotation_matrix_world[:, 2] = world_finger_vector_normalized
    rotation_matrix_world[:, 0] = np.cross(world_finger_vector_normalized, world_base_vector_normalized)
    rotation_matrix_world[:, 0] /= np.linalg.norm(rotation_matrix_world[:, 0])
    rotation_matrix_world[:, 1] = np.cross(rotation_matrix_world[:, 2], rotation_matrix_world[:, 0])

    world_quaternion = R.from_matrix(rotation_matrix_world).as_quat()

    return world_quaternion

def calculate_finger_rotation_norm(points_norm, finger_name='INDEX'):
    if finger_name == 'INDEX':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 5, 6, 5, 13
    elif finger_name == 'MIDDLE':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 9, 10, 5, 13
    elif finger_name == 'RING':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 13, 14, 5, 13
    elif finger_name == 'PINKY':
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 17, 18, 5, 13
    else:  # == THUMB
        idx_mcp, idx_pip, idx_mcp1, idx_mcp2 = 2, 3, 5, 9

    norm_mcp1 = points_norm[idx_mcp1]
    norm_mcp2 = points_norm[idx_mcp2]
    norm_pip = points_norm[idx_pip]
    norm_mcp = points_norm[idx_mcp]

    norm_base_vector = norm_mcp2 - norm_mcp1
    norm_finger_vector = norm_pip - norm_mcp

    norm_base_vector_normalized = norm_base_vector / np.linalg.norm(norm_base_vector)
    norm_finger_vector_normalized = norm_finger_vector / np.linalg.norm(norm_finger_vector)

    rotation_matrix_norm = R.from_rotvec(np.cross(norm_base_vector_normalized, norm_finger_vector_normalized)).as_matrix()
    norm_quaternion = R.from_matrix(rotation_matrix_norm).as_quat()

    return norm_quaternion

def draw_hand(world_points): 
    glLineWidth(5)
    glLoadIdentity()
    glBegin(GL_LINES)
    for edge in hand_edges:
        for vertex in edge:
            p = world_points[vertex]
            glVertex3fv((-p[0], p[1], p[2]))
    glEnd()
    for p in world_points:
        glPushAttrib(GL_LIGHTING_BIT)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0, 1, 0, 0.5])
        glLoadIdentity()
        glTranslatef(-p[0], p[1], p[2])
        glutSolidSphere(0.01 / 2, 16, 16)
        glPopAttrib()

def draw_cone_with_quaternion(rotation_quaternion, position = (0, 0, -0.5)):
    glPushMatrix()
    rotation_quaternion[0] *= -1
    rotation_quaternion[3] *= -1
    rotation = R.from_quat(rotation_quaternion)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    glTranslatef(*position)
    glMultMatrixf(rotation_matrix.T)
    glutSolidCone(0.05, 0.15, 16, 16)
    glPopMatrix()

class ImageLoader:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.width = 0
        self.height = 0
        self.img_data = 0
        self.Texture = glGenTextures(1)

    def load(self, image: cv2.Mat):
        im = image
        tx_image = cv2.flip(im, 0)
        tx_image = Image.fromarray(tx_image)
        self.width = tx_image.size[0]
        self.height = tx_image.size[1]
        self.img_data = tx_image.tobytes('raw', 'BGRX', 0, -1)

        glBindTexture(GL_TEXTURE_2D, self.Texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img_data)

    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(self.x, self.y, 0)
        glEnable(GL_TEXTURE_2D)  
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(self.width, 0)
        glTexCoord2f(1, 1)
        glVertex2f(self.width, self.height)
        glTexCoord2f(0, 1)
        glVertex2f(0, self.height)
        glEnd()
        glDisable(GL_TEXTURE_2D)


ball_pos_start = [0, 0, -0.4]
ball_pos = list(ball_pos_start)
ball_grabbed = False

def landmarks_to_array(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# Initialize hand landmarker options and create the hand landmarker
BaseOptions = mp.tasks.BaseOptions
base_options = BaseOptions(model_asset_path='./hypotheses/hand_landmarker.task')
options = HandLandmarkerOptions(base_options=base_options,
                                running_mode=vision.RunningMode.VIDEO,                     
                                min_hand_detection_confidence=0.7,
                                min_tracking_confidence=0.7)
hand_landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture('test.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
pygame.init()
display = (width, height)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
glutInit()

im_loader = ImageLoader(0, 0)

draw_mediapipe = False
frame_timestamp_ms = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                draw_mediapipe = not draw_mediapipe
                print(f'toggling draw media pipe now: {draw_mediapipe}')
            if event.key == pygame.K_b:
                ball_pos = list(ball_pos_start)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, height, 0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
    glEnable(GL_LIGHT0)

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # Use the new API for hand landmark detection
    hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    frame_timestamp_ms += 1  # Increase the timestamp for the next frame

    # Convert the detected results to match the previous structure
    if hand_landmarker_result.hand_landmarks:
        multi_hand_landmarks = [landmark for landmark in hand_landmarker_result.hand_landmarks]
        multi_hand_world_landmarks = [landmark for landmark in hand_landmarker_result.hand_world_landmarks]

    # Draw the hand annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_height, frame_width, channels = image.shape
    focal_length = frame_width * 1.6
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    distortion = np.zeros((4, 1))
    fov_x = np.rad2deg(2 * np.arctan2(focal_length, 2 * focal_length))

    model_points_total = []
    world_points_total = []

    
    if hand_landmarker_result.hand_landmarks:
        for i in range(len(hand_landmarker_result.hand_landmarks)):
            hand_landmarks = hand_landmarker_result.hand_landmarks[i]
            world_landmarks = hand_landmarker_result.hand_world_landmarks[i]
            print(hand_landmarker_result.handedness)
            
            if draw_mediapipe:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            
            model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks])
            image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks])
            success, rvecs, tvecs, = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_SQPNP
            )

            transformation = np.eye(4)
            transformation[0:3, 3] = tvecs.squeeze()
            model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)
            model_points_total.append(model_points_hom)
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
            world_points_total.append(world_points)

    # Draw the video frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, height, 0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    
    im_loader.load(image)
    glColor3f(1, 1, 1)
    im_loader.draw()
    
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glClear(GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov_x, (display[0]/display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    if len(world_points_total) > 0:
        glLoadIdentity()
        for i in range(len(world_points_total)):
            #index_position = [-world_points_total[i][6][0], world_points_total[i][6][1], world_points_total[i][6][2]]
            index_quaternion = calculate_finger_rotation_cone(world_points_total[i], 'INDEX')
            glLoadIdentity()
            draw_cone_with_quaternion(rotation_quaternion = index_quaternion)
            

    pygame.display.flip()
