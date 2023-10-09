import cv2
import numpy as np
import random
import time
import dlib
#import f_detector
import imutils
#from eye_blink_detection import *
from eye_key_funcs import *

# upload face/eyes predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

width , height = 800, 500 #pixels
# The first value represents the horizontal offset and the second value represents the vertical offset
offset = (100, 80) # pixel offset (x, y) of pic coordinates

#resize_eye_frame = 4.5 # scaling factor for window's size
#resize_frame = 0.3 # scaling factor for window's size
# # ------------------------------------
resize_eye_frame = 5  
resize_frame = 0.3

cap = cv2.VideoCapture(1)

# Load your cockpit picture
cockpit = cv2.imread('cockpit.jpg')


size_screen = (320 , 240)
#size_screen = (cap.get(700), cap.get(1200))

#for blink detection


def adjust_frame(frame):
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.flip(frame, 1)
    return frame

# Function to load and resize an image to match the screen size
def load_and_resize_image(image_path, screen_size):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (int(screen_size[1]), int(screen_size[0])))
    return resized_image

# Function to capture the screen size
def get_screen_size():
    # On most systems, you can use the following to get the screen size
    screen_size = (320, 240)  # Replace with your screen resolution
    return screen_size


def __init__(self):
        # cargar modelo para detecction frontal de rostros
        self.detector_faces = dlib.get_frontal_face_detector()
        # cargar modelo para deteccion de puntos de ojos
        self.predictor_eyes = dlib.shape_predictor(cfg.eye_landmarks)
#get eye coordinate
def get_eye_coordinates(landmarks, points):

    x_left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    x_right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)

    y_top = half_point(landmarks.part(points[1]), landmarks.part(points[2]))
    y_bottom = half_point(landmarks.part(points[5]), landmarks.part(points[4]))

    return x_left, x_right, y_top, y_bottom

#calibration
corners = [(offset), (width+offset[0], height+offset[1]), 
            (width+offset[0], offset[1]), (offset[0], height+offset[1])]

corner = 0
calibration_cut = []


# Main function
def main():
    screen_size = get_screen_size()
    image_path = 'cockpit.jpg'  # Replace with the path to your image
    resized_image = load_and_resize_image(image_path, screen_size)
    #cv2.imshow ('cockpit', resized_image)
    # Initialize a blank calibration page with the same size as the cockpit image
    #cockpit_image = np.zeros_like(cockpit)
    #cv2.imshow ("Cockpit", cockpit)

    # Create a VideoCapture object for your webcam
    cap = cv2.VideoCapture(1)  # Change the camera index if needed
    corner = 0
    while(corner<4):
         # calibration of 4 corners

        ret, frame = cap.read()   # Capture frame
        frame = adjust_frame(frame)  # rotate / flip

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

    # messages for calibration
        cv2.putText(cockpit, 'calibration: look at the circle and blink', tuple((np.array(size_screen)/7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255, 0, 0), 3)
        cv2.circle(cockpit, corners[corner], 40, (0, 255, 0), -1)
        


    # detect faces in frame
        faces = detector(gray_scale_frame)
        if len(faces)> 1:
            print('please avoid multiple faces.')
            sys.exit()

        for face in faces:
            display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

            landmarks = predictor(gray_scale_frame, face) # find points in face
            display_face_points(frame, landmarks, [0, 68], color='yellow') # draw face points

            # get position of right eye and display lines
            right_eye_coordinates= get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            display_eye_lines(frame, right_eye_coordinates, 'green')

    # define the coordinates of the pupil from the centroid of the right eye
            landmarks = predictor(gray_scale_frame, face)
            right_eye_coordinates= get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

            if is_blinking(right_eye_coordinates):
                calibration_cut.append(pupil_coordinates)

        # visualize message
                cv2.putText(cockpit, 'ok',
                            tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 5)
            # to avoid is_blinking=True in the next frame
                time.sleep(2)
                corner = corner + 1

        print(calibration_cut, '    len: ', len(calibration_cut))
        show_window('projection', cockpit)
        show_window('frame', cv2.resize(frame,  (320, 240)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.destroyAllWindows()

    # Release the webcam and close OpenCV windows
    #cap.release()
    cv2.destroyAllWindows()


# find limits & offsets for the calibrated frame-cut
    x_cut_min, x_cut_max, y_cut_min, y_cut_max = find_cut_limits(calibration_cut)
    offset_calibrated_cut = [ x_cut_min, y_cut_min ]
    
    print('message for user')
    cv2.putText(cockpit, 'calibration done.',
                tuple((np.array(size_screen)/5).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1.4,(255, 255, 255), 3)
    show_window('projection', cockpit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    while(True):
        #star_time = time.time()
        ret, frame = cap.read()   # Capture frame
        frame = adjust_frame(frame)  # rotate / flip

        cut_frame = np.copy(frame[y_cut_min:y_cut_max, x_cut_min:x_cut_max, :])

        # make & display on frame the keyboard
        #keyboard_page = make_black_page(size = size_screen)
        #dysplay_keyboard(img = keyboard_page, keys = key_points)
        #text_page = make_white_page(size = (200, 800))

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

        faces = detector(gray_scale_frame)  # detect faces in frame
        if len(faces)> 1:
            print('please avoid multiple faces..')
            sys.exit()
           

        for face in faces:
            display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

            landmarks = predictor(gray_scale_frame, face) # find points in face
            display_face_points(frame, landmarks, [0, 68], color='red') # draw face points

            # get position of right eye and display lines
            right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
            display_eye_lines(frame, right_eye_coordinates, 'green')

    # define the coordinates of the pupil from the centroid of the right eye
        pupil_on_frame = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

        # work on the calbrated cut-frame
        pupil_on_cut = np.array([pupil_on_frame[0] - offset_calibrated_cut[0], pupil_on_frame[1] - offset_calibrated_cut[1]])
        cv2.circle(cut_frame, (pupil_on_cut[0], pupil_on_cut[1]), int(take_radius_eye(right_eye_coordinates)/1.5), (255, 0, 0), 3)

        if pupil_on_cut_valid(pupil_on_cut, cut_frame):

            pupil_on_cockpit = project_on_page(img_from = cut_frame[:,:, 0], # needs a 2D image for the 2D shape
                                                img_to = cockpit[:,:, 0], # needs a 2D image for the 2D shape
                                                point = pupil_on_cut)

            # draw circle at pupil_on_keyboard on the keyboard
            cv2.circle(cockpit, (pupil_on_cockpit[0], pupil_on_cockpit[1]), 40, (0, 255, 0), 3)

            '''if is_blinking(right_eye_coordinates):

                pressed_key = identify_key(key_points = key_points, coordinate_X = pupil_on_keyboard[1], coordinate_Y = pupil_on_keyboard[0])

                if pressed_key:
                    if pressed_key=='del':
                        string_to_write = string_to_write[: -1]
                    else:
                        string_to_write = string_to_write + pressed_key

                time.sleep(0.3) # to avoid is_blinking=True in the next frame

        # print on screen the string
        cv2.putText(text_page, string_to_write,
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)

        else:

            img_post = frame 
        # visualizacion 
            end_time = time.time() - star_time    
            FPS = 1/end_time'''

        # visualize windows
        cockpit1 = cv2.imread('cockpit1.jpg')
        show_window('cockpit1', cockpit1)
        show_window('cockpit', cockpit)
        #show_window('cockpit1', cockpit1)
        show_window('frame', cv2.resize(frame, (int(frame.shape[1] *resize_frame), int(frame.shape[0] *resize_frame))))
        show_window('cut_frame', cv2.resize(cut_frame, (int(cut_frame.shape[1] *resize_eye_frame), int(cut_frame.shape[0] *resize_eye_frame))))
        #show_window('text_page', text_page)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('tracked_eyes_report.jpg', cockpit)

            #show_window("cockpit", cockpit)
            break

# -------------------------------------------------------------------

    #show cockpit screen after tracking
            #cv2.imwrite('tracked_eyes_report.jpg', cockpit)

    shut_off(cap)
    


    #show_window('cockpit', cockpit) # Shut camera / windows off


if __name__ == "__main__":
    main()
