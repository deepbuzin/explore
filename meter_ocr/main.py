import cv2
import pytesseract
from glob import glob
from os.path import join

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'

    for img_path in glob(join('data', '*', '*.png')):
        img = cv2.imread(img_path)

        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow('meter', img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(pytesseract.image_to_string(img, lang='bad_rus'))

        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        else:
            pass

    cv2.destroyAllWindows()

