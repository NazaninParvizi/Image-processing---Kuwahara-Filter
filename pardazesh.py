
import cv2 as cv
import numpy as np



def main():


    img = cv.imread('image_man/original.png')
    
    #gray
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = np.array([50, 0, 0])
    upper_red = np.array([200, 100, 255])
    mask = cv.inRange(hsv, lower_red, upper_red)
    cv.imshow('mask', mask)
    
    Nimage = kuwahara(img, 1)
    cv.imshow('Orginal Photo', img)
    cv.imshow('After Kuwahara Filter', Nimage)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    

def kuwahara(image, kernel=5):

    length, breadth, channel = image.shape[0], image.shape[1], image.shape[2]

    a = int((kernel - 1) / 2)
    a = a if a >= 2 else 2

    image = np.pad(image, ((a, a), (a, a), (0, 0)), "edge")

    average, variance = cv.integral2(image)
    average = (average[:-a - 1, :-a - 1] + average[a + 1:, a + 1:] -
               average[a + 1:, :-a - 1] - average[:-a - 1, a + 1:]) / (a + 1) ** 2
    variance = ((variance[:-a - 1, :-a - 1] + variance[a + 1:, a + 1:]
                 - variance[a + 1:, :-a - 1] - variance[:-a - 1, a + 1:]) / (a + 1) ** 2 - average ** 2).sum(axis=2)

    def filter(i, j):
        return np.array([
            average[i, j], average[i + a, j], average[i, j + a], average[i + a, j + a] 
        ])[(np.array([ variance[i, j], variance[i + a, j], variance[i, j + a], variance[i + a, j + a]
                      ]).argmin(axis=0).flatten(), j.flatten(),i.flatten())].reshape(breadth, length, channel).transpose(1, 0, 2)

    filterimg = filter(*np.meshgrid(np.arange(length), np.arange(breadth)))

    filterimg = filterimg.astype(image.dtype)
    return filterimg




if __name__ == '__main__':
    main()